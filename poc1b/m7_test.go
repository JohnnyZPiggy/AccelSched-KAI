package main

// M7 gate test. Order (hard-gated):
//   TrustBoundary  - planning-cohort provenance + the separately versioned P2H-PC-1 receiver. If it fails,
//                    cohort arms are NOT_RUN (TRUST_BOUNDARY_BLOCKED / NATIVE_NONZERO_LP_NOT_RUN); Stage A still runs.
//   Native         - 31 declared M6 scenes x declared arms: the Native-OFF / PGFP baselines, the lambda grid,
//                    accurate / clairvoyant forecast diagnostics (Stage A) and the legitimate-cohort LP arms (B1).
//   LatencyB2      - independent producer cold/warm experiment (also runs if B1 is blocked).

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"testing"
	"time"

	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/node_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/framework"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/test_utils"
)

func m7Scenes() []m6Scene {
	out := []m6Scene{}
	seeds := m6SeedsPerRegime
	if v, err := strconv.Atoi(os.Getenv("M7_DEV_SEEDS")); err == nil && v > 0 && v < seeds { // development aid only; never set for evidence runs
		seeds = v
	}
	for _, r := range m6Regimes {
		for seed := 0; seed < seeds; seed++ {
			out = append(out, m6MakeScene(r, seed))
		}
	}
	return append(out, m6MechanismScenes()...)
}

// Arms are declared here, before any run. lambda grid: 0, 5, 20, 40 (M6), 80. Equal-PGFP-coefficient control:
// pgfp_l20 (AF coefficient 20*1) versus lp_mix_cohort (AF coefficient 40*(1-.5)=20, plus the price term).
func m7Cfgs() []m7Cfg {
	return []m7Cfg{
		{Name: "native_off", Off: true, Transport: "cold"},
		{Name: "pgfp_l40", Alpha: 0, Lambda: 40, Kind: "p2h", Transport: "cold"},
		{Name: "pgfp_l40_worker", Alpha: 0, Lambda: 40, Kind: "p2h", Transport: "worker"},
		{Name: "lp_mix_single", Alpha: .5, Lambda: 40, Kind: "p2h", Transport: "worker"},
		{Name: "pgfp_l0", Alpha: 0, Lambda: 0, Kind: "p2h", Transport: "worker"},
		{Name: "pgfp_l5", Alpha: 0, Lambda: 5, Kind: "p2h", Transport: "worker"},
		{Name: "pgfp_l20", Alpha: 0, Lambda: 20, Kind: "p2h", Transport: "worker"},
		{Name: "pgfp_l80", Alpha: 0, Lambda: 80, Kind: "p2h", Transport: "worker"},
		{Name: "pgfp_l40_accurate_forecast", Alpha: 0, Lambda: 40, Kind: "p2h", Forecast: "accurate", Transport: "worker"},
		{Name: "pgfp_l40_clairvoyant_forecast", Alpha: 0, Lambda: 40, Kind: "p2h", Forecast: "clairvoyant", Transport: "worker", Label: "OFFLINE_ORACLE_INFORMATION_NOT_NATIVE"},
		{Name: "pc_alpha0_l20_cohort", Alpha: 0, Lambda: 20, Kind: "p2hpc", Cohort: "forecast", Transport: "worker"},
		{Name: "lp_mix_cohort", Alpha: .5, Lambda: 40, Kind: "p2hpc", Cohort: "forecast", Transport: "worker"},
		{Name: "lp_mix_cohort_oracle", Alpha: .5, Lambda: 40, Kind: "p2hpc", Cohort: "oracle", Transport: "worker", Label: "OFFLINE_ORACLE_INFORMATION_NOT_NATIVE"},
	}
}

// m7Decisions extracts everything that must match between two arms that claim identical behaviour.
func m7Decisions(rec map[string]any) map[string]any {
	steps := []any{}
	for _, s := range rec["steps"].([]any) {
		m := s.(map[string]any)
		steps = append(steps, map[string]any{"chosen": m["chosen_node"], "S": m["native_scores_S"], "order": m["native_order_S"], "after": m["after_resources"], "bind": m["bind"]})
	}
	return map[string]any{"steps": steps, "future": rec["future"], "admitted": rec["future_admitted"], "final": rec["final_resources"]}
}

func m7Same(a, b map[string]any) bool {
	x, _ := json.Marshal(m7Decisions(a))
	y, _ := json.Marshal(m7Decisions(b))
	return string(x) == string(y)
}

func m7Legality(t *testing.T) map[string]any {
	t.Helper()
	receipt := map[string]any{}
	count := 0
	for _, sc := range m7Scenes() {
		alt := sc
		alt.Future = []m6Job{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}
		for i := range sc.Arrivals {
			a := m7ForecastCohort(sc, i, i+1)
			b := m7ForecastCohort(alt, i, i+1)
			if a.Hash != b.Hash {
				t.Fatalf("scene %s: the declared-forecast cohort depends on the held-out future (leak)", sc.ID)
			}
			held := map[string]bool{}
			for k := range sc.Future {
				held[fmt.Sprintf("native-job-%d", len(sc.Arrivals)+k)] = true
			}
			for _, id := range a.IDs {
				if held[id] {
					t.Fatalf("cohort member %s collides with a held-out job id", id)
				}
			}
			if a.Source != "DECLARED_FORECAST" || len(a.IDs) == 0 || len(a.IDs) > 8 {
				t.Fatal("cohort provenance/bounds")
			}
			o := m7OracleCohort(sc, i, i+1)
			if o.Source != "OFFLINE_ORACLE_INFORMATION_NOT_NATIVE" || o.Hash == a.Hash {
				t.Fatal("oracle cohort must be labelled and distinct")
			}
			count++
		}
	}
	receipt["decisions_checked"], receipt["declared_forecast_cohort_independent_of_held_out_future"] = count, true
	receipt["cohort_source_for_native_arms"], receipt["oracle_source_label"] = "DECLARED_FORECAST", "OFFLINE_ORACLE_INFORMATION_NOT_NATIVE"
	return receipt
}

func m7TrustBoundary(t *testing.T, dir string) map[string]any {
	t.Helper()
	receipt := map[string]any{"legality": m7Legality(t)}
	sc := m6MakeScene(m6Regimes[0], 0)
	e := m7Session(t, "m7-trust", sc)
	_, cands, _ := p2hEligibility(e.m6Env)
	praw, err := p2hProjection(t, e.m6Env)
	if err != nil {
		t.Fatal(err)
	}
	cohort := m7ForecastCohort(sc, 0, 1)
	mk := func(raw []byte, c m7Cohort, producer string, phase int) *pcGrant {
		g := p2hGrant{owner: e.f.session, task: e.f.task, nodes: map[string]*node_info.NodeInfo{}, names: append([]string{}, e.names...), candidates: cands, phase: phase,
			envelopeHash: bridgeHash(raw), projectionHash: bridgeHash(praw), producerHash: producer}
		for name, n := range e.f.session.ClusterInfo.Nodes {
			g.nodes[name] = n
		}
		g.project = func(s *framework.Session) ([]byte, error) { return p2hProjection(t, e.m6Env) }
		return &pcGrant{p2hGrant: g, cohortHash: c.Hash, cohortSource: c.Source, cohortIDs: c.IDs}
	}
	p := m7Exec(t, "p2hpc", "cold", praw, 1, .5, sc.Forecast, &cohort, filepath.Join(dir, "pc-legit"))
	good := mk(p.raw, cohort, p.producerHash, 1)
	if _, err := decodePC(p.raw, good); err != nil {
		t.Fatal("legitimate P2H-PC-1 envelope rejected: ", err)
	}
	var obj map[string]any
	_ = json.Unmarshal(p.raw, &obj)
	mutations := map[string]func(map[string]any){
		"cohort_hash":          func(a map[string]any) { a["cohort_sha256"] = "0000" },
		"cohort_source_oracle": func(a map[string]any) { a["cohort_source"] = "OFFLINE_ORACLE_INFORMATION_NOT_NATIVE" },
		"cohort_source_new":    func(a map[string]any) { a["cohort_source"] = "SECRET_FUTURE" },
		"cohort_ids_missing":   func(a map[string]any) { a["cohort_ids"] = a["cohort_ids"].([]any)[1:] },
		"cohort_ids_extra":     func(a map[string]any) { a["cohort_ids"] = append(a["cohort_ids"].([]any), "oracle:0") },
		"cohort_ids_reordered": func(a map[string]any) { c := a["cohort_ids"].([]any); c[0], c[1] = c[1], c[0] },
		"cohort_ids_not_str":   func(a map[string]any) { a["cohort_ids"].([]any)[0] = nil },
		"version_p2h":          func(a map[string]any) { a["version"] = "P2H-1" },
		"pairs_missing":        func(a map[string]any) { a["pairs"] = a["pairs"].([]any)[1:] },
		"pairs_reordered": func(a map[string]any) {
			q := a["pairs"].([]any)
			q[0], q[1] = q[1], q[0]
		},
		"candidates_reordered": func(a map[string]any) { c := a["candidates"].([]any); c[0], c[1] = c[1], c[0] },
		"unknown_field":        func(a map[string]any) { a["unknown"] = 1 },
		"missing_cohort_field": func(a map[string]any) { delete(a, "cohort_sha256") },
		"task_uid":             func(a map[string]any) { a["task_uid"] = "foreign" },
		"phase":                func(a map[string]any) { a["phase"] = 2 },
		"artifact_hash":        func(a map[string]any) { a["artifact_sha256"] = "0000" },
	}
	rejected := []string{}
	for name, mutate := range mutations {
		var a map[string]any
		_ = json.Unmarshal(p.raw, &a)
		mutate(a)
		b, _ := json.Marshal(a)
		g := *good
		g.used, g.envelopeHash = false, bridgeHash(b)
		if _, err := decodePC(b, &g); err == nil {
			t.Fatal("PC receiver accepted mutation ", name)
		}
		rejected = append(rejected, name)
	}
	sort.Strings(rejected)
	receipt["receiver_mutations_rejected"] = rejected
	// A grant that declares a different cohort than the envelope carries is refused.
	wrong := m7ForecastCohort(sc, 1, 1)
	if _, err := decodePC(p.raw, mk(p.raw, wrong, p.producerHash, 1)); err == nil {
		t.Fatal("cohort differing from the grant's declared cohort accepted")
	}
	// The frozen P2H receiver and the PC receiver reject each other's envelopes.
	p2 := m7Exec(t, "p2h", "cold", praw, 1, 0, sc.Forecast, nil, filepath.Join(dir, "p2h-legit"))
	pg := *good
	pg.envelopeHash = p2.envelope
	if _, err := decodePC(p2.raw, &pg); err == nil {
		t.Fatal("PC receiver accepted a P2H envelope")
	}
	fg := good.p2hGrant
	fg.envelopeHash = p.envelope
	if _, err := decodeP2H(p.raw, &fg); err == nil {
		t.Fatal("frozen P2H receiver accepted a P2H-PC-1 envelope")
	}
	receipt["frozen_p2h_and_pc_receivers_reject_each_other"] = true
	// Python refuses oracle/unknown sources only when labelled otherwise; the native arms use DECLARED_FORECAST only.
	if !e.pc.begin(p.raw, good, m4bLambda) {
		t.Fatal("legitimate PC grant did not activate")
	}
	e.pc.endPhase()
	if e.pc.begin(p.raw, good, m4bLambda) {
		t.Fatal("spent PC grant reactivated")
	}
	receipt["legitimate_grant_activates_once"] = true
	return receipt
}

func m7Native(t *testing.T, dir string, cohortAllowed bool) (map[string]any, []any) {
	t.Helper()
	normalized := map[string]any{"version": "M7-N1", "mode": "NATIVE_FAKE_SESSION_SIMULATED_WHOLE_GPU_PROBE_ONLY", "projection": p2hVersion,
		"cohort_arms_run": cohortAllowed, "scenes": []any{}, "arms": []string{}}
	if !cohortAllowed {
		normalized["b1_status"] = "NATIVE_NONZERO_LP_NOT_RUN / TRUST_BOUNDARY_BLOCKED"
	}
	timings := []any{}
	cfgs := m7Cfgs()
	for _, cfg := range cfgs {
		if cfg.Cohort != "" && !cohortAllowed {
			continue
		}
		normalized["arms"] = append(normalized["arms"].([]string), cfg.Name)
	}
	for _, sc := range m7Scenes() {
		perArm := map[string]any{}
		for _, cfg := range cfgs {
			if cfg.Cohort != "" && !cohortAllowed {
				continue
			}
			cfg := cfg
			t.Run(sc.ID+"/"+strconv.Quote(cfg.Name), func(t *testing.T) {
				d := filepath.Join(dir, "scenes", sc.Regime, strconv.Itoa(sc.Seed), cfg.Name)
				_ = os.MkdirAll(d, 0700)
				rec, tim := m7RunArm(t, sc, cfg, d)
				perArm[cfg.Name] = rec
				timings = append(timings, tim)
			})
		}
		// Integrity controls inside the run: the worker transport must not change a single decision, the
		// alpha=0 planning-cohort pipeline must reproduce the single-target chain, and lambda=0 must equal OFF.
		if len(perArm) == len(cfgs) || (!cohortAllowed && len(perArm) > 0) {
			must := func(a, b, why string) {
				if x, ok := perArm[a]; ok {
					if y, ok2 := perArm[b]; ok2 && !m7Same(x.(map[string]any), y.(map[string]any)) {
						t.Fatalf("%s: %s differs from %s (%s)", sc.ID, a, b, why)
					}
				}
			}
			must("pgfp_l40", "pgfp_l40_worker", "cold vs worker transport")
			must("pc_alpha0_l20_cohort", "pgfp_l20", "alpha=0 cohort pipeline equals single-target chain")
			off := perArm["native_off"].(map[string]any)
			l0 := perArm["pgfp_l0"].(map[string]any)
			if !m7Same(map[string]any{"steps": off["steps"], "future": off["future"], "future_admitted": off["future_admitted"], "final_resources": off["final_resources"]},
				map[string]any{"steps": l0["steps"], "future": l0["future"], "future_admitted": l0["future_admitted"], "final_resources": l0["final_resources"]}) {
				t.Fatalf("%s: lambda=0 must be indistinguishable from NATIVE_OFF", sc.ID)
			}
		}
		normalized["scenes"] = append(normalized["scenes"].([]any), map[string]any{"id": sc.ID, "arms": perArm})
	}
	return normalized, timings
}

// TestM7 is the M7 gate test.
func TestM7(t *testing.T) {
	test_utils.InitTestingInfrastructure()
	dir := os.Getenv("M7_EVIDENCE_DIR")
	if dir == "" {
		dir = t.TempDir()
	}
	_ = os.MkdirAll(dir, 0700)
	t.Cleanup(func() {
		if m7Shared != nil {
			m7Shared.stop()
		}
	})
	var trust map[string]any
	trustOK := t.Run("TrustBoundary", func(t *testing.T) { trust = m7TrustBoundary(t, filepath.Join(dir, "trust")) })
	if !trustOK {
		_ = os.WriteFile(filepath.Join(dir, "blocked.json"), []byte(`{"b1":"NATIVE_NONZERO_LP_NOT_RUN","reason":"TRUST_BOUNDARY_BLOCKED"}`), 0600)
	}
	var normalized map[string]any
	var timings []any
	if !t.Run("Native", func(t *testing.T) { normalized, timings = m7Native(t, filepath.Join(dir, "native"), trustOK) }) {
		t.Fatal("native stage failed")
	}
	normalized["trust_boundary"] = trust
	if m7Shared != nil {
		normalized["worker_restarts_during_native_stage"] = m7Shared.Restarts
	}
	raw, err := json.Marshal(normalized)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(dir, "normalized.json"), raw, 0600); err != nil {
		t.Fatal(err)
	}
	nativeSave(t, dir, "timings.json", timings)
	var latency map[string]any
	if !t.Run("LatencyB2", func(t *testing.T) {
		latency = m7LatencyB2(t, filepath.Join(dir, "latency"), filepath.Join(dir, "native"))
	}) {
		t.Fatal("B2 failed")
	}
	rawL, _ := json.Marshal(latency["normalized"])
	_ = os.WriteFile(filepath.Join(dir, "latency_normalized.json"), rawL, 0600)
	nativeSave(t, dir, "latency_timings.json", latency["timings"])
	t.Log("normalized_sha256 ", m4bHash(raw), " latency_normalized_sha256 ", m4bHash(rawL), " at ", time.Now().Format(time.RFC3339))
}
