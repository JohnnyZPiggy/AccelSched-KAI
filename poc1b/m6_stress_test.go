package main

// M6 Stage B/C: heterogeneous 4-node whole-GPU native stress matrix (runs only after the complete
// Stage A PASS). Scenes are a pure function of (regime, seed); the regime/seed matrix is declared
// below BEFORE any run and is never edited after seeing outcomes. Each scene runs NATIVE_OFF,
// NATIVE_DYNAMIC_PGFP (alpha=0) and NATIVE_DYNAMIC_PGFP_LP (alpha=.5) in three independently
// constructed official Fake Sessions with the same nodes, arrivals, forecast and held-out future.
// Per arrival: P2H projection -> controlled producer (real M3-A build/load) -> one-use grant ->
// OrderedNodesByTask/NodeOrderFn -> actual allocate.New().Execute + Fake Cache Bind -> seal.
// Held-out future jobs are replayed one at a time through native Allocate WITHOUT advice in every
// arm, only after all current decisions are frozen. Simulated GPUs; not physical GPUs.

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"testing"
	"time"

	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/actions/allocate"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/pod_status"
)

type m6Mix [5]int // percent over m6Types

var m6Types = []struct {
	Name string
	Job  m6Job
}{{"small", m6Job{1, 2, 1}}, {"mid", m6Job{2, 4, 2}}, {"large", m6Job{4, 8, 4}}, {"wide", m6Job{6, 8, 4}}, {"cpuheavy", m6Job{1, 8, 8}}}

type m6Regime struct {
	Name                     string
	Forecast, Realized, Arrs m6Mix
	SameShape                bool
}

// Declared a priori. Arrivals are small (<=2 GPUs) so every arm can always place them; the held-out
// future contains the large/wide jobs that fragmentation actually affects.
var m6Regimes = []m6Regime{
	{"matched", m6Mix{20, 20, 30, 20, 10}, m6Mix{20, 20, 30, 20, 10}, m6Mix{40, 35, 0, 0, 25}, false},
	{"shift_large_to_small", m6Mix{10, 10, 40, 30, 10}, m6Mix{50, 30, 10, 0, 10}, m6Mix{40, 35, 0, 0, 25}, false},
	{"shift_small_to_large", m6Mix{50, 30, 10, 0, 10}, m6Mix{10, 10, 40, 30, 10}, m6Mix{40, 35, 0, 0, 25}, false},
	{"scarce", m6Mix{10, 20, 40, 20, 10}, m6Mix{10, 20, 40, 20, 10}, m6Mix{10, 60, 0, 0, 30}, false},
	{"cpu_joint", m6Mix{20, 20, 30, 20, 10}, m6Mix{20, 20, 30, 20, 10}, m6Mix{10, 10, 0, 0, 80}, false},
	{"same_shape", m6Mix{20, 20, 30, 20, 10}, m6Mix{20, 20, 30, 20, 10}, m6Mix{40, 35, 0, 0, 25}, true},
}

const m6SeedsPerRegime = 5

type m6Scene struct {
	ID       string        `json:"id"`
	Regime   string        `json:"regime"`
	Seed     int           `json:"seed"`
	Nodes    []m6Node      `json:"nodes"`
	Arrivals []m6Job       `json:"arrivals"`
	Forecast []m4bForecast `json:"forecast"`
	Future   []m6Job       `json:"held_out_future"`
}

func m6H(parts ...any) uint64 {
	h := sha256.Sum256([]byte(fmt.Sprint(parts...)))
	return binary.BigEndian.Uint64(h[:8])
}

func m6Draw(mix m6Mix, tag ...any) (string, m6Job) {
	point, total := int(m6H(tag...)%100), 0
	for i, share := range mix {
		total += share
		if point < total {
			return m6Types[i].Name, m6Types[i].Job
		}
	}
	panic("mix must sum to 100")
}

func m6MakeScene(r m6Regime, seed int) m6Scene {
	tag := r.Name + ":" + strconv.Itoa(seed)
	sc := m6Scene{ID: tag, Regime: r.Name, Seed: seed}
	if r.SameShape {
		sc.Nodes = []m6Node{m6Pool[1], m6Pool[1], m6Pool[1], m6Pool[1]}
	} else { // all four distinct shapes in a seed-dependent order
		order := []int{0, 1, 2, 3}
		for i := len(order) - 1; i > 0; i-- {
			j := int(m6H(tag, "perm", i) % uint64(i+1))
			order[i], order[j] = order[j], order[i]
		}
		for _, k := range order {
			sc.Nodes = append(sc.Nodes, m6Pool[k])
		}
	}
	count := 4 + int(m6H(tag, "arrivals")%2)
	for i := 0; i < count; i++ {
		_, j := m6Draw(r.Arrs, tag, "arrival", i)
		sc.Arrivals = append(sc.Arrivals, j)
	}
	for i := 0; i < 3; i++ {
		_, j := m6Draw(r.Forecast, tag, "forecast", i)
		sc.Forecast = append(sc.Forecast, m4bForecast{ID: fmt.Sprintf("F%d", i), Demand: []int{j.GPUs, j.Cores, j.GiB}, Weight: 1 + int(m6H(tag, "weight", i)%3), Eligible: m6Names(4)})
	}
	for i := 0; i < 3; i++ {
		_, j := m6Draw(r.Realized, tag, "realized", i)
		sc.Future = append(sc.Future, j)
	}
	return sc
}

type m6Times struct{ Fixture, Projection, Producer, LPSolve, Begin, NativeScore, Allocate, Future float64 }

// m6LPSeconds sums the solver attempt times recorded by the controlled producer (M3-A execution report).
func m6LPSeconds(dir string) float64 {
	raw, err := os.ReadFile(filepath.Join(dir, "producer", "execution.json"))
	if err != nil {
		return 0
	}
	var ex struct {
		Attempts []struct {
			Elapsed float64 `json:"elapsed_seconds"`
		} `json:"attempts"`
	}
	if json.Unmarshal(raw, &ex) != nil {
		return 0
	}
	total := 0.0
	for _, a := range ex.Attempts {
		total += a.Elapsed
	}
	return total
}

func m6RunArm(t *testing.T, sc m6Scene, arm, dir string) (map[string]any, map[string]any) {
	t.Helper()
	times := m6Times{}
	start := time.Now()
	e := m6Session(t, "m6-"+sc.ID+"-"+arm, sc.Nodes, m6Topology(sc.ID, sc.Nodes, sc.Arrivals[0], nil))
	times.Fixture = time.Since(start).Seconds()
	alpha := 0.0
	if arm == "native_dynamic_pgfp_lp" {
		alpha = 0.5
	}
	steps, phase := []any{}, 0
	seen := map[string]bool{}
	for i, job := range sc.Arrivals {
		if i > 0 {
			e.arrive(t, i, job)
		}
		stepDir := filepath.Join(dir, fmt.Sprintf("arrival-%d", i))
		if err := os.MkdirAll(stepDir, 0700); err != nil {
			t.Fatal(err)
		}
		beforeRes := m6Resources(e.f.session, e.names)
		beforeDomain := probeDomain(t, e.f)
		elig, cands, err := p2hEligibility(e)
		if err != nil {
			t.Fatal(err)
		}
		if len(cands) == 0 {
			t.Fatalf("scene %s arm %s: arrival %d has no native candidate; the scene design invariant (arrivals always placeable) is violated", sc.ID, arm, i)
		}
		eligOut := map[string]any{}
		for _, name := range e.names {
			eligOut[name] = elig[name]
		}
		s0, order0, secs := m6Scores(e, t, cands)
		times.NativeScore += secs
		step := map[string]any{"index": i, "task": string(e.f.task.UID), "request": job, "before_resources": beforeRes, "before_domain_sha256": m4bHash(beforeDomain),
			"eligibility": eligOut, "candidates": cands, "native_scores_S0": s0, "native_order_S0": order0}
		var produced m6Produced
		active := false
		if arm != "native_off" {
			phase++
			projStart := time.Now()
			praw, err := p2hProjection(t, e)
			times.Projection += time.Since(projStart).Seconds()
			if err != nil {
				t.Fatal("P2H projection refused a state the scene declares supported: ", err)
			}
			produced = m6Produce(t, e, phase, alpha, sc.Forecast, filepath.Join(stepDir, "handoff"))
			times.Producer += produced.seconds
			times.LPSolve += m6LPSeconds(filepath.Join(stepDir, "handoff"))
			begin := time.Now()
			active = e.plugin.begin(produced.raw, produced.grant, m4bLambda)
			times.Begin += time.Since(begin).Seconds()
			if !active {
				t.Fatal("fresh certified P2H artifact did not activate at arrival ", i)
			}
			for key, h := range map[string]string{"projection": bridgeHash(praw), "envelope": produced.envelope, "artifact": m4bHash(produced.artifact)} {
				if seen[key+h] {
					t.Fatal("arrival reused a previous ", key, ": advice must be recomputed from the next native state")
				}
				seen[key+h] = true
			}
			step["lambda"], step["alpha"], step["BF"], step["BP"] = m4bLambda, alpha, m4bBF, m4bBP
			step["projection_sha256"], step["envelope_sha256"], step["artifact_sha256"] = bridgeHash(praw), produced.envelope, m4bHash(produced.artifact)
			step["controls"] = json.RawMessage(produced.controls)
			step["artifact_pairs"] = m4bPairs(t, produced.artifact)
		}
		scores, order, secs := m6Scores(e, t, cands)
		times.NativeScore += secs
		step["native_scores_S"], step["native_order_S"] = scores, order
		if active {
			audit := e.plugin.record.audit()
			step["advice_entries_returned"], step["score_calls_before_action"] = audit.Entries, audit.Total
			returned := map[string]float64{}
			for _, entry := range audit.Entries {
				if entry.Task == string(e.f.task.UID) {
					returned[entry.Node] = entry.Score
				}
			}
			if len(returned) != len(cands) || !audit.Complete {
				t.Fatal("native scoring did not use the bridge for every candidate ", audit)
			}
			for _, name := range cands {
				if scores[name]-s0[name] != returned[name] {
					t.Fatal("score not attributable to S0 + lambda*A ", name, scores[name], s0[name], returned[name])
				}
			}
		} else if !reflect.DeepEqual(scores, s0) || e.plugin.phaseActive {
			t.Fatal("NATIVE_OFF must be the unadvised native score")
		}
		binds := len(*e.binds)
		allocStart := time.Now()
		allocate.New().Execute(e.f.session)
		times.Allocate += time.Since(allocStart).Seconds()
		if active {
			e.plugin.endPhase()
		}
		p := e.f.session.ClusterInfo.PodGroupInfos[e.f.job.UID].GetAllPodsMap()[e.f.task.UID]
		if p == nil || p.Status != pod_status.Binding || len(*e.binds)-binds != 1 {
			t.Fatal("native Allocate did not bind exactly one task at arrival ", i)
		}
		afterRes := m6Resources(e.f.session, e.names)
		if err := m6Conserved(beforeRes, afterRes, e.names, p.NodeName, int64(job.GPUs), int64(job.Cores)*1000, int64(job.GiB)*nativeGiB); err != nil {
			t.Fatal("resource conservation: ", err)
		}
		step["chosen_node"], step["status_after"], step["bind"] = p.NodeName, p.Status.String(), (*e.binds)[len(*e.binds)-1]
		step["after_resources"], step["after_domain_sha256"] = afterRes, m4bHash(probeDomain(t, e.f))
		steps = append(steps, step)
	}
	// Held-out future: hidden until now; native Allocate without advice, identical rule for every arm.
	futureStart := time.Now()
	future, admitted, admittedGPUs, admittedCores := []any{}, 0, 0, 0
	for k, job := range sc.Future {
		e.arrive(t, len(sc.Arrivals)+k, job)
		_, cands, err := p2hEligibility(e)
		if err != nil {
			t.Fatal(err)
		}
		before := m6Resources(e.f.session, e.names)
		binds := len(*e.binds)
		allocate.New().Execute(e.f.session)
		p := e.f.session.ClusterInfo.PodGroupInfos[e.f.job.UID].GetAllPodsMap()[e.f.task.UID]
		row := map[string]any{"index": k, "task": string(e.f.task.UID), "request": job, "candidates": cands, "before_resources": before}
		if len(*e.binds)-binds == 1 && p.Status == pod_status.Binding {
			admitted++
			admittedGPUs += job.GPUs
			admittedCores += job.Cores
			row["outcome"], row["node"] = "ACCEPTED", p.NodeName
			if err := m6Conserved(before, m6Resources(e.f.session, e.names), e.names, p.NodeName, int64(job.GPUs), int64(job.Cores)*1000, int64(job.GiB)*nativeGiB); err != nil {
				t.Fatal("future resource conservation: ", err)
			}
		} else if len(*e.binds) != binds || p.Status != pod_status.Pending {
			t.Fatal("future task in unexpected native state")
		} else if len(cands) == 0 {
			row["outcome"], row["reason"] = "REJECTED", "NO_FEASIBLE_NODE"
		} else {
			row["outcome"], row["reason"] = "REJECTED", "FEASIBLE_BUT_NOT_BOUND"
		}
		if row["outcome"] == "REJECTED" {
			// Harness bookkeeping, not a scheduling decision: a rejected held-out job is withdrawn (no resources,
			// no node) so that it cannot head-of-line block the independent later arrivals of the replay.
			if err := e.f.job.UpdateTaskStatus(p, pod_status.Failed); err != nil {
				t.Fatal("withdrawing a rejected held-out job: ", err)
			}
			row["withdrawn_after_rejection"] = true
		}
		row["after_resources"] = m6Resources(e.f.session, e.names)
		future = append(future, row)
	}
	times.Future = time.Since(futureStart).Seconds()
	record := map[string]any{"arm": arm, "scene": sc, "steps": steps, "future": future, "future_admitted": admitted, "future_admitted_gpus": admittedGPUs, "future_admitted_cores": admittedCores,
		"future_total": len(sc.Future), "final_resources": m6Resources(e.f.session, e.names), "hooks": probeHooks(t, e.f.session), "bind_count": len(*e.binds),
		"native_future_admission": "NATIVE_ALLOCATE_NO_ADVICE_IDENTICAL_FOR_ALL_ARMS"}
	return record, map[string]any{"arm": arm, "scene": sc.ID, "regime": sc.Regime, "arrivals": len(sc.Arrivals), "seconds": times}
}

// m6SameBatch is the multi-task same-batch regression with full phase invalidation and a fresh grant.
func m6SameBatch(t *testing.T, dir string) map[string]any {
	t.Helper()
	receipt := map[string]any{}
	jobs := []m6Job{{2, 2, 1}, {2, 2, 1}}
	e := m6Session(t, "m6-same-batch", m6Pool, m6Topology("same-batch", m6Pool, jobs[0], nil))
	_, cands, _ := p2hEligibility(e)
	first := m6Produce(t, e, 1, 0, m6Forecast, filepath.Join(dir, "batch-first"))
	if !e.plugin.begin(first.raw, first.grant, m4bLambda) {
		t.Fatal("first grant")
	}
	allocate.New().Execute(e.f.session)
	e.plugin.endPhase()
	// Full phase invalidation: right after endPhase every node scores exactly the native S0.
	afterClose, _, _ := m6Scores(e, t, e.names)
	e.arrive(t, 1, jobs[1])
	if e.plugin.begin(first.raw, first.grant, m4bLambda) {
		t.Fatal("the first batch member's spent grant must not cover the second")
	}
	// Two simultaneous Pending members cannot even be projected (no batch-start scoring is expressible).
	e.arrive(t, 2, jobs[1])
	if _, err := p2hProjection(t, e); err == nil {
		t.Fatal("two Pending batch members projected")
	}
	receipt["spent_grant_rejected_for_second_member"], receipt["two_pending_members_refused"] = true, true
	receipt["scores_after_phase_close_are_native"] = len(afterClose) == len(e.names)
	_ = cands
	return receipt
}

// m6MechanismScenes are fixed, hand-declared scenes from the M5 hole-preservation mechanism. They were
// ADDED after the first complete dev run found no beneficial PGFP scene in the random matrix (disclosed
// in the PR); they are designed from the mechanism, not selected by searching outcomes.
func m6MechanismScenes() []m6Scene {
	four := m4bForecast{ID: "F0", Demand: []int{4, 2, 1}, Weight: 2, Eligible: m6Names(4)}
	return []m6Scene{{ID: "mechanism_hole:0", Regime: "mechanism_hole", Seed: 0,
		Nodes:    []m6Node{{6, 16, 64}, {6, 16, 64}, {6, 16, 64}, {6, 16, 64}},
		Arrivals: []m6Job{{2, 2, 1}, {2, 2, 1}, {2, 2, 1}, {2, 2, 1}},
		Forecast: []m4bForecast{four},
		Future:   []m6Job{{4, 2, 1}, {4, 2, 1}, {4, 2, 1}, {4, 2, 1}}}}
}

func m6StageBC(t *testing.T, dir string) (map[string]any, []any) {
	t.Helper()
	normalized := map[string]any{"version": "M6-B1", "mode": "NATIVE_FAKE_SESSION_SIMULATED_WHOLE_GPU_MULTINODE_PROBE_ONLY", "projection": p2hVersion,
		"lambda": m4bLambda, "BF": m4bBF, "BP": m4bBP, "alpha_lp": 0.5, "seeds_per_regime": m6SeedsPerRegime, "regimes": []string{}, "scenes": []any{}}
	timings := []any{}
	scenes := []m6Scene{}
	for _, r := range m6Regimes {
		normalized["regimes"] = append(normalized["regimes"].([]string), r.Name)
		for seed := 0; seed < m6SeedsPerRegime; seed++ {
			scenes = append(scenes, m6MakeScene(r, seed))
		}
	}
	normalized["regimes"] = append(normalized["regimes"].([]string), "mechanism_hole")
	scenes = append(scenes, m6MechanismScenes()...)
	for _, sc := range scenes {
		{
			seed := sc.Seed
			perArm := map[string]any{}
			for _, arm := range m4bArms {
				t.Run(sc.ID+"/"+strconv.Quote(arm), func(t *testing.T) {
					d := filepath.Join(dir, "scenes", sc.Regime, strconv.Itoa(seed), arm)
					if err := os.MkdirAll(d, 0700); err != nil {
						t.Fatal(err)
					}
					rec, tim := m6RunArm(t, sc, arm, d)
					perArm[arm] = rec
					timings = append(timings, tim)
				})
			}
			if len(perArm) == len(m4bArms) {
				ref := perArm["native_off"].(map[string]any)["steps"].([]any)[0].(map[string]any)
				for _, arm := range m4bArms[1:] {
					got := perArm[arm].(map[string]any)["steps"].([]any)[0].(map[string]any)
					for _, key := range []string{"before_domain_sha256", "before_resources", "native_scores_S0", "native_order_S0", "candidates"} {
						if !reflect.DeepEqual(ref[key], got[key]) {
							t.Fatalf("%s/%s first-arrival %s differs from NATIVE_OFF", sc.ID, arm, key)
						}
					}
				}
			}
			normalized["scenes"] = append(normalized["scenes"].([]any), map[string]any{"id": sc.ID, "arms": perArm})
		}
	}
	return normalized, timings
}
