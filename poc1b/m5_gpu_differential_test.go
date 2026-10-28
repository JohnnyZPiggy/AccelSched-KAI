package main

// M5 Stage B (runs only after the complete Stage A PASS): native whole-GPU fragmentation
// differential. NATIVE_OFF, NATIVE_DYNAMIC_PGFP (alpha=0) and NATIVE_DYNAMIC_PGFP_LP (alpha=.5,
// predeclared) each use a separately constructed official Fake Session from the same immutable
// trace. Every arrival: P2 projection of the live native state -> controlled Python producer
// (real M3-A build/load) -> private one-use grant -> actual OrderedNodesByTask/NodeOrderFn ->
// actual allocate.New().Execute with the Fake Cache -> seal -> recompute from the NEXT state.
// Held-out future arrivals run through native Allocate without advice for every arm.
// Simulated Fake-Session GPUs are not physical GPUs.

import (
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
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/test_utils"
)

type m5Times struct{ Fixture, Projection, Producer, Begin, NativeScore, Allocate float64 }

type m5Trace struct {
	Name     string        `json:"name"`
	Nodes    [2]m5Node     `json:"nodes"`
	Arrivals []m5Job       `json:"arrivals"`
	Forecast []m4bForecast `json:"forecast"`
	Future   []m5Job       `json:"held_out_future"`
}

var (
	m5Big6   = [2]m5Node{{6, 16}, {6, 16}}
	m5Fore4  = []m4bForecast{{ID: "F0", Demand: []int{4, 2, 1}, Weight: 2, Eligible: m4bBoth}}
	m5Fore4c = []m4bForecast{{ID: "F0", Demand: []int{4, 4, 1}, Weight: 2, Eligible: m4bBoth}}
	m5Fore1  = []m4bForecast{{ID: "F0", Demand: []int{1, 1, 1}, Weight: 2, Eligible: m4bBoth}}
)

// Hand-declared a priori from the hole-preservation / stranded-GPU mechanisms; never tuned on outcomes.
var m5Traces = []m5Trace{
	{"split_vs_concentrated", m5Big6, []m5Job{{2, 2}, {2, 2}}, m5Fore4, []m5Job{{4, 2}, {4, 2}}},
	{"shift_harm", m5Big6, []m5Job{{2, 2}, {2, 2}}, m5Fore4, []m5Job{{6, 2}}},
	{"joint_gpu_cpu_fragmentation", [2]m5Node{{8, 16}, {8, 12}}, []m5Job{{1, 6}, {1, 6}}, m5Fore4c, []m5Job{{4, 4}, {4, 4}}},
	{"same_shape_equal_lp", [2]m5Node{{8, 16}, {8, 16}}, []m5Job{{2, 2}, {2, 2}, {2, 2}}, m5Fore4, []m5Job{{4, 2}, {4, 2}}},
	{"forecast_small_no_benefit", m5Big6, []m5Job{{2, 2}, {2, 2}}, m5Fore1, []m5Job{{4, 2}, {4, 2}}},
}

func m5Feasible(e *m5Env) (map[string]bool, map[string]string) {
	s, task, job := e.f.session, e.f.task, e.f.job
	feasible, errs := map[string]bool{}, map[string]string{}
	if err := s.PrePredicateFn(task, job); err != nil {
		errs["pre"] = err.Error()
	}
	for _, name := range m4bBoth {
		n := s.ClusterInfo.Nodes[name]
		if err := s.PredicateFn(task, job, n); err != nil {
			errs[name] = err.Error()
		}
		feasible[name] = s.FittingNode(task, n, false)
	}
	return feasible, errs
}

// m5Conserved: idle+used stays constant per coordinate; only the chosen node's idle drops by the request.
func m5Conserved(before, after map[string]any, node string, gpus, millis, mem int64) error {
	for _, name := range m4bBoth {
		b, a := before[name].(map[string]any), after[name].(map[string]any)
		for _, key := range []string{"allocatable", "releasing"} {
			if !reflect.DeepEqual(b[key], a[key]) {
				return fmt.Errorf("%s %s changed", name, key)
			}
		}
		bi, ai := b["idle"].([]int64), a["idle"].([]int64)
		bu, au := b["used"].([]int64), a["used"].([]int64)
		want := []int64{0, 0, 0}
		if name == node {
			want = []int64{gpus, millis, mem}
		}
		for k := 0; k < 3; k++ {
			if bi[k]+bu[k] != ai[k]+au[k] || bi[k]-ai[k] != want[k] || au[k]-bu[k] != want[k] {
				return fmt.Errorf("%s coordinate %d not conserved", name, k)
			}
		}
	}
	return nil
}

func m5RunArm(t *testing.T, tr m5Trace, arm, dir string) (map[string]any, map[string]any) {
	t.Helper()
	times := m5Times{}
	start := time.Now()
	e := m5Session(t, "m5-"+tr.Name+"-"+arm, m5Topology(tr.Name, tr.Nodes, tr.Arrivals[0], nil))
	e.nodes = tr.Nodes
	times.Fixture = time.Since(start).Seconds()
	alpha := 0.0
	if arm == "native_dynamic_pgfp_lp" {
		alpha = 0.5
	}
	steps, phase := []any{}, 0
	var prevProjection, prevArtifact, prevEnvelope string
	for i, job := range tr.Arrivals {
		if i > 0 {
			e.arrive(t, i, job)
		}
		stepDir := filepath.Join(dir, fmt.Sprintf("arrival-%d", i))
		if err := os.MkdirAll(stepDir, 0700); err != nil {
			t.Fatal(err)
		}
		beforeRes := m4bResources(t, e.f.session)
		beforeDomain := probeDomain(t, e.f)
		feasible, perr := m5Feasible(e)
		me := &m4bEnv{f: e.f}
		s0, order0, secs := m4bScores(me, t)
		times.NativeScore += secs
		step := map[string]any{"index": i, "task": string(e.f.task.UID), "request": job, "before_resources": beforeRes, "before_domain_sha256": m4bHash(beforeDomain),
			"feasible": feasible, "predicate_errors": perr, "native_scores_S0": s0, "native_order_S0": order0}
		var produced m5Produced
		active := false
		if arm != "native_off" {
			phase++
			projStart := time.Now()
			praw, err := p2Projection(t, e)
			times.Projection += time.Since(projStart).Seconds()
			if err != nil {
				t.Fatal("P2 projection refused a state the trace declares supported: ", err)
			}
			produced = m5Produce(t, e, phase, alpha, tr.Forecast, filepath.Join(stepDir, "handoff"))
			times.Producer += produced.seconds
			begin := time.Now()
			active = e.plugin.begin(produced.raw, produced.grant, m4bLambda)
			times.Begin += time.Since(begin).Seconds()
			if !active {
				t.Fatal("fresh certified P2 artifact did not activate at arrival ", i)
			}
			step["lambda"], step["alpha"], step["BF"], step["BP"] = m4bLambda, alpha, m4bBF, m4bBP
			step["projection_sha256"], step["envelope_sha256"], step["artifact_sha256"] = bridgeHash(praw), produced.envelope, m4bHash(produced.artifact)
			if i > 0 && (prevProjection == bridgeHash(praw) || prevArtifact == m4bHash(produced.artifact) || prevEnvelope == produced.envelope) {
				t.Fatal("arrival reused a previous projection/artifact/envelope: advice must be recomputed from the next native state")
			}
			prevProjection, prevArtifact, prevEnvelope = bridgeHash(praw), m4bHash(produced.artifact), produced.envelope
			step["controls"] = json.RawMessage(produced.controls)
			step["artifact_pairs"] = m4bPairs(t, produced.artifact)
		}
		scores, order, secs := m4bScores(me, t)
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
			if len(returned) != 2 || !audit.Complete {
				t.Fatal("native scoring did not actually use the bridge for both nodes ", audit)
			}
			for _, name := range m4bBoth {
				if scores[name]-s0[name] != returned[name] {
					t.Fatal("score not attributable to S0 + lambda*A ", name, scores[name], s0[name], returned[name])
				}
			}
		} else if !reflect.DeepEqual(scores, s0) || e.plugin.phaseActive {
			t.Fatal("NATIVE_OFF must be the unadvised native score")
		}
		if !feasible["node0"] || !feasible["node1"] {
			t.Fatal("fewer than two natively feasible nodes at arrival ", i, feasible)
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
		afterRes := m4bResources(t, e.f.session)
		if err := m5Conserved(beforeRes, afterRes, p.NodeName, int64(job.GPUs), int64(job.Cores)*1000, nativeGiB); err != nil {
			t.Fatal("resource conservation: ", err)
		}
		step["chosen_node"], step["status_after"], step["bind"] = p.NodeName, p.Status.String(), (*e.binds)[len(*e.binds)-1]
		step["after_resources"], step["after_domain_sha256"] = afterRes, m4bHash(probeDomain(t, e.f))
		steps = append(steps, step)
	}
	future, admitted, admittedGPUs := []any{}, 0, 0
	for k, job := range tr.Future {
		e.arrive(t, len(tr.Arrivals)+k, job)
		feasible, perr := m5Feasible(e)
		before := m4bResources(t, e.f.session)
		binds := len(*e.binds)
		allocStart := time.Now()
		allocate.New().Execute(e.f.session)
		times.Allocate += time.Since(allocStart).Seconds()
		p := e.f.session.ClusterInfo.PodGroupInfos[e.f.job.UID].GetAllPodsMap()[e.f.task.UID]
		row := map[string]any{"index": k, "task": string(e.f.task.UID), "request": job, "feasible": feasible, "predicate_errors": perr, "before_resources": before}
		if len(*e.binds)-binds == 1 && p.Status == pod_status.Binding {
			admitted++
			admittedGPUs += job.GPUs
			row["outcome"], row["node"] = "ACCEPTED", p.NodeName
			if err := m5Conserved(before, m4bResources(t, e.f.session), p.NodeName, int64(job.GPUs), int64(job.Cores)*1000, nativeGiB); err != nil {
				t.Fatal("future resource conservation: ", err)
			}
		} else if len(*e.binds) != binds || p.Status != pod_status.Pending {
			t.Fatal("future task in unexpected native state")
		} else if !feasible["node0"] && !feasible["node1"] {
			row["outcome"], row["reason"] = "REJECTED", "NO_FEASIBLE_NODE"
		} else {
			row["outcome"], row["reason"] = "REJECTED", "FEASIBLE_BUT_NOT_BOUND"
		}
		row["after_resources"] = m4bResources(t, e.f.session)
		future = append(future, row)
	}
	record := map[string]any{"arm": arm, "trace": tr, "steps": steps, "future": future, "future_admitted": admitted, "future_admitted_gpus": admittedGPUs,
		"future_total": len(tr.Future), "final_resources": m4bResources(t, e.f.session), "hooks": probeHooks(t, e.f.session), "bind_count": len(*e.binds),
		"native_future_admission": "NATIVE_ALLOCATE_NO_ADVICE_IDENTICAL_FOR_ALL_ARMS"}
	return record, map[string]any{"arm": arm, "trace": tr.Name, "seconds": times}
}

func m5StageB(t *testing.T, dir string) (map[string]any, []any) {
	t.Helper()
	normalized := map[string]any{"version": "M5-B1", "mode": "NATIVE_FAKE_SESSION_SIMULATED_WHOLE_GPU_PROBE_ONLY", "projection": p2Version,
		"lambda": m4bLambda, "BF": m4bBF, "BP": m4bBP, "alpha_lp": 0.5, "traces": map[string]any{}}
	timings := []any{}
	for _, tr := range m5Traces {
		perTrace := map[string]any{}
		for _, arm := range m4bArms {
			t.Run(tr.Name+"/"+strconv.Quote(arm), func(t *testing.T) {
				d := filepath.Join(dir, tr.Name, arm)
				if err := os.MkdirAll(d, 0700); err != nil {
					t.Fatal(err)
				}
				rec, tim := m5RunArm(t, tr, arm, d)
				perTrace[arm] = rec
				timings = append(timings, tim)
			})
		}
		if len(perTrace) == len(m4bArms) {
			ref := perTrace["native_off"].(map[string]any)["steps"].([]any)[0].(map[string]any)
			for _, arm := range m4bArms[1:] {
				got := perTrace[arm].(map[string]any)["steps"].([]any)[0].(map[string]any)
				for _, key := range []string{"before_domain_sha256", "before_resources", "native_scores_S0", "native_order_S0", "feasible"} {
					if !reflect.DeepEqual(ref[key], got[key]) {
						t.Fatalf("%s/%s first-arrival %s differs from NATIVE_OFF", tr.Name, arm, key)
					}
				}
			}
		}
		normalized["traces"].(map[string]any)[tr.Name] = perTrace
	}
	return normalized, timings
}

func m5EvidenceDir(t *testing.T) string {
	if d := os.Getenv("M5_EVIDENCE_DIR"); d != "" {
		if err := os.MkdirAll(d, 0700); err != nil {
			t.Fatal(err)
		}
		return d
	}
	return t.TempDir()
}

// TestM5WholeGPU is the hard gate: Stage B is NOT_RUN unless the whole of Stage A passed.
func TestM5WholeGPU(t *testing.T) {
	test_utils.InitTestingInfrastructure()
	dir := m5EvidenceDir(t)
	var stageA map[string]any
	okA := t.Run("StageA", func(t *testing.T) { stageA = m5StageA(t, filepath.Join(dir, "stage_a")) })
	if !okA {
		raw, _ := json.Marshal(map[string]any{"status": "P2_GPU_SEMANTICS_BLOCKED", "stage_b": "NOT_RUN"})
		_ = os.WriteFile(filepath.Join(dir, "blocked.json"), raw, 0600)
		t.Fatal("P2_GPU_SEMANTICS_BLOCKED: Stage B NOT_RUN")
	}
	rawA, err := json.Marshal(stageA)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(dir, "stage_a.json"), rawA, 0600); err != nil {
		t.Fatal(err)
	}
	var normalized map[string]any
	var timings []any
	if !t.Run("StageB", func(t *testing.T) { normalized, timings = m5StageB(t, filepath.Join(dir, "stage_b")) }) {
		t.Fatal("Stage B failed")
	}
	normalized["stage_a_sha256"] = m4bHash(rawA)
	raw, err := json.Marshal(normalized)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(dir, "normalized.json"), raw, 0600); err != nil {
		t.Fatal(err)
	}
	nativeSave(t, dir, "timings.json", timings)
	t.Log("stage_a_sha256 ", m4bHash(rawA), " normalized_sha256 ", m4bHash(raw))
}
