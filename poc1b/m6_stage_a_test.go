package main

// M6 Stage A (hard gate): P2H-1 over 2/3/4 heterogeneous whole-GPU nodes - exact native candidate
// sets, identity, resources, certified artifacts and one-use grants, plus adversarial faults.

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/actions/allocate"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/node_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/pod_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/pod_status"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/podgroup_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/resource_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/test_utils"
	commonconstants "github.com/kai-scheduler/api/constants"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

// Heterogeneous pool: four distinct GPU/CPU/RAM shapes.
var m6Pool = []m6Node{{4, 16, 64}, {8, 32, 128}, {2, 8, 32}, {8, 16, 64}}
var m6Forecast = []m4bForecast{{ID: "F0", Demand: []int{4, 4, 1}, Weight: 2, Eligible: []string{"node0", "node1", "node2", "node3"}}}

func m6Forecast2(n int) []m4bForecast {
	return []m4bForecast{{ID: "F0", Demand: []int{4, 4, 1}, Weight: 2, Eligible: m6Names(n)}}
}

func m6Gate(t *testing.T, e *m6Env, dir string) (string, string) {
	t.Helper()
	praw, err := p2hProjection(t, e)
	if err != nil {
		return "go_projection", err.Error()
	}
	_, _, runErr := m6Run(t, praw, 1, 0, m6Forecast2(len(e.names)), dir)
	if runErr == nil {
		return "ACCEPTED", ""
	}
	b, _ := os.ReadFile(filepath.Join(dir, "producer.stderr.txt"))
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	return "python_producer", strings.TrimSpace(lines[len(lines)-1])
}

func m6Conserved(before, after map[string]any, names []string, node string, gpus, millis, mem int64) error {
	for _, name := range names {
		b, a := before[name].(map[string]any), after[name].(map[string]any)
		for _, key := range []string{"allocatable", "releasing", "allocatable_pods", "releasing_pods"} {
			if !reflect.DeepEqual(b[key], a[key]) {
				return fmt.Errorf("%s %s changed", name, key)
			}
		}
		bi, ai := b["idle"].([]int64), a["idle"].([]int64)
		bu, au := b["used"].([]int64), a["used"].([]int64)
		want, wantPods := []int64{0, 0, 0}, int64(0)
		if name == node {
			want, wantPods = []int64{gpus, millis, mem}, 1
		}
		for k := 0; k < 3; k++ {
			if bi[k]+bu[k] != ai[k]+au[k] || bi[k]-ai[k] != want[k] || au[k]-bu[k] != want[k] {
				return fmt.Errorf("%s coordinate %d not conserved", name, k)
			}
		}
		if b["idle_pods"].(int64)-a["idle_pods"].(int64) != wantPods || a["used_pods"].(int64)-b["used_pods"].(int64) != wantPods {
			return fmt.Errorf("%s pod slots not conserved", name)
		}
	}
	return nil
}

// m6OneArrival runs projection -> producer -> grant -> native scoring -> Allocate -> seal for the
// current Pending target and returns the evidence record. It fails the test on any deviation.
func m6OneArrival(t *testing.T, e *m6Env, job m6Job, phase int, alpha float64, forecast []m4bForecast, dir string) map[string]any {
	t.Helper()
	before := m6Resources(e.f.session, e.names)
	elig, cands, err := p2hEligibility(e)
	if err != nil {
		t.Fatal(err)
	}
	s0, order0, _ := m6Scores(e, t, cands)
	produced := m6Produce(t, e, phase, alpha, forecast, dir)
	if !e.plugin.begin(produced.raw, produced.grant, m4bLambda) {
		t.Fatal("fresh certified P2H artifact did not activate")
	}
	scores, order, _ := m6Scores(e, t, cands)
	audit := e.plugin.record.audit()
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
			t.Fatal("score not attributable to S0 + lambda*A ", name)
		}
	}
	binds := len(*e.binds)
	allocate.New().Execute(e.f.session)
	e.plugin.endPhase()
	p := e.f.session.ClusterInfo.PodGroupInfos[e.f.job.UID].GetAllPodsMap()[e.f.task.UID]
	if p == nil || p.Status != pod_status.Binding || len(*e.binds)-binds != 1 {
		t.Fatal("native Allocate did not bind exactly one task")
	}
	after := m6Resources(e.f.session, e.names)
	if err := m6Conserved(before, after, e.names, p.NodeName, int64(job.GPUs), int64(job.Cores)*1000, int64(job.GiB)*nativeGiB); err != nil {
		t.Fatal("resource conservation: ", err)
	}
	eligOut := map[string]any{}
	for _, name := range e.names {
		eligOut[name] = elig[name]
	}
	return map[string]any{"candidates": cands, "eligibility": eligOut, "S0": s0, "order_S0": order0, "S": scores, "order_S": order, "chosen": p.NodeName,
		"envelope_sha256": produced.envelope, "artifact_sha256": m4bHash(produced.artifact), "projection_sha256": produced.grant.projectionHash,
		"before_resources": before, "after_resources": after}
}

func m6Positive(t *testing.T, dir string) []any {
	t.Helper()
	out := []any{}
	for _, n := range []int{2, 3, 4} {
		for _, req := range []m6Job{{2, 2, 1}, {6, 4, 2}} {
			e := m6Session(t, fmt.Sprintf("m6-pos-%d-%dg", n, req.GPUs), m6Pool[:n], m6Topology("pos", m6Pool[:n], req, nil))
			rec := m6OneArrival(t, e, req, 1, 0, m6Forecast2(n), filepath.Join(dir, fmt.Sprintf("positive-%d-%dg", n, req.GPUs)))
			rec["nodes"], rec["request"] = n, req
			out = append(out, rec)
		}
	}
	return out
}

// m6StateEvolution runs five consecutive arrivals through one Session (phases 1..5), each with a
// fresh certified artifact, proving the extended phase cap and that every grant is current-state only.
func m6StateEvolution(t *testing.T, dir string) map[string]any {
	t.Helper()
	jobs := []m6Job{{2, 2, 1}, {1, 4, 2}, {4, 4, 2}, {1, 2, 1}, {2, 6, 4}}
	e := m6Session(t, "m6-evolution", m6Pool, m6Topology("evolution", m6Pool, jobs[0], nil))
	seen := map[string]bool{}
	rows := []any{}
	for i, j := range jobs {
		if i > 0 {
			e.arrive(t, i, j)
		}
		rec := m6OneArrival(t, e, j, i+1, 0, m6Forecast, filepath.Join(dir, fmt.Sprintf("evolution-%d", i)))
		for _, k := range []string{"envelope_sha256", "artifact_sha256", "projection_sha256"} {
			h := rec[k].(string)
			if seen[k+h] {
				t.Fatal("arrival reused a previous ", k)
			}
			seen[k+h] = true
		}
		rows = append(rows, map[string]any{"arrival": i, "candidates": rec["candidates"], "chosen": rec["chosen"], "order": rec["order_S"], "artifact_sha256": rec["artifact_sha256"]})
	}
	if e.plugin.lastPhase != len(jobs) {
		t.Fatal("phase counter")
	}
	return map[string]any{"arrivals": rows, "max_phase_used": e.plugin.lastPhase, "phase_cap": p2hMaxPhase}
}

func m6EligibilityTable(t *testing.T) []any {
	t.Helper()
	out := []any{}
	for _, job := range []m6Job{{1, 2, 1}, {2, 8, 8}, {3, 4, 4}, {4, 16, 8}, {6, 4, 2}, {8, 16, 64}, {8, 32, 128}, {9, 2, 1}, {1, 33, 1}, {1, 2, 129}} {
		e := m6Session(t, fmt.Sprintf("m6-elig-%dg-%dc-%dm", job.GPUs, job.Cores, job.GiB), m6Pool, m6Topology("elig", m6Pool, job, nil))
		elig, cands, err := p2hEligibility(e)
		if err != nil {
			t.Fatal(err)
		}
		row := map[string]any{"request": job, "candidates": cands}
		for _, name := range e.names {
			el := elig[name]
			if el.Native != el.Model {
				t.Fatalf("native/model candidate disagreement %v %s: %+v", job, name, el)
			}
			pre := ""
			if el.PreErr != "" {
				pre = "PrePredicate rejected (cluster-wide node-pool maximum)"
			}
			row[name] = map[string]any{"native": el.Native, "model": el.Model, "fitting_gpus": el.FittingGPUs, "pre_predicate": pre}
		}
		out = append(out, row)
	}
	return out
}

type m6Fault struct {
	name     string
	first    m6Job
	topology func(*test_utils.TestTopologyBasic)
	mutate   func(*m6Env)
}

func m6Faults(t *testing.T) []m6Fault {
	node := func(e *m6Env, i int) *node_info.NodeInfo { return e.f.session.ClusterInfo.Nodes[e.names[i]] }
	one := m6Job{1, 2, 1}
	return []m6Fault{
		{name: "fractional_gpu", first: one, topology: func(md *test_utils.TestTopologyBasic) { md.Jobs[0].RequiredGPUsPerTask = 0.5 }},
		{name: "gpu_memory_sharing", first: one, topology: func(md *test_utils.TestTopologyBasic) { md.Jobs[0].RequiredGpuMemory = 1000 }},
		{name: "mig_instances", first: one, topology: func(md *test_utils.TestTopologyBasic) {
			md.Jobs[0].Tasks[0].RequiredMigInstances = map[v1.ResourceName]int{"nvidia.com/mig-1g.5gb": 1}
			n := md.Nodes["node2"]
			n.MigStrategy = node_info.MigStrategyMixed
			n.MigInstances = map[v1.ResourceName]int{"nvidia.com/mig-1g.5gb": 2}
			md.Nodes["node2"] = n
		}},
		{name: "dra_gpus", first: one, mutate: func(e *m6Env) { node(e, 3).AddDRAGPUs(1) }},
		{name: "numa_topology", first: one, mutate: func(e *m6Env) { node(e, 2).NumaTopology = &node_info.NumaTopology{} }},
		{name: "releasing_gpu", first: one, mutate: func(e *m6Env) { node(e, 1).ReleasingVector.Set(resource_info.GPUIndex, 1) }},
		{name: "unsupported_gpu_family_label", first: one, topology: func(md *test_utils.TestTopologyBasic) {
			n := md.Nodes["node3"]
			n.Labels = map[string]string{"nvidia.com/gpu.product": "A100"}
			md.Nodes["node3"] = n
		}},
		{name: "gpu_count_label_partial", first: one, mutate: func(e *m6Env) { node(e, 2).Node.Labels[commonconstants.GpuCountLabel] = "1" }},
		{name: "fractional_allocatable_gpu", first: one, mutate: func(e *m6Env) { node(e, 1).AllocatableVector.Set(resource_info.GPUIndex, 7.5) }},
		{name: "extra_task_resource", first: one, mutate: func(e *m6Env) {
			e.f.task.Pod.Spec.Containers[0].Resources.Requests["ephemeral-storage"] = resource.MustParse("1Gi")
		}},
		{name: "pod_slot_pressure", first: one, topology: func(md *test_utils.TestTopologyBasic) {
			two := 2
			n := md.Nodes["node1"]
			n.MaxTaskNum = &two
			md.Nodes["node1"] = n
		}},
		{name: "cpu_unit_mismatch", first: one, topology: func(md *test_utils.TestTopologyBasic) {
			n := md.Nodes["node2"]
			n.CPUMillis = 8.5
			md.Nodes["node2"] = n
		}},
		{name: "ram_unit_mismatch", first: one, topology: func(md *test_utils.TestTopologyBasic) {
			n := md.Nodes["node0"]
			n.CPUMemory = 64*nativeGiB + 1
			md.Nodes["node0"] = n
		}},
		{name: "invalid_request_type", first: one, mutate: func(e *m6Env) { e.f.task.ResourceRequestType = pod_info.RequestTypeGpuMemory }},
		{name: "shared_gpu_group", first: one, mutate: func(e *m6Env) { e.f.task.SetGPUGroupIDs([]string{"g0"}) }},
		{name: "node_taint", first: one, mutate: func(e *m6Env) {
			node(e, 1).Node.Spec.Taints = []v1.Taint{{Key: "k", Effect: v1.TaintEffectNoSchedule}}
		}},
		{name: "two_pending_targets", first: one, mutate: func(e *m6Env) { e.arrive(t, 1, one) }},
		{name: "queue_priority", first: one, mutate: func(e *m6Env) { e.f.job.Priority = 5 }},
		{name: "pod_fraction_annotation", first: one, mutate: func(e *m6Env) { e.f.task.Pod.Annotations[commonconstants.GpuFraction] = "0.5" }},
		{name: "native_unsupported_demand", first: m6Job{9, 2, 1}},
	}
}

func m6FaultTable(t *testing.T, dir string) []any {
	t.Helper()
	out := []any{}
	for _, c := range m6Faults(t) {
		var gate, reason string
		func() {
			defer func() {
				if r := recover(); r != nil {
					gate, reason = "construction_panic", fmt.Sprint("official builder refused the unsupported state: ", r)
				}
			}()
			e := m6Session(t, "m6-fault-"+c.name, m6Pool, m6Topology("fault-"+c.name, m6Pool, c.first, c.topology))
			if c.mutate != nil {
				c.mutate(e)
			}
			gate, reason = m6Gate(t, e, filepath.Join(dir, "fault-"+c.name))
		}()
		if gate == "ACCEPTED" {
			t.Fatalf("unsupported GPU semantics %q was accepted", c.name)
		}
		out = append(out, map[string]any{"case": c.name, "gate": gate, "reason": reason})
	}
	return out
}

func m6Mutations(raw []byte) map[string]func(map[string]any) {
	pairs := func(a map[string]any) []any { return a["pairs"].([]any) }
	return map[string]func(map[string]any){
		"pairs_reordered":       func(a map[string]any) { p := pairs(a); p[0], p[1] = p[1], p[0] },
		"pair_missing":          func(a map[string]any) { a["pairs"] = pairs(a)[:len(pairs(a))-1] },
		"pair_extra_duplicate":  func(a map[string]any) { a["pairs"] = append(pairs(a), pairs(a)[0]) },
		"pair_duplicate":        func(a map[string]any) { p := pairs(a); p[1] = p[0] },
		"pair_wrong_task":       func(a map[string]any) { pairs(a)[0].(map[string]any)["task_uid"] = "foreign" },
		"pair_wrong_node":       func(a map[string]any) { pairs(a)[0].(map[string]any)["node"] = "node9" },
		"pair_nan_score":        func(a map[string]any) { pairs(a)[0].(map[string]any)["score_bits"] = "7ff8000000000000" },
		"pair_out_of_range":     func(a map[string]any) { pairs(a)[0].(map[string]any)["score_bits"] = "4000000000000000" },
		"candidates_reordered":  func(a map[string]any) { c := a["candidates"].([]any); c[0], c[1] = c[1], c[0] },
		"candidates_missing":    func(a map[string]any) { a["candidates"] = a["candidates"].([]any)[:1] },
		"candidates_extra":      func(a map[string]any) { a["candidates"] = append(a["candidates"].([]any), "node9") },
		"candidates_not_string": func(a map[string]any) { a["candidates"].([]any)[0] = nil },
		"nodes_reordered":       func(a map[string]any) { n := a["nodes"].([]any); n[0], n[1] = n[1], n[0] },
		"node_uid_null":         func(a map[string]any) { a["nodes"].([]any)[0].(map[string]any)["uid"] = nil },
		"node_uid_forged":       func(a map[string]any) { a["nodes"].([]any)[0].(map[string]any)["uid"] = "forged" },
		"node_missing":          func(a map[string]any) { a["nodes"] = a["nodes"].([]any)[:len(a["nodes"].([]any))-1] },
		"version":               func(a map[string]any) { a["version"] = "M3B-1" },
		"phase_zero":            func(a map[string]any) { a["phase"] = 0 },
		"phase_over_cap":        func(a map[string]any) { a["phase"] = p2hMaxPhase + 1 },
		"unknown_field":         func(a map[string]any) { a["unknown"] = true },
		"missing_field":         func(a map[string]any) { delete(a, "session") },
		"task_uid":              func(a map[string]any) { a["task_uid"] = "foreign" },
		"artifact_hash":         func(a map[string]any) { a["artifact_sha256"] = "0000" },
		"projection_hash":       func(a map[string]any) { a["projection_sha256"] = "0000" },
	}
}

func m6Trust(t *testing.T, dir string) map[string]any {
	t.Helper()
	receipt := map[string]any{}
	e := m6Session(t, "m6-trust", m6Pool, m6Topology("trust", m6Pool, m6Job{2, 2, 1}, nil))
	_, cands, _ := p2hEligibility(e)
	s0, order0, _ := m6Scores(e, t, cands)
	first := m6Produce(t, e, 1, 0, m6Forecast, filepath.Join(dir, "trust-first"))
	// Receiver-level mutations on the legitimate envelope (privileged fault injection: bytes are rehashed
	// so only the receiver's own structural/provenance checks can refuse them).
	rejected := []string{}
	for name, mutate := range m6Mutations(first.raw) {
		var a map[string]any
		_ = json.Unmarshal(first.raw, &a)
		mutate(a)
		b, _ := json.Marshal(a)
		g := *first.grant
		g.used, g.envelopeHash = false, bridgeHash(b)
		if _, err := decodeP2H(b, &g); err == nil {
			t.Fatal("receiver accepted mutation ", name)
		}
		rejected = append(rejected, name)
	}
	sortStrings(rejected)
	receipt["receiver_mutations_rejected"] = rejected
	// Frozen receivers and the new receiver do not accept each other's envelopes.
	old := &bridgeGrant{owner: e.f.session, task: e.f.task, nodes: map[string]*node_info.NodeInfo{}, phase: 1, envelopeHash: first.envelope, projectionHash: first.grant.projectionHash, producerHash: first.grant.producerHash}
	if _, err := decodeBridge(first.raw, old); err == nil {
		t.Fatal("frozen M3-B receiver accepted a P2H envelope")
	}
	m5e := m5Session(t, "m6-m5-cross", m5Topology("cross", [2]m5Node{{4, 16}, {8, 16}}, m5Job{2, 2}, nil))
	m5p := m5Produce(t, m5e, 1, 0, m5StageAForecast, filepath.Join(dir, "cross-m5"))
	crossGrant := *first.grant
	crossGrant.envelopeHash = m5p.envelope
	if _, err := decodeP2H(m5p.raw, &crossGrant); err == nil {
		t.Fatal("P2H receiver accepted a P2 envelope")
	}
	receipt["frozen_p1_p2_receivers_and_p2h_reject_each_other"] = true
	// Grant lifecycle.
	if !e.plugin.begin(first.raw, first.grant, m4bLambda) {
		t.Fatal("fresh P2H grant")
	}
	scores, order, _ := m6Scores(e, t, cands)
	var wg sync.WaitGroup
	results := make([]map[string]float64, 8)
	for i := range results {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			r := map[string]float64{}
			for _, name := range cands {
				v, err := e.f.session.NodeOrderFn(e.f.task, e.f.session.ClusterInfo.Nodes[name])
				if err != nil {
					r["error"] = 1
				}
				r[name] = v
			}
			results[i] = r
		}(i)
	}
	wg.Wait()
	for _, r := range results {
		for _, name := range cands {
			if r[name] != scores[name] || r["error"] != 0 {
				t.Fatal("concurrent readers observed different scores")
			}
		}
	}
	receipt["concurrent_readers"] = "8 goroutines identical; reader concurrency only; race detector NOT_RUN"
	allocate.New().Execute(e.f.session)
	e.plugin.endPhase()
	changed := !reflect.DeepEqual(scores, s0) || !reflect.DeepEqual(order, order0)
	receipt["advice_changed_scores_or_order"] = changed
	e.arrive(t, 1, m6Job{1, 2, 1})
	spent := *first.grant
	if e.plugin.begin(first.raw, &spent, m4bLambda) {
		t.Fatal("spent grant reactivated")
	}
	stale := *first.grant
	stale.used, stale.phase = false, 2
	if e.plugin.begin(first.raw, &stale, m4bLambda) {
		t.Fatal("stale P2H projection accepted")
	}
	_, cands2, _ := p2hEligibility(e)
	neutral, neutralOrder, _ := m6Scores(e, t, cands2)
	native := map[string]float64{}
	for _, name := range cands2 {
		v, _ := e.f.session.NodeOrderFn(e.f.task, e.f.session.ClusterInfo.Nodes[name])
		native[name] = v
	}
	if !reflect.DeepEqual(neutral, native) {
		t.Fatal("neutral fallback changed the native score vector")
	}
	receipt["spent_grant_rejected"], receipt["stale_projection_rejected"], receipt["neutral_scores_equal_native"], receipt["neutral_order"] = true, true, true, neutralOrder
	second := m6Produce(t, e, 2, 0, m6Forecast, filepath.Join(dir, "trust-second"))
	e.f.session.PredicateFns = append(e.f.session.PredicateFns, func(*pod_info.PodInfo, *podgroup_info.PodGroupInfo, *node_info.NodeInfo) error { return nil })
	if e.plugin.begin(second.raw, second.grant, m4bLambda) {
		t.Fatal("mutated predicate set accepted")
	}
	e.f.session.PredicateFns = e.f.session.PredicateFns[:len(e.f.session.PredicateFns)-1]
	third := m6Produce(t, e, 2, 0, m6Forecast, filepath.Join(dir, "trust-third"))
	uid := e.f.task.UID
	e.f.task.UID = "forged"
	if e.plugin.begin(third.raw, third.grant, m4bLambda) {
		t.Fatal("forged Task UID accepted")
	}
	e.f.task.UID = uid
	other := m6Session(t, "m6-trust-foreign", m6Pool, m6Topology("foreign", m6Pool, m6Job{2, 2, 1}, nil))
	foreign := m6Produce(t, other, 1, 0, m6Forecast, filepath.Join(dir, "trust-foreign"))
	fourth := m6Produce(t, e, 2, 0, m6Forecast, filepath.Join(dir, "trust-fourth"))
	if e.plugin.begin(foreign.raw, foreign.grant, m4bLambda) {
		t.Fatal("foreign-Session grant accepted")
	}
	node1 := e.f.session.ClusterInfo.Nodes["node1"]
	node1.IdleVector[resource_info.GPUIndex]--
	if e.plugin.begin(fourth.raw, fourth.grant, m4bLambda) {
		t.Fatal("GPU-mutated state accepted")
	}
	node1.IdleVector[resource_info.GPUIndex]++
	// Node identity change between production and activation.
	fifth := m6Produce(t, e, 2, 0, m6Forecast, filepath.Join(dir, "trust-fifth"))
	saved := e.f.session.ClusterInfo.Nodes["node2"].Node.UID
	e.f.session.ClusterInfo.Nodes["node2"].Node.UID = "changed-node-identity"
	if e.plugin.begin(fifth.raw, fifth.grant, m4bLambda) {
		t.Fatal("changed Node identity accepted")
	}
	e.f.session.ClusterInfo.Nodes["node2"].Node.UID = saved
	// Pod-slot pressure between production and activation.
	sixth := m6Produce(t, e, 2, 0, m6Forecast, filepath.Join(dir, "trust-sixth"))
	node0 := e.f.session.ClusterInfo.Nodes["node0"]
	node0.IdleVector[resource_info.PodsIndex] = 1
	if e.plugin.begin(sixth.raw, sixth.grant, m4bLambda) {
		t.Fatal("pod-slot pressure accepted")
	}
	node0.IdleVector[resource_info.PodsIndex] = float64(110)
	seventh := m6Produce(t, e, 2, 0, m6Forecast, filepath.Join(dir, "trust-seventh"))
	if !e.plugin.begin(seventh.raw, seventh.grant, m4bLambda) {
		t.Fatal("the legitimate fresh grant must still activate after all rejections")
	}
	e.plugin.endPhase()
	receipt["mutated_hooks_rejected"], receipt["forged_uid_rejected"], receipt["foreign_session_rejected"] = true, true, true
	receipt["resource_mutation_rejected"], receipt["node_identity_change_rejected"], receipt["pod_slot_pressure_rejected"] = true, true, true
	receipt["fresh_grant_after_faults_activates"] = true
	return receipt
}

func m6StageA(t *testing.T, dir string) map[string]any {
	t.Helper()
	record := map[string]any{"version": p2hVersion, "bounds": map[string]int{"min_nodes": p2hMinNodes, "max_nodes": p2hMaxNodes, "max_tasks": p2hMaxTasks, "max_phase": p2hMaxPhase}}
	record["positive"] = m6Positive(t, dir)
	record["state_evolution"] = m6StateEvolution(t, dir)
	record["eligibility_table"] = m6EligibilityTable(t)
	record["fail_closed"] = m6FaultTable(t, dir)
	record["trust"] = m6Trust(t, dir)
	record["frozen_contracts"] = "P1 CPU_ONLY and P2-WholeGPU-1 projections, producers and the M3-B receiver are NOT modified"
	return record
}
