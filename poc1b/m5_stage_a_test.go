package main

// M5 Stage A tests: authentic whole-GPU semantics, P2-WholeGPU-1 mapping proof, native-vs-model
// eligibility equivalence on a restricted domain, fail-closed rejection table and grant trust.

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

var m5Nodes = [2]m5Node{{4, 16}, {8, 16}}
var m5StageAForecast = []m4bForecast{{ID: "F0", Demand: []int{4, 2, 1}, Weight: 2, Eligible: m4bBoth}}

func m5Basic(t *testing.T, name string, first m5Job, mutate func(*test_utils.TestTopologyBasic)) *m5Env {
	t.Helper()
	e := m5Session(t, "m5-"+name, m5Topology(name, m5Nodes, first, mutate))
	e.nodes = m5Nodes
	return e
}

// m5Gate classifies which gate refuses a state: the Go P2 projection or the independent Python producer.
func m5Gate(t *testing.T, e *m5Env, dir string) (string, string) {
	t.Helper()
	praw, err := p2Projection(t, e)
	if err != nil {
		return "go_projection", err.Error()
	}
	_, _, runErr := m5Run(t, praw, 1, 0, m5StageAForecast, dir)
	if runErr == nil {
		return "ACCEPTED", ""
	}
	b, _ := os.ReadFile(filepath.Join(dir, "producer.stderr.txt"))
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	return "python_producer", strings.TrimSpace(lines[len(lines)-1])
}

func m5MappingProof(t *testing.T) map[string]any {
	t.Helper()
	e := m5Basic(t, "mapping", m5Job{2, 2}, nil)
	task := e.f.task
	names := []string{}
	for i := 0; i < task.VectorMap.Len(); i++ {
		names = append(names, string(task.VectorMap.ResourceAt(i)))
	}
	want := map[int]string{resource_info.CPUIndex: "cpu", resource_info.MemoryIndex: "memory", resource_info.GPUIndex: "gpu", resource_info.PodsIndex: "pods"}
	for i, w := range want {
		if names[i] != w {
			t.Fatal("native index map differs from the P2 contract: ", i, names[i], w)
		}
	}
	req := task.Pod.Spec.Containers[0].Resources.Requests
	gpu := req[resource_info.GPUResourceName]
	if task.GpuRequirement.GPUs() != 2 || task.ResourceRequestType != pod_info.RequestTypeRegular || task.ResReqVector.Get(resource_info.GPUIndex) != 2 || gpu.Value() != 2 ||
		task.ResReqVector.Get(resource_info.CPUIndex) != 2000 || task.ResReqVector.Get(resource_info.MemoryIndex) != nativeGiB || task.ResReqVector.Get(resource_info.PodsIndex) != 1 {
		t.Fatal("authentic whole-GPU Pod does not satisfy GpuRequirement/ResourceVector/Pod request identity")
	}
	for _, name := range m4bBoth {
		n := e.f.session.ClusterInfo.Nodes[name]
		if n.AllocatableVector.Get(resource_info.GPUIndex) != float64(e.nodes[map[string]int{"node0": 0, "node1": 1}[name]].GPUs) {
			t.Fatal("authentic GPU node allocatable mismatch")
		}
	}
	podReq := map[string]string{}
	for k, q := range req {
		podReq[string(k)] = q.String()
	}
	return map[string]any{"index_names": names, "pod_requests": podReq, "gpu_requirement_gpus": task.GpuRequirement.GPUs(), "request_type": string(task.ResourceRequestType),
		"res_req_vector": task.ResReqVector.Clone(), "accepted_vector": task.AcceptedResourceVector.Clone(),
		"node_allocatable": map[string]any{"node0": e.f.session.ClusterInfo.Nodes["node0"].AllocatableVector.Clone(), "node1": e.f.session.ClusterInfo.Nodes["node1"].AllocatableVector.Clone()},
		"unit_rule":        "model = [GPU, CPU milli/1000, RAM bytes/2^30]; any remainder or fractional GPU is rejected"}
}

func m5Positive(t *testing.T, dir string) []any {
	t.Helper()
	out := []any{}
	for i, first := range []m5Job{{1, 2}, {2, 4}, {4, 2}} {
		e := m5Basic(t, fmt.Sprintf("pos%d", i), first, nil)
		praw, err := p2Projection(t, e)
		if err != nil {
			t.Fatal("authentic whole-GPU state refused: ", err)
		}
		p, _, runErr := m5Run(t, praw, 1, 0, m5StageAForecast, filepath.Join(dir, fmt.Sprintf("positive-%d", i)))
		if runErr != nil {
			t.Fatal("producer refused an authentic supported state: ", runErr, p.stderr)
		}
		out = append(out, map[string]any{"first": first, "projection_sha256": bridgeHash(praw), "envelope_sha256": p.envelope, "artifact_sha256": m4bHash(p.artifact)})
	}
	// A used state: one whole-GPU task bound by actual native Allocate, then a second authentic arrival.
	e := m5Basic(t, "used", m5Job{2, 2}, nil)
	before := m4bHash(mustProjection(t, e))
	allocate.New().Execute(e.f.session)
	if p := e.f.job.GetAllPodsMap()["native-job-0"]; p.Status != pod_status.Binding {
		t.Fatal("native Allocate did not bind the whole-GPU Pod")
	}
	e.arrive(t, 1, m5Job{1, 2})
	praw := mustProjection(t, e)
	p, _, runErr := m5Run(t, praw, 1, 0, m5StageAForecast, filepath.Join(dir, "positive-used"))
	if runErr != nil {
		t.Fatal(runErr, p.stderr)
	}
	node := e.f.job.GetAllPodsMap()["native-job-0"].NodeName
	out = append(out, map[string]any{"after_bind_node": node, "projection_sha256": bridgeHash(praw), "differs_from_initial": bridgeHash(praw) != before,
		"used_idle": []any{e.f.session.ClusterInfo.Nodes[node].UsedVector.Clone(), e.f.session.ClusterInfo.Nodes[node].IdleVector.Clone()}})
	return out
}

func mustProjection(t *testing.T, e *m5Env) []byte {
	t.Helper()
	b, err := p2Projection(t, e)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func m5EligibilityTable(t *testing.T) []any {
	t.Helper()
	out := []any{}
	for _, job := range []m5Job{{1, 2}, {2, 2}, {4, 2}, {5, 2}, {8, 2}, {9, 2}, {1, 16}, {1, 17}, {4, 16}, {8, 16}, {8, 17}} {
		e := m5Basic(t, fmt.Sprintf("elig-%dg-%dc", job.GPUs, job.Cores), job, nil)
		elig, err := p2EligibilityFor(e)
		if err != nil {
			t.Fatal(err)
		}
		row := map[string]any{"request": job}
		feasible := 0
		for _, name := range m4bBoth {
			el := elig[name]
			if el.Native != el.Model {
				t.Fatalf("native/model candidate disagreement %v %s: %+v", job, name, el)
			}
			if el.Native {
				feasible++
				if el.FittingGPUs < job.GPUs {
					t.Fatal("FittingGPUs smaller than a feasible whole-GPU request")
				}
			}
			pre := ""
			if el.PreErr != "" {
				pre = "PrePredicate rejected (cluster-wide node-pool maximum)"
			}
			row[name] = map[string]any{"native": el.Native, "model": el.Model, "fitting_gpus": el.FittingGPUs, "pre_predicate": pre}
		}
		row["feasible_nodes"] = feasible
		out = append(out, row)
	}
	return out
}

type m5Fault struct {
	name     string
	topology func(*test_utils.TestTopologyBasic)
	mutate   func(*m5Env)
	first    m5Job
}

func m5Faults(t *testing.T) []m5Fault {
	node0 := func(e *m5Env) *node_info.NodeInfo { return e.f.session.ClusterInfo.Nodes["node0"] }
	return []m5Fault{
		{name: "fractional_gpu", first: m5Job{1, 2}, topology: func(md *test_utils.TestTopologyBasic) { md.Jobs[0].RequiredGPUsPerTask = 0.5 }},
		{name: "gpu_memory_sharing", first: m5Job{1, 2}, topology: func(md *test_utils.TestTopologyBasic) { md.Jobs[0].RequiredGpuMemory = 1000 }},
		{name: "mig_instances", first: m5Job{1, 2}, topology: func(md *test_utils.TestTopologyBasic) {
			md.Jobs[0].Tasks[0].RequiredMigInstances = map[v1.ResourceName]int{"nvidia.com/mig-1g.5gb": 1}
			n := md.Nodes["node0"]
			n.MigStrategy = node_info.MigStrategyMixed
			n.MigInstances = map[v1.ResourceName]int{"nvidia.com/mig-1g.5gb": 2}
			md.Nodes["node0"] = n
		}},
		{name: "dra_gpus", first: m5Job{1, 2}, mutate: func(e *m5Env) { node0(e).AddDRAGPUs(1) }},
		{name: "numa_topology", first: m5Job{1, 2}, mutate: func(e *m5Env) { node0(e).NumaTopology = &node_info.NumaTopology{} }},
		{name: "releasing_gpu", first: m5Job{1, 2}, mutate: func(e *m5Env) { node0(e).ReleasingVector.Set(resource_info.GPUIndex, 1) }},
		{name: "unsupported_gpu_family_label", first: m5Job{1, 2}, topology: func(md *test_utils.TestTopologyBasic) {
			n := md.Nodes["node0"]
			n.Labels = map[string]string{"nvidia.com/gpu.product": "A100"}
			md.Nodes["node0"] = n
		}},
		{name: "gpu_count_label_partial", first: m5Job{1, 2}, mutate: func(e *m5Env) { node0(e).Node.Labels[commonconstants.GpuCountLabel] = "3" }},
		{name: "fractional_allocatable_gpu", first: m5Job{1, 2}, mutate: func(e *m5Env) {
			node0(e).AllocatableVector.Set(resource_info.GPUIndex, 3.5)
		}},
		{name: "extra_task_resource", first: m5Job{1, 2}, mutate: func(e *m5Env) {
			r := e.f.task.Pod.Spec.Containers[0].Resources.Requests
			r["ephemeral-storage"] = resource.MustParse("1Gi")
		}},
		{name: "pod_slot_bottleneck", first: m5Job{1, 2}, topology: func(md *test_utils.TestTopologyBasic) {
			two := 2
			n := md.Nodes["node0"]
			n.MaxTaskNum = &two
			md.Nodes["node0"] = n
		}},
		{name: "cpu_unit_mismatch", first: m5Job{1, 2}, topology: func(md *test_utils.TestTopologyBasic) {
			n := md.Nodes["node0"]
			n.CPUMillis = 16.5
			md.Nodes["node0"] = n
		}},
		{name: "ram_unit_mismatch", first: m5Job{1, 2}, topology: func(md *test_utils.TestTopologyBasic) {
			n := md.Nodes["node0"]
			n.CPUMemory = m5MemGiB*nativeGiB + 1
			md.Nodes["node0"] = n
		}},
		{name: "invalid_request_type", first: m5Job{1, 2}, mutate: func(e *m5Env) { e.f.task.ResourceRequestType = pod_info.RequestTypeGpuMemory }},
		{name: "shared_gpu_group", first: m5Job{1, 2}, mutate: func(e *m5Env) { e.f.task.SetGPUGroupIDs([]string{"g0"}) }},
		{name: "node_taint", first: m5Job{1, 2}, mutate: func(e *m5Env) {
			node0(e).Node.Spec.Taints = []v1.Taint{{Key: "k", Effect: v1.TaintEffectNoSchedule}}
		}},
		{name: "two_pending_targets", first: m5Job{1, 2}, mutate: func(e *m5Env) { e.arrive(t, 1, m5Job{1, 2}) }},
		{name: "queue_priority", first: m5Job{1, 2}, mutate: func(e *m5Env) { e.f.job.Priority = 5 }},
		{name: "pod_fraction_annotation", first: m5Job{1, 2}, mutate: func(e *m5Env) { e.f.task.Pod.Annotations[commonconstants.GpuFraction] = "0.5" }},
		{name: "one_node_ineligible", first: m5Job{6, 2}},
		{name: "no_node_eligible", first: m5Job{9, 2}},
	}
}

func m5FaultTable(t *testing.T, dir string) []any {
	t.Helper()
	out := []any{}
	for _, c := range m5Faults(t) {
		var rejected string
		var gate string
		func() {
			defer func() {
				if r := recover(); r != nil {
					gate, rejected = "construction_panic", fmt.Sprint("official builder refused the unsupported state: ", r)
				}
			}()
			e := m5Basic(t, "fault-"+c.name, c.first, c.topology)
			if c.mutate != nil {
				c.mutate(e)
			}
			gate, rejected = m5Gate(t, e, filepath.Join(dir, "fault-"+c.name))
		}()
		if gate == "ACCEPTED" {
			t.Fatalf("unsupported GPU semantics %q was accepted", c.name)
		}
		out = append(out, map[string]any{"case": c.name, "gate": gate, "reason": rejected})
	}
	return out
}

func m5TrustFaults(t *testing.T, dir string) map[string]any {
	t.Helper()
	receipt := map[string]any{}
	e := m5Basic(t, "trust", m5Job{2, 2}, nil)
	first := m5Produce(t, e, 1, 0, m5StageAForecast, filepath.Join(dir, "trust-first"))
	s0, order0, _ := m4bScoresFor(t, e.f, e.plugin)
	if !e.plugin.begin(first.raw, first.grant, m4bLambda) {
		t.Fatal("fresh P2 grant")
	}
	scores, order, _ := m4bScoresFor(t, e.f, e.plugin)
	// Concurrent read-only scoring while the phase is active: identical results, unchanged candidate set.
	var wg sync.WaitGroup
	results := make([]map[string]float64, 8)
	for i := range results {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			r := map[string]float64{}
			for _, name := range m4bBoth {
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
		if r["node0"] != scores["node0"] || r["node1"] != scores["node1"] || r["error"] != 0 {
			t.Fatal("concurrent readers observed different scores", r, scores)
		}
	}
	receipt["concurrent_readers"] = "8 goroutines identical; reader concurrency only; race detector NOT_RUN"
	allocate.New().Execute(e.f.session)
	e.plugin.endPhase()
	if !(scores["node0"] != s0["node0"] || scores["node1"] != s0["node1"] || reflect.DeepEqual(order, order0)) {
		t.Fatal("advice neither changed scores nor kept order")
	}
	e.arrive(t, 1, m5Job{1, 2})
	// 1: spent grant. 2: unspent copy of a stale grant. 3: mutated hooks. 4: UID mutation. 5: foreign Session.
	spent := *first.grant
	if e.plugin.begin(first.raw, &spent, m4bLambda) {
		t.Fatal("spent grant reactivated")
	}
	stale := *first.grant
	stale.used, stale.phase = false, 2
	if e.plugin.begin(first.raw, &stale, m4bLambda) {
		t.Fatal("stale P2 projection accepted")
	}
	neutral, neutralOrder, _ := m4bScoresFor(t, e.f, e.plugin)
	nativeOnly := m4bNativeOnly(t, e)
	if !reflect.DeepEqual(neutral, nativeOnly) {
		t.Fatal("neutral fallback changed the native score vector")
	}
	receipt["spent_grant_rejected"], receipt["stale_projection_rejected"] = true, true
	receipt["neutral_scores_equal_native"], receipt["neutral_order"] = true, neutralOrder
	second := m5Produce(t, e, 2, 0, m5StageAForecast, filepath.Join(dir, "trust-second"))
	e.f.session.PredicateFns = append(e.f.session.PredicateFns, func(*pod_info.PodInfo, *podgroup_info.PodGroupInfo, *node_info.NodeInfo) error { return nil })
	if e.plugin.begin(second.raw, second.grant, m4bLambda) {
		t.Fatal("mutated predicate set accepted")
	}
	e.f.session.PredicateFns = e.f.session.PredicateFns[:len(e.f.session.PredicateFns)-1]
	receipt["mutated_hooks_rejected"] = true
	third := m5Produce(t, e, 2, 0, m5StageAForecast, filepath.Join(dir, "trust-third"))
	uid := e.f.task.UID
	e.f.task.UID = "forged"
	if e.plugin.begin(third.raw, third.grant, m4bLambda) {
		t.Fatal("forged Task UID accepted")
	}
	e.f.task.UID = uid
	receipt["forged_uid_rejected"] = true
	other := m5Basic(t, "trust-foreign", m5Job{2, 2}, nil)
	foreign := m5Produce(t, other, 1, 0, m5StageAForecast, filepath.Join(dir, "trust-foreign"))
	fourth := m5Produce(t, e, 2, 0, m5StageAForecast, filepath.Join(dir, "trust-fourth"))
	if e.plugin.begin(foreign.raw, foreign.grant, m4bLambda) {
		t.Fatal("foreign-Session grant accepted")
	}
	receipt["foreign_session_rejected"] = true
	e.f.session.ClusterInfo.Nodes["node0"].IdleVector[resource_info.GPUIndex]--
	if e.plugin.begin(fourth.raw, fourth.grant, m4bLambda) {
		t.Fatal("GPU-mutated state accepted")
	}
	e.f.session.ClusterInfo.Nodes["node0"].IdleVector[resource_info.GPUIndex]++
	receipt["resource_mutation_rejected"] = true
	// Duplicate/incomplete mapping and tampered bytes (privileged fault injection, rehashed).
	fifth := m5Produce(t, e, 2, 0, m5StageAForecast, filepath.Join(dir, "trust-fifth"))
	var obj map[string]any
	_ = json.Unmarshal(fifth.raw, &obj)
	for i, mutate := range []func(map[string]any){
		func(a map[string]any) { a["pairs"] = a["pairs"].([]any)[:1] },
		func(a map[string]any) { p := a["pairs"].([]any); a["pairs"] = []any{p[0], p[0]} },
		func(a map[string]any) { a["nodes"].([]any)[0].(map[string]any)["name"] = "node1" },
		func(a map[string]any) { a["version"] = "P2" },
	} {
		var a map[string]any
		_ = json.Unmarshal(fifth.raw, &a)
		mutate(a)
		b, _ := json.Marshal(a)
		g := *fifth.grant
		g.used, g.envelopeHash = false, bridgeHash(b)
		if _, err := decodeBridge(b, &g); err == nil {
			t.Fatal("incomplete/duplicate/tampered mapping accepted ", i)
		}
	}
	receipt["incomplete_duplicate_mapping_rejected"] = true
	if !e.plugin.begin(fifth.raw, fifth.grant, m4bLambda) {
		t.Fatal("the legitimate fresh P2 grant must still activate after all rejections")
	}
	e.plugin.endPhase()
	receipt["fresh_grant_after_faults_activates"] = true
	return receipt
}

func m5StageA(t *testing.T, dir string) map[string]any {
	t.Helper()
	record := map[string]any{"version": p2Version}
	record["mapping_proof"] = m5MappingProof(t)
	record["positive"] = m5Positive(t, dir)
	record["eligibility_table"] = m5EligibilityTable(t)
	record["fail_closed"] = m5FaultTable(t, dir)
	record["trust"] = m5TrustFaults(t, dir)
	record["p1_cpu_only_contract"] = "UNCHANGED (bridgeProjection/producer not modified)"
	return record
}

// m4bScoresFor / m4bNativeOnly adapt the M4-B score helpers to the M5 environment.
func m4bScoresFor(t *testing.T, f nativeFixture, _ *artifactBridgeProbe) (map[string]float64, []string, float64) {
	t.Helper()
	e := &m4bEnv{f: f}
	return m4bScores(e, t)
}

func m4bNativeOnly(t *testing.T, e *m5Env) map[string]float64 {
	t.Helper()
	scores := map[string]float64{}
	for _, name := range m4bBoth {
		v, err := e.f.session.NodeOrderFn(e.f.task, e.f.session.ClusterInfo.Nodes[name])
		if err != nil {
			t.Fatal(err)
		}
		scores[name] = v
	}
	return scores
}
