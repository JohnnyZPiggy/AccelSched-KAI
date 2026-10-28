package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"testing"
	"time"

	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/actions/allocate"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/common_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/node_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/pod_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/pod_status"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/resource_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/cache"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/framework"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/test_utils"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/test_utils/tasks_fake"
	commonconstants "github.com/kai-scheduler/api/constants"
	"go.uber.org/mock/gomock"
	v1 "k8s.io/api/core/v1"
)

func bridgeDir(t *testing.T) string {
	t.Helper()
	if d := os.Getenv("POC1B_BRIDGE_EVIDENCE_DIR"); d != "" {
		if e := os.MkdirAll(d, 0700); e != nil {
			t.Fatal(e)
		}
		return d
	}
	return t.TempDir()
}
func bridgeVector(v resource_info.ResourceVector) ([]int64, error) {
	r := []int64{}
	for _, k := range []int{resource_info.GPUIndex, resource_info.CPUIndex, resource_info.MemoryIndex} {
		x := v.Get(k)
		if !finiteAdvice(x) || x < 0 || x != math.Trunc(x) || x > 1e12 {
			return nil, fmt.Errorf("noninteger/unbounded native resource")
		}
		r = append(r, int64(x))
	}
	return r, nil
}

// P1 is deliberately a strict allow-list of the pinned B4 CPU fixture and its
// one-task arrival extension. Full public projections retain pod-slot, queue,
// accepted/bookkeeping state; supported resources are never confused with total.
func bridgeProjection(t *testing.T, f nativeFixture) ([]byte, error) {
	s := f.session
	if s == nil || s.ClusterInfo == nil || len(s.ClusterInfo.PodGroupInfos) != 1 || len(s.ClusterInfo.Nodes) != 2 || len(s.ClusterInfo.Topologies) != 0 || len(s.ClusterInfo.ResourceClaims) != 0 || len(s.ClusterInfo.ResourceSlices) != 0 {
		return nil, fmt.Errorf("unsupported native membership/features")
	}
	job := s.ClusterInfo.PodGroupInfos[common_info.NewPodGroupID("", "native-job")]
	if job == nil || string(job.Queue) != "queue0" || job.Priority != 0 || len(job.GetAllPodsMap()) > 3 {
		return nil, fmt.Errorf("unsupported job")
	}
	for _, set := range job.PodSets {
		if set.GetMinAvailable() != 1 {
			return nil, fmt.Errorf("gang unsupported")
		}
	}
	nodes := []any{}
	for _, name := range []string{"node0", "node1"} {
		n := s.ClusterInfo.Nodes[name]
		if n == nil || n.Name != name || n.Node == nil || n.Node.Name != name || n.NumaTopology != nil || n.NodeResourceTopology != nil || n.HasDRAGPUs || len(n.LegacyMIGTasks) != 0 || len(n.Node.Spec.Taints) != 0 || n.MaxTaskNum < 4 {
			return nil, fmt.Errorf("unsupported node semantics")
		}
		labels := map[string]string{commonconstants.GpuCountLabel: "0", node_info.GpuMemoryLabel: "100", "node-role.kubernetes.io/mig-enabled": "false", commonconstants.MigStrategyLabel: "", tasks_fake.NodeAffinityKey: name}
		if !reflect.DeepEqual(n.Node.Labels, labels) {
			return nil, fmt.Errorf("unknown node labels/family")
		}
		entry := map[string]any{"name": name, "uid": string(n.Node.UID), "shape": "S", "family": "CPU_ONLY_TEST", "native": map[string]any{"spec": n.Node.Spec, "labels": n.Node.Labels, "annotations": n.Node.Annotations, "status": n.Node.Status}}
		for key, v := range map[string]resource_info.ResourceVector{"allocatable": n.AllocatableVector, "idle": n.IdleVector, "used": n.UsedVector, "releasing": n.ReleasingVector} {
			x, e := bridgeVector(v)
			if e != nil {
				return nil, e
			}
			entry[key] = x
		}
		members := []string{}
		for uid := range n.PodInfos {
			members = append(members, string(uid))
		}
		sort.Strings(members)
		entry["members"] = members
		nodes = append(nodes, entry)
	}
	tasks := []any{}
	ids := []string{}
	for id := range job.GetAllPodsMap() {
		ids = append(ids, string(id))
	}
	sort.Strings(ids)
	pending := 0
	for _, id := range ids {
		p := job.GetAllPodsMap()[common_info.PodID(id)]
		if p == nil || p.Pod == nil || string(p.Pod.UID) != id || p.ResourceRequestType != pod_info.RequestTypeRegular || !p.NUMAPlacement.IsEmpty() || p.GpuRequirement.GPUs() != 0 || p.GpuRequirement.GpuMemory() != 0 || len(p.GpuRequirement.MigResources()) != 0 || len(p.GpuRequirement.DraGpuCounts()) != 0 || p.VectorMap.Len() != 4 {
			return nil, fmt.Errorf("unsupported task/resources")
		}
		var spec map[string]json.RawMessage
		b, _ := json.Marshal(p.Pod.Spec)
		_ = json.Unmarshal(b, &spec)
		for key := range spec {
			if key == "affinity" && p.Pod.Spec.Affinity != nil && reflect.DeepEqual(*p.Pod.Spec.Affinity, v1.Affinity{}) {
				continue
			}
			if key != "containers" && key != "nodeName" && key != "schedulerName" {
				return nil, fmt.Errorf("unrepresented PodSpec %s", key)
			}
		}
		if len(p.Pod.Spec.Containers) != 1 || len(p.Pod.Spec.Containers[0].Resources.Limits) != 0 {
			return nil, fmt.Errorf("unsupported container")
		}
		req := p.Pod.Spec.Containers[0].Resources.Requests
		for k := range req {
			q := req[k]
			if k == v1.ResourcePods && q.Value() == 1 && q.MilliValue() == 1000 {
				continue
			}
			if k == resource_info.GPUResourceName && q.IsZero() {
				continue
			}
			if k != v1.ResourceCPU && k != v1.ResourceMemory {
				return nil, fmt.Errorf("extra resource %s=%s (all %v)", k, q.String(), req)
			}
		}
		expectedContainer := v1.Container{Name: "main", Resources: v1.ResourceRequirements{Requests: req}}
		if !reflect.DeepEqual(p.Pod.Spec.Containers[0], expectedContainer) || p.Pod.Spec.SchedulerName != "kai-scheduler" {
			return nil, fmt.Errorf("unsupported container/predicate semantics")
		}
		if float64(req.Cpu().MilliValue()) != p.ResReqVector.Get(resource_info.CPUIndex) || float64(req.Memory().Value()) != p.ResReqVector.Get(resource_info.MemoryIndex) {
			return nil, fmt.Errorf("native request/Pod unit mismatch")
		}
		r, e := bridgeVector(p.ResReqVector)
		if e != nil {
			return nil, e
		}
		a, e := bridgeVector(p.AcceptedResourceVector)
		if e != nil {
			return nil, e
		}
		if p.Status == pod_status.Pending {
			pending++
			if p.NodeName != "" {
				return nil, fmt.Errorf("Pending node assignment")
			}
		}
		tasks = append(tasks, map[string]any{"uid": id, "pod_uid": string(p.Pod.UID), "status": p.Status.String(), "node": p.NodeName, "request": r, "accepted": a, "native": map[string]any{"spec": p.Pod.Spec, "labels": p.Pod.Labels, "annotations": p.Pod.Annotations, "gpu": nativeGPU(&p.GpuRequirement), "numa": p.NUMAPlacement.Clone(), "job": p.Job}})
	}
	if pending != 1 {
		return nil, fmt.Errorf("one Pending target required")
	}
	// Eligibility is genuinely native, not inferred from spare capacity.
	for _, p := range job.GetAllPodsMap() {
		if p.Status == pod_status.Pending {
			if s.PrePredicateFn(p, job) != nil {
				return nil, fmt.Errorf("native prepredicate")
			}
			for _, name := range []string{"node0", "node1"} {
				if s.PredicateFn(p, job, s.ClusterInfo.Nodes[name]) != nil || !s.FittingNode(p, s.ClusterInfo.Nodes[name], false) {
					return nil, fmt.Errorf("native candidate not eligible")
				}
			}
		}
	}
	out := map[string]any{"version": "P1", "session": s.ID, "nodes": nodes, "tasks": tasks, "domain": probeDomain(t, f), "config": s.Config, "hooks": probeHooks(t, s), "semantics": "AUDITED_B4_CPU_ONLY_SINGLE_PENDING"}
	return json.Marshal(out)
}

// The sole capability factory lives in _test.go. A trusted local process,
// expected committed sources, successful validator exit and exact stdout bind
// the grant; user JSON/claimed status/digests cannot mint this in-process object.
func bridgeProduce(t *testing.T, f nativeFixture, phase int, dir string) ([]byte, *bridgeGrant) {
	t.Helper()
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	root, e := filepath.Abs("..")
	if e != nil {
		t.Fatal(e)
	}
	paths := []string{"src/accelsched_kai/bridge_envelope.py", "src/accelsched_kai/bounded_advice.py", "src/accelsched_kai/adapter.py", "src/accelsched_kai/oracle.py", "src/accelsched_kai/advisor.py", "src/accelsched_kai/schema.py"}
	var producerHash string
	for _, name := range paths {
		b, e := os.ReadFile(filepath.Join(root, name))
		if e != nil {
			t.Fatal(e)
		}
		b = bytes.ReplaceAll(b, []byte("\r\n"), []byte("\n"))
		cmd := exec.Command("git", "-C", root, "show", "HEAD:"+name)
		expected, e := cmd.Output()
		if e != nil || !bytes.Equal(b, expected) {
			t.Fatal("uncommitted/mismatched controlled producer", name, e)
		}
		if name == paths[0] {
			producerHash = bridgeHash(b)
		}
	}
	praw, e := bridgeProjection(t, f)
	if e != nil {
		t.Fatal(e)
	}
	request := nativeJSON(t, map[string]any{"projection_b64": base64.StdEncoding.EncodeToString(praw), "phase": phase})
	input := filepath.Join(dir, "request.json")
	if e = os.WriteFile(input, request, 0600); e != nil {
		t.Fatal(e)
	}
	python := os.Getenv("M3B_PYTHON")
	if python == "" {
		t.Fatal("M3B_PYTHON must identify existing frozen environment")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	code := "import runpy,sys;sys.path.insert(0,sys.argv[1]);sys.argv=sys.argv[2:];runpy.run_module('accelsched_kai.bridge_envelope',run_name='__main__')"
	output := filepath.Join(dir, "producer")
	cmd := exec.CommandContext(ctx, python, "-I", "-B", "-c", code, filepath.Join(root, "src"), "bridge_envelope", input, output)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	raw, err := cmd.Output()
	_ = os.WriteFile(filepath.Join(dir, "producer.stderr.txt"), stderr.Bytes(), 0600)
	nativeSave(t, dir, "process.json", map[string]any{"argv": cmd.Args, "exit_error": fmt.Sprint(err), "timeout_seconds": 30, "producer_sha256": producerHash, "envelope_sha256": bridgeHash(raw), "identity": "controlled local subprocess, committed sources checked; no crypto signature"})
	if err != nil {
		t.Fatal("controlled producer failed", err, stderr.String())
	}
	if file, e := os.ReadFile(filepath.Join(output, "envelope.json")); e != nil || !bytes.Equal(file, raw) {
		t.Fatal("producer handoff mismatch")
	}
	var hooks []uintptr
	for _, fn := range f.session.PredicateFns {
		hooks = append(hooks, reflect.ValueOf(fn).Pointer())
	}
	grant := &bridgeGrant{owner: f.session, task: f.task, nodes: map[string]*node_info.NodeInfo{}, phase: phase, envelopeHash: bridgeHash(raw), projectionHash: bridgeHash(praw), producerHash: producerHash}
	for name, n := range f.session.ClusterInfo.Nodes {
		grant.nodes[name] = n
	}
	grant.project = func(s *framework.Session) ([]byte, error) {
		if s != f.session {
			return nil, fmt.Errorf("foreign Session")
		}
		now := []uintptr{}
		for _, fn := range s.PredicateFns {
			now = append(now, reflect.ValueOf(fn).Pointer())
		}
		if !reflect.DeepEqual(now, hooks) {
			return nil, fmt.Errorf("predicate implementation changed")
		}
		return bridgeProjection(t, f)
	}
	nativeSave(t, dir, "projection.json", json.RawMessage(praw))
	return raw, grant
}

func bridgeAction(t *testing.T, f nativeFixture, mock *cache.MockCache, want string) probeActionReceipt {
	t.Helper()
	uid := f.task.UID
	r := probeActionReceipt{Before: probeDomain(t, f), Config: nativeJSON(t, f.session.Config), Bind: []json.RawMessage{}, Pipeline: []json.RawMessage{}, Evict: []json.RawMessage{}, Result: "native Execute; actual fake Bind nil; no real Kubernetes"}
	mock.EXPECT().Bind(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).DoAndReturn(func(p *pod_info.PodInfo, n string, a map[string]string, numa cache.NUMAPrediction) error {
		r.Bind = append(r.Bind, nativeJSON(t, map[string]any{"task": nativeTaskState(p), "node": n, "annotations": a, "numa": numa, "returned_error": nil}))
		return nil
	}).Times(1)
	mock.EXPECT().TaskPipelined(gomock.Any(), gomock.Any()).Times(0)
	mock.EXPECT().Evict(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Times(0)
	allocate.New().Execute(f.session)
	j := f.session.ClusterInfo.PodGroupInfos[f.job.UID]
	p := j.GetAllPodsMap()[uid]
	if p == nil || p.Status != pod_status.Binding || p.NodeName != want {
		t.Fatal("native bridge Action", p, "want", want)
	}
	r.After = probeDomain(t, f)
	r.UID = string(uid)
	r.Node = p.NodeName
	r.Status = p.Status.String()
	return r
}

func bridgeScores(t *testing.T, f nativeFixture) map[string]float64 {
	t.Helper()
	s := f.session
	nodes := []*node_info.NodeInfo{s.ClusterInfo.Nodes["node1"], s.ClusterInfo.Nodes["node0"]}
	s.OrderedNodesByTask(nodes, f.task)
	r := map[string]float64{}
	for _, n := range nodes {
		v, e := s.NodeOrderFn(f.task, n)
		if e != nil {
			t.Fatal(e)
		}
		r[n.Name] = v
	}
	return r
}

func TestTrustedArtifactBridgeNative(t *testing.T) {
	test_utils.InitTestingInfrastructure()
	dir := bridgeDir(t)
	if _, ok := framework.GetPluginBuilder(artifactBridgeName); ok {
		t.Fatal("unique test registry occupied")
	}
	framework.RegisterPluginBuilder(artifactBridgeName, newArtifactBridge)
	builder, ok := framework.GetPluginBuilder(artifactBridgeName)
	if !ok {
		t.Fatal("registry")
	}
	first := map[string][]byte{}
	var offScore *probeScoringReceipt
	var offAction *probeActionReceipt
	var oldRecords []*adviceRecord
	var oldAudits []nodeProbeAudit
	for repeat := 0; repeat < 3; repeat++ {
		for _, arm := range []string{"off", "zero", "nonzero", "stale"} {
			t.Run(fmt.Sprintf("%d_%s", repeat, arm), func(t *testing.T) {
				f, mock, _ := actionFixture(t, "B")
				f.session.ID = "bridge-session"
				d := filepath.Join(dir, fmt.Sprintf("repeat-%d", repeat), arm)
				if e := os.MkdirAll(d, 0700); e != nil {
					t.Fatal(e)
				}
				base := bridgeScores(t, f)
				if base["node0"] != 1019 || base["node1"] != 1010 {
					t.Fatal("B4 scale")
				}
				p := builder(nil).(*artifactBridgeProbe)
				hooks := probeHooks(t, f.session)
				before := probeDomain(t, f)
				if arm != "off" {
					p.OnSessionOpen(f.session)
					t.Cleanup(func() { p.OnSessionClose(f.session) })
					hooks["NodeOrderFns"]++
					if !reflect.DeepEqual(hooks, probeHooks(t, f.session)) || !bytes.Equal(before, probeDomain(t, f)) {
						t.Fatal("open side effect")
					}
				}
				var raw []byte
				var grant *bridgeGrant
				lambda := float64(40)
				if arm == "zero" || arm == "off" {
					lambda = 0
				}
				if arm != "off" {
					raw, grant = bridgeProduce(t, f, 1, filepath.Join(d, "handoff"))
					if arm == "stale" {
						f.session.ClusterInfo.Nodes["node0"].IdleVector[resource_info.CPUIndex] -= 1000
					}
					active := p.begin(raw, grant, lambda)
					if active != (arm != "stale") {
						t.Fatal("activation", arm, active)
					}
					if arm == "stale" {
						f.session.ClusterInfo.Nodes["node0"].IdleVector[resource_info.CPUIndex] += 1000
					}
				}
				want := map[string]float64{"node0": base["node0"], "node1": base["node1"]}
				order := []string{"node0", "node1"}
				if arm == "nonzero" {
					want["node0"] -= 10
					order = []string{"node1", "node0"}
				}
				scores := adviceScoring(t, f, want, order)
				var scoring nodeProbeAudit
				if p.record != nil {
					scoring = p.record.audit()
				}
				p.endPhase()
				// A grant is one-phase/one-use; Action needs a fresh validated producer.
				if arm == "zero" || arm == "nonzero" {
					raw, grant = bridgeProduce(t, f, 2, filepath.Join(d, "action-handoff"))
					if !p.begin(raw, grant, lambda) {
						t.Fatal("fresh Action activation")
					}
				}
				action := bridgeAction(t, f, mock, order[0])
				if arm == "off" && offScore == nil {
					offScore = &scores
					offAction = &action
				}
				if arm == "zero" || arm == "stale" {
					if !reflect.DeepEqual(*offScore, scores) || !reflect.DeepEqual(*offAction, action) {
						t.Fatal("OFF/neutral/stale native differential")
					}
				}
				var act nodeProbeAudit
				if p.record != nil {
					act = p.record.audit()
				}
				p.endPhase()
				if arm == "nonzero" || arm == "zero" {
					if act.Total != 2 || !act.Complete {
						t.Fatal("native Action did not actually use bridge", act)
					}
				}
				norm := nativeJSON(t, map[string]any{"scoring": scores, "action": action, "returned_scores": scoring.Entries, "action_scores": act.Entries, "score_calls": scoring.Total, "action_calls": act.Total})
				if old := first[arm]; old != nil && !bytes.Equal(old, norm) {
					t.Fatal("repeat mismatch")
				}
				first[arm] = norm
				nativeSave(t, d, "normalized.json", json.RawMessage(norm))
				nativeSave(t, d, "raw-audit.json", map[string]any{"scoring": scoring, "action": act, "inactive": p.inactive.Load()})
				if arm != "off" {
					p.OnSessionClose(f.session)
					seal := p.record.audit()
					inactive := p.inactive.Load()
					f = probeRefresh(t, f)
					nodes := []*node_info.NodeInfo{f.session.ClusterInfo.Nodes["node1"], f.session.ClusterInfo.Nodes["node0"]}
					if len(f.session.OrderedNodesByTask(nodes, f.task)) != 2 || !reflect.DeepEqual(seal, p.record.audit()) || p.inactive.Load() != inactive {
						t.Fatal("active post-close writes/drop")
					}
					nativeSave(t, d, "sealed.json", seal)
					for _, r := range oldRecords {
						if r == p.record {
							t.Fatal("shared bridge record")
						}
					}
					oldRecords = append(oldRecords, p.record)
					oldAudits = append(oldAudits, seal)
				}
				for i, r := range oldRecords {
					if !reflect.DeepEqual(r.audit(), oldAudits[i]) {
						t.Fatal("cross-Session sealed contamination")
					}
				}
			})
		}
	}
}

func TestBridgeStateEvolution(t *testing.T) {
	test_utils.InitTestingInfrastructure()
	dir := filepath.Join(bridgeDir(t), "evolution")
	f, mock, _ := actionFixture(t, "B")
	f.session.ID = "bridge-evolution"
	p := newArtifactBridge(nil).(*artifactBridgeProbe)
	p.OnSessionOpen(f.session)
	t.Cleanup(func() { p.OnSessionClose(f.session) })
	raw, g := bridgeProduce(t, f, 1, filepath.Join(dir, "first"))
	if !p.begin(raw, g, 40) {
		t.Fatal("first")
	}
	first := bridgeAction(t, f, mock, "node1")
	p.endPhase()
	// New genuine official Pod/PodInfo arrival, not forced placement/Statement.
	current := f.session.ClusterInfo.PodGroupInfos[f.job.UID]
	pod := tasks_fake.BuildPod("native-job-1", "", &tasks_fake.TestTaskBasic{State: pod_status.Pending}, v1.PodPending, f.task.Pod.Spec.Containers[0].Resources.Requests, "", 0, nil, "native-job")
	next := pod_info.NewTaskInfo(pod, f.task.VectorMap)
	current.AddTaskInfo(next)
	f = nativeFixture{f.session, current, next}
	if next.Status != pod_status.Pending || next.NodeName != "" {
		t.Fatal("arrival")
	}
	// Old bytes+old capability cannot reactivate, and active native scoring proves
	// neutral behavior before the second Action (not merely a no-work Action).
	if p.begin(raw, g, 40) {
		t.Fatal("old grant reused")
	}
	base := bridgeScores(t, f)
	if p.inactive.Load() == 0 {
		t.Fatal("no active stale scoring")
	}
	currentProjection, err := bridgeProjection(t, f)
	if err != nil || bridgeHash(currentProjection) == g.projectionHash {
		t.Fatal("state evolution not actually measured", err)
	}
	raw2, g2 := bridgeProduce(t, f, 2, filepath.Join(dir, "fresh"))
	if !p.begin(raw2, g2, 40) {
		t.Fatal("fresh state not activated")
	}
	want := map[string]float64{"node0": base["node0"] - 10, "node1": base["node1"]}
	order := []string{"node1", "node0"}
	if want["node0"] > want["node1"] {
		order = []string{"node0", "node1"}
	}
	actual := adviceScoring(t, f, want, order)
	p.endPhase()
	raw3, g3 := bridgeProduce(t, f, 3, filepath.Join(dir, "second-action"))
	if !p.begin(raw3, g3, 40) {
		t.Fatal("second Action activation")
	}
	second := bridgeAction(t, f, mock, order[0])
	p.endPhase()
	nativeSave(t, dir, "two-actions.json", map[string]any{"first": first, "stale_neutral_scores": base, "fresh_scoring": actual, "second": second, "old_grant_rejected": true})
}

func TestBridgeFaultsAndTrust(t *testing.T) {
	test_utils.InitTestingInfrastructure()
	dir := filepath.Join(bridgeDir(t), "faults")
	f, _, _ := actionFixture(t, "B")
	f.session.ID = "bridge-faults"
	p := newArtifactBridge(nil).(*artifactBridgeProbe)
	p.OnSessionOpen(f.session)
	t.Cleanup(func() { p.OnSessionClose(f.session) })
	raw, original := bridgeProduce(t, f, 1, filepath.Join(dir, "trusted"))
	copyGrant := func() *bridgeGrant { g := *original; g.used = false; return &g }
	if p.begin(raw, nil, 40) {
		t.Fatal("arbitrary JSON activated without capability")
	}
	var obj map[string]any
	_ = json.Unmarshal(raw, &obj)
	mutations := []func(map[string]any){
		func(a map[string]any) { a["version"] = "fake" }, func(a map[string]any) { a["producer"] = "CERTIFIED_OFFLINE_ADVICE" },
		func(a map[string]any) { a["artifact_sha256"] = "0000" }, func(a map[string]any) { a["task_uid"] = "foreign" },
		func(a map[string]any) { a["pairs"] = []any{} }, func(a map[string]any) { pairs := a["pairs"].([]any); a["pairs"] = append(pairs, pairs[0]) },
		func(a map[string]any) { a["pairs"].([]any)[0].(map[string]any)["score_bits"] = "7ff8000000000000" },
		func(a map[string]any) { a["pairs"].([]any)[0].(map[string]any)["score_bits"] = "4000000000000000" },
		func(a map[string]any) { a["nodes"].([]any)[0].(map[string]any)["name"] = "node1" },
		func(a map[string]any) { a["unknown"] = true },
	}
	for i, mutate := range mutations {
		var a map[string]any
		_ = json.Unmarshal(raw, &a)
		mutate(a)
		b, _ := json.Marshal(a)
		if p.begin(b, copyGrant(), 40) {
			t.Fatal("rehashed untrusted bytes accepted", i)
		}
		g := copyGrant()
		g.envelopeHash = bridgeHash(b)
		if _, err := decodeBridge(b, g); err == nil {
			t.Fatal("strict decoder accepted privileged fault", i)
		}
	}
	// Trusted decoder defense is tested separately; replacing the digest below is
	// a privileged fault injection, not an untrusted caller's available capability.
	for _, b := range [][]byte{append([]byte(" "), raw...), []byte(`{"version":"M3B-1","version":"M3B-1"}`), []byte(`{"phase":1.0}`), bytes.Repeat([]byte("x"), 65537)} {
		g := copyGrant()
		g.envelopeHash = bridgeHash(b)
		if _, e := decodeBridge(b, g); e == nil {
			t.Fatal("malformed decoder")
		}
	}
	for _, lambda := range []float64{-1, math.NaN(), math.Inf(1), 1e6 + 1} {
		if p.begin(raw, copyGrant(), lambda) {
			t.Fatal("invalid lambda")
		}
	}
	// Actual native state faults, each compared before and after restoration.
	for _, idx := range []int{resource_info.CPUIndex, resource_info.GPUIndex, resource_info.MemoryIndex} {
		for _, which := range []string{"idle", "used", "releasing"} {
			n := f.session.ClusterInfo.Nodes["node0"]
			v := n.IdleVector
			if which == "used" {
				v = n.UsedVector
			}
			if which == "releasing" {
				v = n.ReleasingVector
			}
			v[idx]++
			if p.begin(raw, copyGrant(), 40) {
				t.Fatal("stale resources", idx, which)
			}
			v[idx]--
		}
	}
	old := f.task.Status
	f.task.Status = pod_status.Allocated
	if p.begin(raw, copyGrant(), 40) {
		t.Fatal("stale status")
	}
	f.task.Status = old
	f.task.ResReqVector[resource_info.CPUIndex]++
	if p.begin(raw, copyGrant(), 40) {
		t.Fatal("stale demand")
	}
	f.task.ResReqVector[resource_info.CPUIndex]--
	n := f.session.ClusterInfo.Nodes["node0"]
	delete(f.session.ClusterInfo.Nodes, "node0")
	f.session.ClusterInfo.Nodes["foreign"] = n
	if p.begin(raw, copyGrant(), 40) {
		t.Fatal("renamed node")
	}
	delete(f.session.ClusterInfo.Nodes, "foreign")
	f.session.ClusterInfo.Nodes["node0"] = n
	nodeCopy := *n
	f.session.ClusterInfo.Nodes["node0"] = &nodeCopy
	if p.begin(raw, copyGrant(), 40) {
		t.Fatal("reused display name with different native identity")
	}
	f.session.ClusterInfo.Nodes["node0"] = n
	f.task.Pod.Spec.NodeSelector = map[string]string{"unsupported": "x"}
	if p.begin(raw, copyGrant(), 40) {
		t.Fatal("predicate semantics")
	}
	f.task.Pod.Spec.NodeSelector = nil
	priority := f.job.Priority
	f.job.Priority = 1
	if p.begin(raw, copyGrant(), 40) {
		t.Fatal("job state changed")
	}
	f.job.Priority = priority
	n.PodInfos[f.task.UID] = f.task
	if p.begin(raw, copyGrant(), 40) {
		t.Fatal("pod membership")
	}
	delete(n.PodInfos, f.task.UID)
	n.NumaTopology = &node_info.NumaTopology{}
	if p.begin(raw, copyGrant(), 40) {
		t.Fatal("NUMA")
	}
	n.NumaTopology = nil
	oldLabel := n.Node.Labels[commonconstants.GpuCountLabel]
	n.Node.Labels[commonconstants.GpuCountLabel] = "1"
	if p.begin(raw, copyGrant(), 40) {
		t.Fatal("family labels")
	}
	n.Node.Labels[commonconstants.GpuCountLabel] = oldLabel
	for _, set := range f.job.PodSets {
		set.SetMinAvailable(2)
		if p.begin(raw, copyGrant(), 40) {
			t.Fatal("Gang")
		}
		set.SetMinAvailable(1)
	}
	originalPredicates := f.session.PredicateFns
	f.session.PredicateFns = append(append(originalPredicates[:0:0], originalPredicates...), originalPredicates[0])
	if p.begin(raw, copyGrant(), 40) {
		t.Fatal("predicate hook mutation")
	}
	f.session.PredicateFns = originalPredicates
	other, _, _ := actionFixture(t, "B")
	other.session.ID = f.session.ID
	q := newArtifactBridge(nil).(*artifactBridgeProbe)
	q.OnSessionOpen(other.session)
	t.Cleanup(func() { q.OnSessionClose(other.session) })
	if q.begin(raw, copyGrant(), 40) {
		t.Fatal("foreign Session same label")
	}
	if !p.begin(raw, copyGrant(), 0) {
		t.Fatal("valid neutral grant rejected")
	}
	scores := bridgeScores(t, f)
	if scores["node0"] != 1019 || scores["node1"] != 1010 {
		t.Fatal("diagnostics lost native candidates")
	}
	p.endPhase()
	nativeSave(t, dir, "results.json", map[string]any{"tampered_envelopes": len(mutations), "arbitrary_json_inactive": true, "resource_faults": 9, "same_label_foreign_session_rejected": true, "native_zero_scores": scores, "test_failed": t.Failed()})
}
