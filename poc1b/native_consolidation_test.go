package main

// B5c reuses the pinned first successful Consolidation topology and frozen B5a
// fixture/receipt/state helpers. Standalone CLI glue is needed because the frozen
// prior offline helpers fix two tasks/one node, unlike this fixture. GPU resources
// are simulated in memory; no device, API server or Scheduler process is used.
// Use the B4 workspace-only normalized modfile; frozen dependencies stay intact.
import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"testing"

	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/actions/consolidation"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/common_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/eviction_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/pod_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/pod_status"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/podgroup_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/resource_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/cache"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/constants"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/framework"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/test_utils"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/test_utils/jobs_fake"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/test_utils/nodes_fake"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/test_utils/tasks_fake"
	"go.uber.org/mock/gomock"
	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes/fake"
)

func consolidationFixtureNew(t *testing.T) preemptFixture {
	t.Helper()
	ctrl := gomock.NewController(t)
	metadata := test_utils.TestTopologyBasic{
		Name: "2 jobs of 1 task running on different GPUs, two pending jobs, only first can be consolidated",
		Jobs: []*jobs_fake.TestJobBasic{
			{Name: "running_job0", RequiredGPUsPerTask: 2, Priority: constants.PriorityTrainNumber, QueueName: "queue0", Tasks: []*tasks_fake.TestTaskBasic{{NodeName: "node0", State: pod_status.Running}}},
			{Name: "running_job1", RequiredGPUsPerTask: 2, Priority: constants.PriorityTrainNumber, QueueName: "queue0", Tasks: []*tasks_fake.TestTaskBasic{{NodeName: "node1", State: pod_status.Running}}},
			{Name: "pending_job0", RequiredGPUsPerTask: 3, Priority: constants.PriorityTrainNumber, QueueName: "queue0", Tasks: []*tasks_fake.TestTaskBasic{{State: pod_status.Pending}}},
			{Name: "pending_job1", RequiredGPUsPerTask: 2, Priority: constants.PriorityTrainNumber, QueueName: "queue0", Tasks: []*tasks_fake.TestTaskBasic{{State: pod_status.Pending}}},
		},
		Nodes:  map[string]nodes_fake.TestNodeBasic{"node0": {GPUs: 4}, "node1": {GPUs: 4}},
		Queues: []test_utils.TestQueueBasic{{Name: "queue0", DeservedGPUs: 2}},
		Mocks:  &test_utils.TestMock{CacheRequirements: &test_utils.CacheMocking{}},
	}
	s := test_utils.BuildSession(metadata, ctrl)
	s.ID = "native-consolidation-first-success"
	// Preserve the public official helper's actual parameter and default budgets.
	// -1 selects the cached/unlimited route; it is not a strict victim count cap.
	if s.GetMaxNumberConsolidationPreemptees() != -1 {
		t.Fatal("official fake consolidation parameter changed", s.GetMaxNumberConsolidationPreemptees())
	}
	if s.Config.ScenarioSearchBudgets == nil || s.Config.ScenarioSearchBudgets.MaxJobSearchDuration == nil || s.Config.ScenarioSearchBudgets.MaxJobSearchDuration.Duration <= 0 {
		t.Fatal("missing finite official search budgets")
	}
	for _, budget := range s.Config.ScenarioSearchBudgets.MaxActionSearchDuration {
		if budget.Duration <= 0 {
			t.Fatal("invalid official action budget")
		}
	}
	for _, budget := range s.Config.ScenarioSearchBudgets.MaxGeneratorSearchDuration {
		if budget.Duration <= 0 {
			t.Fatal("invalid official generator budget")
		}
	}
	mock, ok := s.Cache.(*cache.MockCache)
	if !ok {
		t.Fatal("not official fake Cache")
	}
	if _, ok := s.Cache.KubeClient().(*fake.Clientset); !ok {
		t.Fatal("not in-memory fake clientset")
	}
	f := preemptFixture{s, mock, map[string]nativeFixture{}}
	for _, name := range []string{"running_job0", "running_job1", "pending_job0", "pending_job1"} {
		job := s.ClusterInfo.PodGroupInfos[common_info.NewPodGroupID("", name)]
		if job == nil {
			t.Fatal("missing official job", name)
		}
		task := job.GetAllPodsMap()[common_info.PodID(name+"-0")]
		if task == nil {
			t.Fatal("missing official task", name)
		}
		f.jobs[name] = nativeFixture{s, job, task}
		wantGPU := float64(2)
		if name == "pending_job0" {
			wantGPU = 3
		}
		if task.ResReqVector.Get(resource_info.GPUIndex) != wantGPU {
			t.Fatal("official simulated GPU demand changed")
		}
	}
	for name, j := range f.jobs {
		wantStatus, wantNode := pod_status.Pending, ""
		if name == "running_job0" {
			wantStatus, wantNode = pod_status.Running, "node0"
		}
		if name == "running_job1" {
			wantStatus, wantNode = pod_status.Running, "node1"
		}
		if j.job.Priority != constants.PriorityTrainNumber || string(j.job.Queue) != "queue0" || j.task.Status != wantStatus || j.task.NodeName != wantNode {
			t.Fatal("official initial task oracle changed", name)
		}
	}
	if len(s.ClusterInfo.Nodes) != 2 || len(s.ClusterInfo.PodGroupInfos) != 4 || s.ClusterInfo.Queues["queue0"].Resources.GPU.Quota != 2 {
		t.Fatal("official topology changed")
	}
	for _, n := range s.ClusterInfo.Nodes {
		if n.AllocatableVector.Get(resource_info.GPUIndex) != 4 {
			t.Fatal("not two 4-GPU nodes")
		}
	}
	mock.EXPECT().RecordJobStatusEvent(gomock.Any(), gomock.Any()).Return(nil).Times(4)
	mock.EXPECT().WaitForWorkers(gomock.Any()).Times(1)
	// Public teardown releases scoring pools. The official helper's background
	// informers have no cancel handle: bounded process lifetime, no unsafe Shutdown.
	t.Cleanup(func() { framework.CloseSession(s); ctrl.Finish() })
	return f
}

func consolidationState(t *testing.T, f preemptFixture) json.RawMessage {
	return nativeJSON(t, map[string]any{"public_state": reclaimState(t, f), "max_consolidation_preemptees": f.session.GetMaxNumberConsolidationPreemptees()})
}

func consolidationRun(t *testing.T, f preemptFixture) preemptReceipt {
	t.Helper()
	beforePositions := map[string]struct {
		Status pod_status.PodStatus
		Node   string
	}{}
	for name, j := range f.jobs {
		beforePositions[name] = struct {
			Status pod_status.PodStatus
			Node   string
		}{j.task.Status, j.task.NodeName}
	}
	r := preemptReceipt{Before: consolidationState(t, f), Config: nativeJSON(t, f.session.Config), Victims: []string{}, Evicts: []json.RawMessage{}, Pipelines: []json.RawMessage{}, Binds: []json.RawMessage{}, ActionReturn: "Execute returns void; exact mock nil Evict/void Pipeline and source states asserted; no real operations certified"}
	f.mock.EXPECT().Evict(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).DoAndReturn(func(p *v1.Pod, j *podgroup_info.PodGroupInfo, m eviction_info.EvictionMetadata, message string) error {
		r.Evicts = append(r.Evicts, nativeJSON(t, map[string]any{"pod_uid": string(p.UID), "pod_name": p.Name, "namespace": p.Namespace, "pod_node": p.Spec.NodeName, "job_uid": string(j.UID), "metadata": m, "message": message, "returned_error": nil}))
		return nil
	}).Times(1)
	f.mock.EXPECT().TaskPipelined(gomock.Any(), gomock.Any()).Do(func(p *pod_info.PodInfo, message string) {
		r.Pipelines = append(r.Pipelines, nativeJSON(t, map[string]any{"task": nativeTaskState(p), "message": message, "return": "void"}))
	}).Times(2)
	f.mock.EXPECT().Bind(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Times(0)
	consolidation.New().Execute(f.session)
	// Native scenario search publishes cloned jobs/tasks into the Session. Read
	// authoritative post-Action indexes, not the pre-search fixture pointers.
	for name := range f.jobs {
		job := f.session.ClusterInfo.PodGroupInfos[common_info.NewPodGroupID("", name)]
		if job == nil {
			t.Fatal("native action lost job", name)
		}
		task := job.GetAllPodsMap()[common_info.PodID(name+"-0")]
		if task == nil {
			t.Fatal("native action lost task", name)
		}
		f.jobs[name] = nativeFixture{f.session, job, task}
	}
	r.After = consolidationState(t, f)
	for name, j := range f.jobs {
		expectedStatus, expectedNode := pod_status.Pending, ""
		switch name {
		case "running_job0":
			expectedStatus, expectedNode = pod_status.Running, "node0"
		case "running_job1":
			expectedStatus, expectedNode = pod_status.Pipelined, "node0"
		case "pending_job0":
			expectedStatus, expectedNode = pod_status.Pipelined, "node1"
		}
		if j.task.Status != expectedStatus || j.task.NodeName != expectedNode || j.task.Status == pod_status.Releasing {
			t.Fatalf("official four-task/allPodsReallocated final oracle failed: %s %s/%s", name, j.task.Status, j.task.NodeName)
		}
		if beforePositions[name].Status == pod_status.Running && j.task.Status == pod_status.Pipelined && beforePositions[name].Node != j.task.NodeName {
			r.Victims = append(r.Victims, string(j.task.UID))
		}
	}
	sort.Strings(r.Victims)
	if !reflect.DeepEqual(r.Victims, []string{"running_job1-0"}) || len(r.Evicts) != 1 || len(r.Pipelines) != 2 || len(r.Binds) != 0 {
		t.Fatal("native consolidation cache counts/victim failed")
	}
	var e struct {
		UID  string `json:"pod_uid"`
		Node string `json:"pod_node"`
	}
	if err := json.Unmarshal(r.Evicts[0], &e); err != nil || e.UID != "running_job1-0" || e.Node != "node1" {
		t.Fatal("victim Evict source mismatch", err)
	}
	placements := map[string]string{}
	for _, raw := range r.Pipelines {
		var p struct {
			Task struct {
				UID  string `json:"uid"`
				Node string `json:"node"`
			} `json:"task"`
		}
		if err := json.Unmarshal(raw, &p); err != nil {
			t.Fatal(err)
		}
		placements[p.Task.UID] = p.Task.Node
	}
	if !reflect.DeepEqual(placements, map[string]string{"running_job1-0": "node0", "pending_job0-0": "node1"}) {
		t.Fatal("pipeline victim/new-job destinations differ", placements)
	}
	return r
}

func consolidationOffline(t *testing.T, dir string, f preemptFixture, trace []byte, count int) {
	t.Helper()
	nodes := []any{}
	for _, name := range []string{"node0", "node1"} {
		n := f.session.ClusterInfo.Nodes[name]
		nodes = append(nodes, map[string]any{"id": name, "gpu_family": "SIMULATED_GPU_TEST", "shape": "TEST_ONLY", "capacity": []int{int(n.AllocatableVector.Get(resource_info.GPUIndex)), int(n.AllocatableVector.Get(resource_info.CPUIndex) / 1000), int(n.AllocatableVector.Get(resource_info.MemoryIndex) / nativeGiB)}})
	}
	tasks := []any{}
	for _, name := range []string{"running_job0", "running_job1", "pending_job0", "pending_job1"} {
		p := f.jobs[name].task
		if p.ResReqVector.Get(resource_info.CPUIndex) != 1000 || p.ResReqVector.Get(resource_info.MemoryIndex) != 1000000000 {
			t.Fatal("snapshot relation needs review: official request changed")
		}
		tasks = append(tasks, map[string]any{"id": string(p.UID), "demand": []int{int(p.ResReqVector.Get(resource_info.GPUIndex)), 1, 1}, "value": 1, "eligible": []string{"node0", "node1"}, "gpu_family": nil})
	}
	// TEST_ONLY conservative integer projection: upstream default 20G decimal RAM
	// is floor(20e9/GiB)=18GiB; task 1G becomes ceil(1e9/GiB)=1GiB.
	// CPU is exact: node 20000 cores, task 1 core (upstream fake defaults).
	// This is conservative test metadata, not an exact memory adapter.
	snapshot := map[string]any{"schema_version": "0.1", "units": []string{"whole GPUs", "integer CPU cores", "integer RAM GiB"}, "nodes": nodes, "tasks": tasks, "future": []any{map[string]any{"id": "test-only-future", "demand": []int{1, 1, 1}, "weight": 1, "eligible": []string{"node0", "node1"}, "gpu_family": nil}}, "realized_future": []any{}, "constraints": map[string]any{}}
	nativeSave(t, dir, "test-only-snapshot.json", snapshot)
	if err := os.WriteFile(filepath.Join(dir, "native-consolidation-trace.json"), trace, 0600); err != nil {
		t.Fatal(err)
	}
	suffix := ""
	if runtime.GOOS == "windows" {
		suffix = ".exe"
	}
	cli := filepath.Join(dir, "frozen-poc1-offline"+suffix)
	build := exec.Command(filepath.Join(runtime.GOROOT(), "bin", "go"+suffix), "build", "-o", cli, ".")
	build.Dir = "../poc1"
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("frozen CLI build: %v %s", err, out)
	}
	cmd := exec.Command(cli, "-trace", filepath.Join(dir, "native-consolidation-trace.json"), "-snapshot", filepath.Join(dir, "test-only-snapshot.json"), "-out", filepath.Join(dir, "offline-bundle"))
	t.Logf("frozen offline CLI %q", cmd.Args)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("frozen CLI: %v %s", err, out)
	}
	data, err := os.ReadFile(filepath.Join(dir, "offline-bundle/trace-result.json"))
	if err != nil {
		t.Fatal(err)
	}
	var result struct {
		Mode         string         `json:"mode"`
		Authority    string         `json:"authority"`
		Status       string         `json:"status"`
		Commit       bool           `json:"real_commit_confirmed"`
		Bind         bool           `json:"real_bind_confirmed"`
		Statements   map[string]any `json:"statements"`
		Bindings     map[string]any `json:"bindings"`
		Observations []struct {
			State       string `json:"state"`
			Association string `json:"association"`
		} `json:"observations"`
	}
	if err = json.Unmarshal(data, &result); err != nil {
		t.Fatal(err)
	}
	if result.Mode != "SAVED_UNVERIFIED" || result.Authority != "UNVERIFIED" || result.Status != "TRACE_INCOMPLETE" || result.Commit || result.Bind || len(result.Statements) != 0 || len(result.Bindings) != 0 || len(result.Observations) != count {
		t.Fatalf("unsafe trace interpretation: %s", data)
	}
	for _, o := range result.Observations {
		if o.State != "VIRTUAL_OBSERVED" || o.Association != "UNKNOWN" {
			t.Fatal(o)
		}
	}
	a, err := os.ReadFile(filepath.Join(dir, "test-only-snapshot.json"))
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join(dir, "offline-bundle/snapshot.json"))
	if err != nil || !bytes.Equal(a, b) {
		t.Fatal("snapshot changed", err)
	}
}

func TestNativeConsolidationActionDifferential(t *testing.T) {
	if runtime.Version() != "go1.26.8" {
		t.Fatal("exact Go 1.26.8 required")
	}
	test_utils.InitTestingInfrastructure()
	dir := t.TempDir()
	if root := os.Getenv("POC1B_CONSOLIDATION_EVIDENCE_DIR"); root != "" {
		dir = root
		if err := os.MkdirAll(dir, 0700); err != nil {
			t.Fatal(err)
		}
	}
	plugin := newShadowPlugin(nil).(*shadowPlugin)
	var first *preemptReceipt
	var sealed []*capture
	var seals []captureSnapshot
	for repeat := 0; repeat < 3; repeat++ {
		t.Run(fmt.Sprintf("repeat_%d_G1_G6", repeat), func(t *testing.T) {
			d := filepath.Join(dir, fmt.Sprintf("pair-%d", repeat))
			if err := os.MkdirAll(d, 0700); err != nil {
				t.Fatal(err)
			}
			off, on := consolidationFixtureNew(t), consolidationFixtureNew(t)
			if off.session == on.session || off.session.Cache == on.session.Cache || off.session.ClusterInfo == on.session.ClusterInfo || off.session.ClusterInfo.Nodes["node0"] == on.session.ClusterInfo.Nodes["node0"] {
				t.Fatal("shared Session/Cache/Node")
			}
			for name, a := range off.jobs {
				b := on.jobs[name]
				if a.job == b.job || a.task == b.task || a.task.Pod == b.task.Pod || a.task.VectorMap == b.task.VectorMap || &a.task.ResReqVector[0] == &b.task.ResReqVector[0] || &a.job.AllocatedVector[0] == &b.job.AllocatedVector[0] {
					t.Fatal("shared job/task/resource pointers")
				}
			}
			for _, name := range []string{"node0", "node1"} {
				a, b := off.session.ClusterInfo.Nodes[name], on.session.ClusterInfo.Nodes[name]
				if a == b || &a.IdleVector[0] == &b.IdleVector[0] {
					t.Fatal("shared node resources")
				}
			}
			if plugin.sessionCapture(off.session) != nil {
				t.Fatal("OFF attached")
			}
			nativeSave(t, d, "raw-queue-child-order.json", map[string]any{"off": reclaimRawChildOrder(off), "on": reclaimRawChildOrder(on)})
			baseline := consolidationRun(t, off)
			plugin.OnSessionOpen(on.session)
			t.Cleanup(func() { plugin.OnSessionClose(on.session) })
			c := plugin.sessionCapture(on.session)
			if c == nil {
				t.Fatal("ON did not attach")
			}
			observed := consolidationRun(t, on)
			nativeSave(t, d, "off.json", baseline)
			nativeSave(t, d, "on.json", observed)
			if !reflect.DeepEqual(baseline, observed) {
				t.Fatal("OFF/ON Reclaim Action difference")
			}
			if first == nil {
				first = &baseline
			} else if !reflect.DeepEqual(*first, baseline) {
				t.Fatal("bounded repetition difference")
			}
			plugin.OnSessionClose(on.session)
			if plugin.sessionCapture(on.session) != nil {
				t.Fatal("closed capture retained")
			}
			audit, err := c.Snapshot()
			if err != nil || !audit.Closed || !audit.Complete || audit.SessionError || audit.Invalid != 0 || audit.Dropped != 0 || audit.Contended != 0 || audit.LifecycleErrors != 0 {
				t.Fatal("capture incomplete", err)
			}
			nativeSave(t, d, "capture-audit.json", audit)
			if len(audit.Events) == 0 {
				t.Log("observer blind spot: no genuine supported callback; normal trace export/offline CLI NOT_RUN")
				t.Fatal("official single-victim fixture requires exactly three genuine native virtual callbacks")
			} else {
				// Compare the exact multiset: no unproven callback order assumption.
				got := map[[3]string]int{}
				for i, e := range audit.Events {
					if e.Seq != uint64(i+1) {
						t.Fatal("noncontiguous native callback sequence", e.Seq)
					}
					got[[3]string{e.Kind, e.Task, e.Node}]++
				}
				want := map[[3]string]int{
					{"DEALLOCATE_OBSERVED", "running_job1-0", "node1"}: 1,
					{"ALLOCATE_OBSERVED", "running_job1-0", "node0"}:   1,
					{"ALLOCATE_OBSERVED", "pending_job0-0", "node1"}:   1,
				}
				if len(audit.Events) != 3 || audit.Received != 3 || audit.NextSeq != 3 || !reflect.DeepEqual(got, want) {
					t.Fatalf("expected exactly three native callbacks with the fixture-specific multiset; audit=%+v", audit)
				}
				trace, err := c.MarshalTrace()
				if err != nil {
					t.Fatal(err)
				}
				consolidationOffline(t, d, on, trace, len(audit.Events))
			}
			// Re-execute the native Action with the residual pending job; the official
			// fixture has no successful second consolidation. This checks sealing,
			// not proof of active post-close callback rejection.
			consolidation.New().Execute(on.session)
			after, err := c.Snapshot()
			if err != nil || !reflect.DeepEqual(audit, after) {
				t.Fatal("post-close mutation", err)
			}
			sealed = append(sealed, c)
			seals = append(seals, audit)
			for i, old := range sealed {
				after, err := old.Snapshot()
				if err != nil || !reflect.DeepEqual(after, seals[i]) {
					t.Fatal("cross-session contamination", err)
				}
			}
			t.Logf("G1-G6 repeat=%d victims=%v victim=Pipelined/node0 pending_job0=Pipelined/node1 Evict=1 Pipeline=2 Bind=0 per arm; genuine supported callback count=%d, not eviction/pipeline certification", repeat, observed.Victims, len(audit.Events))
		})
	}
	nativeSave(t, dir, "scope.json", map[string]any{"claim": "CONSOLIDATION_ACTION_DECISION_INVARIANT (tested fixture only)", "test_failed": t.Failed(), "trace_limit": "only actual native AllocateFunc/DeallocateFunc callbacks; no Evict/Pipeline schema coverage or real Commit/Bind confirmation", "normalization": "reuse B3b full documented public projection per job plus priority/queue; excludes timestamps/pointer identities/private caches only; no accepted resource masking; queue child membership set sorted in detached evidence only, raw order audit retained; private queue resource storage projected through public getters", "simulated_gpu": "two nodes with four simulated GPUs each, task demands 2/2/3/2; official defaults: task 1 core/1e9 bytes, node 20000 cores/20e9 bytes; CPU execution only, no physical GPU or NUMA proof", "budget": "official default action/job/generator budgets unchanged; Go test timeout90s separately enforced", "snapshot": "TEST_ONLY integer projection, decimal node RAM conservatively floored and task RAM ceiled to GiB; synthetic future/value/family/shape, no lossless adapter claim", "cleanup": "scoring pools released via public CloseSession; six fake Sessions' background informers end at bounded test-process exit, no per-session cancellation claimed", "rollback": "B3b cold accepted-bookkeeping latch preserved; no cold-restoration claim", "post_close": "repeat Action retains pending_job1 but no second success in this fixture; no active post-close callback rejection claim; cross-session sealing checked here"})
}
