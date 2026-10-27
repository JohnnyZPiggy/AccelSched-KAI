package main

// B5a reuses the exact pinned single-victim Train/Build topology. GPU resources
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
	"time"

	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/actions/preempt"
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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

type preemptFixture struct {
	session *framework.Session
	mock    *cache.MockCache
	jobs    map[string]nativeFixture
}

type preemptReceipt struct {
	Before       json.RawMessage   `json:"before"`
	After        json.RawMessage   `json:"after"`
	Config       json.RawMessage   `json:"resolved_configuration"`
	Victims      []string          `json:"state_victim_uids"`
	Evicts       []json.RawMessage `json:"actual_fake_evict_calls"`
	Pipelines    []json.RawMessage `json:"actual_fake_pipeline_calls"`
	Binds        []json.RawMessage `json:"actual_fake_bind_calls"`
	ActionReturn string            `json:"action_return"`
}

func preemptFixtureNew(t *testing.T) preemptFixture {
	t.Helper()
	ctrl := gomock.NewController(t)
	metadata := test_utils.TestTopologyBasic{
		Name: "1 train job running, 1 build job pending for the same queue - preempt",
		Jobs: []*jobs_fake.TestJobBasic{
			{Name: "running_job0", RequiredGPUsPerTask: 1, Priority: constants.PriorityTrainNumber, QueueName: "queue0", Tasks: []*tasks_fake.TestTaskBasic{{NodeName: "node0", State: pod_status.Running}}},
			{Name: "pending_job0", RequiredGPUsPerTask: 1, Priority: constants.PriorityBuildNumber, QueueName: "queue0", Tasks: []*tasks_fake.TestTaskBasic{{State: pod_status.Pending}}},
		},
		Nodes:  map[string]nodes_fake.TestNodeBasic{"node0": {GPUs: 1}},
		Queues: []test_utils.TestQueueBasic{{Name: "queue0", DeservedGPUs: 1}},
		// Disable helper MaxTimes for these operations, assert exact calls ourselves.
		Mocks: &test_utils.TestMock{CacheRequirements: &test_utils.CacheMocking{}},
	}
	s := test_utils.BuildSession(metadata, ctrl)
	s.ID = "native-preempt-single-victim"
	// Public configuration only: bound scenario search, do not alter callbacks.
	budget := metav1.Duration{Duration: 2 * time.Second}
	s.Config.ScenarioSearchBudgets.MaxActionSearchDuration["preempt"] = metav1.Duration{Duration: 5 * time.Second}
	s.Config.ScenarioSearchBudgets.MaxJobSearchDuration = &budget
	for key := range s.Config.ScenarioSearchBudgets.MaxGeneratorSearchDuration {
		s.Config.ScenarioSearchBudgets.MaxGeneratorSearchDuration[key] = budget
	}
	mock, ok := s.Cache.(*cache.MockCache)
	if !ok {
		t.Fatal("not official fake Cache")
	}
	if _, ok := s.Cache.KubeClient().(*fake.Clientset); !ok {
		t.Fatal("not in-memory fake clientset")
	}
	f := preemptFixture{s, mock, map[string]nativeFixture{}}
	for _, name := range []string{"running_job0", "pending_job0"} {
		job := s.ClusterInfo.PodGroupInfos[common_info.NewPodGroupID("", name)]
		if job == nil {
			t.Fatal("missing official job", name)
		}
		task := job.GetAllPodsMap()[common_info.PodID(name+"-0")]
		if task == nil {
			t.Fatal("missing official task", name)
		}
		f.jobs[name] = nativeFixture{s, job, task}
		if task.ResReqVector.Get(resource_info.GPUIndex) != 1 {
			t.Fatal("not 1 simulated GPU request")
		}
	}
	low, high := f.jobs["running_job0"], f.jobs["pending_job0"]
	if low.job.Priority >= high.job.Priority || low.job.Queue != high.job.Queue || low.task.Status != pod_status.Running || high.task.Status != pod_status.Pending || low.task.NodeName != "node0" || high.task.NodeName != "" || len(s.ClusterInfo.Nodes) != 1 || s.ClusterInfo.Nodes["node0"].AllocatableVector.Get(resource_info.GPUIndex) != 1 {
		t.Fatal("official oracle preconditions changed")
	}
	mock.EXPECT().RecordJobStatusEvent(gomock.Any(), gomock.Any()).Return(nil).Times(2)
	mock.EXPECT().WaitForWorkers(gomock.Any()).Times(1)
	// Public teardown releases scoring pools. The official helper's background
	// informers have no cancel handle: bounded process lifetime, no unsafe Shutdown.
	t.Cleanup(func() { framework.CloseSession(s); ctrl.Finish() })
	return f
}

func preemptState(t *testing.T, f preemptFixture) json.RawMessage {
	t.Helper()
	jobs := map[string]any{}
	for name, j := range f.jobs {
		jobs[name] = map[string]any{"priority": j.job.Priority, "queue": j.job.Queue, "projection": nativeState(t, j)}
	}
	return nativeJSON(t, jobs)
}

func preemptRun(t *testing.T, f preemptFixture) preemptReceipt {
	t.Helper()
	r := preemptReceipt{Before: preemptState(t, f), Config: nativeJSON(t, f.session.Config), Victims: []string{}, Evicts: []json.RawMessage{}, Pipelines: []json.RawMessage{}, Binds: []json.RawMessage{}, ActionReturn: "Execute returns void; exact mock nil Evict/void Pipeline and source states asserted; no real operations certified"}
	f.mock.EXPECT().Evict(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).DoAndReturn(func(p *v1.Pod, j *podgroup_info.PodGroupInfo, m eviction_info.EvictionMetadata, message string) error {
		r.Evicts = append(r.Evicts, nativeJSON(t, map[string]any{"pod_uid": string(p.UID), "pod_name": p.Name, "namespace": p.Namespace, "pod_node": p.Spec.NodeName, "job_uid": string(j.UID), "metadata": m, "message": message, "returned_error": nil}))
		return nil
	}).Times(1)
	f.mock.EXPECT().TaskPipelined(gomock.Any(), gomock.Any()).Do(func(p *pod_info.PodInfo, message string) {
		r.Pipelines = append(r.Pipelines, nativeJSON(t, map[string]any{"task": nativeTaskState(p), "message": message, "return": "void"}))
	}).Times(1)
	f.mock.EXPECT().Bind(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Times(0)
	preempt.New().Execute(f.session)
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
	r.After = preemptState(t, f)
	for _, j := range f.jobs {
		if j.task.Status == pod_status.Releasing {
			r.Victims = append(r.Victims, string(j.task.UID))
		}
	}
	sort.Strings(r.Victims)
	low, high := f.jobs["running_job0"].task, f.jobs["pending_job0"].task
	if !reflect.DeepEqual(r.Victims, []string{"running_job0-0"}) || low.NodeName != "node0" || high.Status != pod_status.Pipelined || high.NodeName != "node0" || len(r.Evicts) != 1 || len(r.Pipelines) != 1 {
		t.Fatalf("native preempt failed oracle: victims=%v low=%s/%s high=%s/%s evicts=%d pipelines=%d", r.Victims, low.Status, low.NodeName, high.Status, high.NodeName, len(r.Evicts), len(r.Pipelines))
	}
	var e struct {
		UID string `json:"pod_uid"`
	}
	if err := json.Unmarshal(r.Evicts[0], &e); err != nil || e.UID != r.Victims[0] {
		t.Fatal("state/actual Evict victim mismatch", err)
	}
	return r
}

func preemptOffline(t *testing.T, dir string, f preemptFixture, trace []byte, count int) {
	t.Helper()
	n := f.session.ClusterInfo.Nodes["node0"]
	tasks := []any{}
	for _, name := range []string{"running_job0", "pending_job0"} {
		p := f.jobs[name].task
		if p.ResReqVector.Get(resource_info.CPUIndex) != 1000 || p.ResReqVector.Get(resource_info.MemoryIndex) != 1000000000 {
			t.Fatal("snapshot relation needs review: official request changed")
		}
		tasks = append(tasks, map[string]any{"id": string(p.UID), "demand": []int{1, 1, 1}, "value": 1, "eligible": []string{"node0"}, "gpu_family": nil})
	}
	// TEST_ONLY conservative integer projection: upstream default 20G decimal RAM
	// is floor(20e9/GiB)=18GiB; task 1G becomes ceil(1e9/GiB)=1GiB.
	// CPU is exact: node 20000 cores, task 1 core (upstream fake defaults).
	// This is conservative test metadata, not an exact memory adapter.
	snapshot := map[string]any{"schema_version": "0.1", "units": []string{"whole GPUs", "integer CPU cores", "integer RAM GiB"}, "nodes": []any{map[string]any{"id": "node0", "gpu_family": "SIMULATED_GPU_TEST", "shape": "TEST_ONLY", "capacity": []int{1, int(n.AllocatableVector.Get(resource_info.CPUIndex) / 1000), int(n.AllocatableVector.Get(resource_info.MemoryIndex) / nativeGiB)}}}, "tasks": tasks, "future": []any{map[string]any{"id": "test-only-future", "demand": []int{1, 1, 1}, "weight": 1, "eligible": []string{"node0"}, "gpu_family": nil}}, "realized_future": []any{}, "constraints": map[string]any{}}
	nativeSave(t, dir, "test-only-snapshot.json", snapshot)
	if err := os.WriteFile(filepath.Join(dir, "native-preempt-trace.json"), trace, 0600); err != nil {
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
	cmd := exec.Command(cli, "-trace", filepath.Join(dir, "native-preempt-trace.json"), "-snapshot", filepath.Join(dir, "test-only-snapshot.json"), "-out", filepath.Join(dir, "offline-bundle"))
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

func TestNativePreemptActionDifferential(t *testing.T) {
	if runtime.Version() != "go1.26.8" {
		t.Fatal("exact Go 1.26.8 required")
	}
	test_utils.InitTestingInfrastructure()
	dir := t.TempDir()
	if root := os.Getenv("POC1B_PREEMPT_EVIDENCE_DIR"); root != "" {
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
			off, on := preemptFixtureNew(t), preemptFixtureNew(t)
			if off.session == on.session || off.session.Cache == on.session.Cache || off.session.ClusterInfo == on.session.ClusterInfo || off.session.ClusterInfo.Nodes["node0"] == on.session.ClusterInfo.Nodes["node0"] {
				t.Fatal("shared Session/Cache/Node")
			}
			for name, a := range off.jobs {
				b := on.jobs[name]
				if a.job == b.job || a.task == b.task || a.task.Pod == b.task.Pod || a.task.VectorMap == b.task.VectorMap || &a.task.ResReqVector[0] == &b.task.ResReqVector[0] || &a.job.AllocatedVector[0] == &b.job.AllocatedVector[0] {
					t.Fatal("shared job/task/resource pointers")
				}
			}
			a, b := off.session.ClusterInfo.Nodes["node0"], on.session.ClusterInfo.Nodes["node0"]
			if &a.IdleVector[0] == &b.IdleVector[0] {
				t.Fatal("shared node resources")
			}
			if plugin.sessionCapture(off.session) != nil {
				t.Fatal("OFF attached")
			}
			baseline := preemptRun(t, off)
			plugin.OnSessionOpen(on.session)
			t.Cleanup(func() { plugin.OnSessionClose(on.session) })
			c := plugin.sessionCapture(on.session)
			if c == nil {
				t.Fatal("ON did not attach")
			}
			observed := preemptRun(t, on)
			nativeSave(t, d, "off.json", baseline)
			nativeSave(t, d, "on.json", observed)
			if !reflect.DeepEqual(baseline, observed) {
				t.Fatal("OFF/ON Preempt Action difference")
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
			if err != nil || !audit.Complete {
				t.Fatal("capture incomplete", err)
			}
			nativeSave(t, d, "capture-audit.json", audit)
			if len(audit.Events) == 0 {
				t.Log("observer blind spot: no genuine supported callback; normal trace export/offline CLI NOT_RUN")
			} else {
				for _, e := range audit.Events {
					if (e.Kind != "ALLOCATE_OBSERVED" && e.Kind != "DEALLOCATE_OBSERVED") || (e.Task != "running_job0-0" && e.Task != "pending_job0-0") || e.Node != "node0" {
						t.Fatal("unexpected native callback", e)
					}
				}
				trace, err := c.MarshalTrace()
				if err != nil {
					t.Fatal(err)
				}
				preemptOffline(t, d, on, trace, len(audit.Events))
			}
			// No direct callbacks or Statements: a post-close Action sees no Pending job.
			preempt.New().Execute(on.session)
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
			t.Logf("G1-G6 repeat=%d victims=%v preemptor=Pipelined/node0 Evict=1 Pipeline=1 Bind=0 per arm; genuine supported callback count=%d, not eviction/pipeline certification", repeat, observed.Victims, len(audit.Events))
		})
	}
	nativeSave(t, dir, "scope.json", map[string]any{"claim": "PREEMPT_ACTION_DECISION_INVARIANT (tested fixture only)", "test_failed": t.Failed(), "trace_limit": "only actual native AllocateFunc/DeallocateFunc callbacks; no Evict/Pipeline schema coverage or real Commit/Bind confirmation", "normalization": "reuse B3b full documented public projection per job plus priority/queue; excludes timestamps/pointer identities/private caches only; no accepted resource masking", "simulated_gpu": "one simulated GPU/node and one GPU/task; official defaults: task 1 core/1e9 bytes, node 20000 cores/20e9 bytes; CPU execution only, no physical GPU or NUMA proof", "budget": "public config preempt action 5s, job 2s, generators 2s; Go test timeout separately enforced", "snapshot": "TEST_ONLY integer projection, decimal node RAM conservatively floored and task RAM ceiled to GiB; synthetic future/value/family/shape, no lossless adapter claim", "cleanup": "scoring pools released via public CloseSession; six fake Sessions' background informers end at bounded test-process exit, no per-session cancellation claimed", "rollback": "B3b cold accepted-bookkeeping latch preserved; no cold-restoration claim", "post_close": "repeat Action has no new Pending job; native post-close callback rejection covered by frozen B3a/B3b; cross-session sealing checked here"})
}
