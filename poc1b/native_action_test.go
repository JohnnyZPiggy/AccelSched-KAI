package main

// B4 executes only the native in-process Allocate Action over official fake
// Sessions. Zero GPUs and absent NUMA topology do not validate device scheduling.
// Frozen go.mod omits the already-selected yaml.v2 requirement needed by the
// official test_utils package. Reproduce with a workspace-only copy of go.mod
// and go.sum: go list -mod=mod -modfile=<copy.mod> -deps -test ./...;
// then go test/vet/build -mod=readonly -modfile=<copy.mod> ./... .
// The original files stay frozen; compare go list -m all for identical versions.
// Ordinary -mod=readonly currently fails before compilation; this is a handoff
// constraint, not permission to update the frozen dependency files.
import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"

	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/actions/allocate"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/common_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/node_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/pod_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/pod_status"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/resource_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/cache"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/framework"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/test_utils"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/test_utils/jobs_fake"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/test_utils/nodes_fake"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/test_utils/tasks_fake"
	"go.uber.org/mock/gomock"
	"k8s.io/client-go/kubernetes/fake"
)

type actionReceipt struct {
	Before          json.RawMessage    `json:"before"`
	After           json.RawMessage    `json:"after"`
	Feasible        map[string]bool    `json:"feasible"`
	PredicateErrors map[string]string  `json:"predicate_errors"`
	Scores          map[string]float64 `json:"native_scores"`
	Choice          string             `json:"choice"`
	Binds           []any              `json:"fake_bind_calls"`
	Config          json.RawMessage    `json:"resolved_configuration"`
	ActionReturn    string             `json:"action_return"`
}

func actionFixture(t *testing.T, name string) (nativeFixture, *cache.MockCache, *gomock.Controller) {
	t.Helper()
	ctrl := gomock.NewController(t)
	small := float64(1)
	if name == "B" {
		small = 4
	}
	metadata := test_utils.TestTopologyBasic{
		Name:   "B4-" + name,
		Jobs:   []*jobs_fake.TestJobBasic{{Name: "native-job", QueueName: "queue0", RequiredCPUsPerTask: 2, RequiredMemoryPerTask: nativeGiB, Tasks: []*tasks_fake.TestTaskBasic{{State: pod_status.Pending}}}},
		Nodes:  map[string]nodes_fake.TestNodeBasic{"node0": {CPUMillis: small, CPUMemory: 8 * nativeGiB}, "node1": {CPUMillis: 8, CPUMemory: 8 * nativeGiB}},
		Queues: []test_utils.TestQueueBasic{{Name: "queue0"}},
		// Zero bypasses the helper's MaxTimes Bind expectation: install Times(1)
		// separately and record actual arguments, count and nil return below.
		Mocks: &test_utils.TestMock{CacheRequirements: &test_utils.CacheMocking{}},
	}
	s := test_utils.BuildSession(metadata, ctrl)
	s.ID = "native-action-" + name
	mock, ok := s.Cache.(*cache.MockCache)
	if !ok {
		t.Fatal("not official cache mock")
	}
	if _, ok := s.Cache.KubeClient().(*fake.Clientset); !ok {
		t.Fatal("not in-memory fake clientset")
	}
	j := s.ClusterInfo.PodGroupInfos[common_info.NewPodGroupID("", "native-job")]
	if j == nil {
		t.Fatal("missing job")
	}
	p := j.GetAllPodsMap()[common_info.PodID("native-job-0")]
	if p == nil {
		t.Fatal("missing task")
	}
	f := nativeFixture{s, j, p}
	if p.ResReqVector.Get(resource_info.CPUIndex) != 2000 || p.ResReqVector.Get(resource_info.MemoryIndex) != nativeGiB || p.ResReqVector.Get(resource_info.GPUIndex) != 0 || s.ClusterInfo.Nodes["node0"].AllocatableVector.Get(resource_info.CPUIndex) != small*1000 || s.ClusterInfo.Nodes["node1"].AllocatableVector.Get(resource_info.CPUIndex) != 8000 {
		t.Fatal("official fake resource units changed")
	}
	// CloseSession is teardown only, never OpenSession/RunApp. It releases the
	// scoring pool. Fake helper informers use Background().Done() (nil): no public
	// cancel handle exists; bounded test-process exit releases them, not Shutdown
	// (which would wait forever). Do not claim per-session informer cancellation.
	mock.EXPECT().RecordJobStatusEvent(gomock.Any(), gomock.Any()).Return(nil).Times(1)
	mock.EXPECT().WaitForWorkers(gomock.Any()).Times(1)
	t.Cleanup(func() { framework.CloseSession(s); ctrl.Finish() })
	return f, mock, ctrl
}

func actionRun(t *testing.T, f nativeFixture, mock *cache.MockCache, name string) actionReceipt {
	t.Helper()
	r := actionReceipt{Before: nativeState(t, f), Feasible: map[string]bool{}, PredicateErrors: map[string]string{}, Scores: map[string]float64{}, Binds: []any{}, Config: nativeJSON(t, f.session.Config), ActionReturn: "Execute returns void; success asserted from state and exactly one nil-return mock Bind"}
	s := f.session
	if err := s.PrePredicateFn(f.task, f.job); err != nil {
		t.Fatal(err)
	}
	fitting := []*node_info.NodeInfo{}
	for _, n := range []string{"node0", "node1"} {
		node := s.ClusterInfo.Nodes[n]
		err := s.PredicateFn(f.task, f.job, node)
		r.Feasible[n] = s.FittingNode(f.task, node, false)
		if err != nil {
			r.PredicateErrors[n] = err.Error()
		}
		if r.Feasible[n] {
			fitting = append(fitting, node)
		}
	}
	if name == "A" && (r.Feasible["node0"] || !r.Feasible["node1"]) {
		t.Fatalf("A not unique feasible: %v; state=%s", r.Feasible, r.Before)
	}
	if name == "B" && (!r.Feasible["node0"] || !r.Feasible["node1"]) {
		t.Fatalf("B not both feasible: %v; state=%s", r.Feasible, r.Before)
	}
	s.NodePreOrderFn(f.task, fitting)
	for _, node := range fitting {
		score, err := s.NodeOrderFn(f.task, node)
		if err != nil {
			t.Fatal(err)
		}
		r.Scores[node.Name] = score
	}
	if name == "B" && !(r.Scores["node0"] > r.Scores["node1"]) {
		t.Fatalf("B lacks strict native score separation: %+v", r.Scores)
	}
	mock.EXPECT().Bind(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).DoAndReturn(func(p *pod_info.PodInfo, node string, a map[string]string, n cache.NUMAPrediction) error {
		r.Binds = append(r.Binds, map[string]any{"task": nativeTaskState(p), "node": node, "annotations": a, "numa": n, "returned_error": nil})
		return nil
	}).Times(1)
	// No task node is assigned by this test. The official action selects it.
	allocate.New().Execute(s)
	r.After = nativeState(t, f)
	r.Choice = f.task.NodeName
	want := "node1"
	if name == "B" {
		want = "node0"
	}
	if r.Choice != want || len(r.Binds) != 1 || f.task.Status != pod_status.Allocated {
		t.Fatalf("native action choice=%s want=%s calls=%d status=%s; expected native Allocated after nil fake Bind", r.Choice, want, len(r.Binds), f.task.Status)
	}
	return r
}

func actionOffline(t *testing.T, dir, cli string, f nativeFixture, trace []byte) {
	t.Helper()
	nodes := []any{}
	eligible := []string{}
	for _, name := range []string{"node0", "node1"} {
		n := f.session.ClusterInfo.Nodes[name]
		nodes = append(nodes, map[string]any{"id": name, "gpu_family": "CPU_ONLY_TEST", "shape": "test-cpu", "capacity": []int{0, int(n.AllocatableVector.Get(resource_info.CPUIndex) / 1000), int(n.AllocatableVector.Get(resource_info.MemoryIndex) / nativeGiB)}})
		if n.AllocatableVector.Get(resource_info.CPUIndex) >= 2000 {
			eligible = append(eligible, name)
		}
	}
	demand := []int{0, int(f.task.ResReqVector.Get(resource_info.CPUIndex) / 1000), int(f.task.ResReqVector.Get(resource_info.MemoryIndex) / nativeGiB)}
	snapshot := map[string]any{"schema_version": "0.1", "units": []string{"whole GPUs", "integer CPU cores", "integer RAM GiB"}, "nodes": nodes, "tasks": []any{map[string]any{"id": string(f.task.UID), "demand": demand, "value": 1, "eligible": eligible, "gpu_family": nil}}, "future": []any{map[string]any{"id": "test-only-future", "demand": demand, "weight": 1, "eligible": eligible, "gpu_family": nil}}, "realized_future": []any{}, "constraints": map[string]any{}}
	nativeSave(t, dir, "test-only-snapshot.json", snapshot)
	if err := os.WriteFile(filepath.Join(dir, "native-action-trace.json"), trace, 0600); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(cli, "-trace", filepath.Join(dir, "native-action-trace.json"), "-snapshot", filepath.Join(dir, "test-only-snapshot.json"), "-out", filepath.Join(dir, "offline-bundle"))
	t.Logf("frozen offline CLI: %q", cmd.Args)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("offline CLI: %v %s", err, out)
	}
	data, err := os.ReadFile(filepath.Join(dir, "offline-bundle", "trace-result.json"))
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
	if result.Mode != "SAVED_UNVERIFIED" || result.Authority != "UNVERIFIED" || result.Status != "TRACE_INCOMPLETE" || result.Commit || result.Bind || len(result.Statements) != 0 || len(result.Bindings) != 0 || len(result.Observations) == 0 {
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
	b, err := os.ReadFile(filepath.Join(dir, "offline-bundle", "snapshot.json"))
	if err != nil || !bytes.Equal(a, b) {
		t.Fatal("snapshot not preserved", err)
	}
}

func TestNativeAllocateActionDifferential(t *testing.T) {
	if runtime.Version() != "go1.26.8" {
		t.Fatal("exact Go 1.26.8 required")
	}
	test_utils.InitTestingInfrastructure()
	dir := t.TempDir()
	if root := os.Getenv("POC1B_ACTION_EVIDENCE_DIR"); root != "" {
		dir = root
		if err := os.MkdirAll(dir, 0700); err != nil {
			t.Fatal(err)
		}
	}
	suffix := ""
	if runtime.GOOS == "windows" {
		suffix = ".exe"
	}
	cli := filepath.Join(dir, "frozen-poc1-offline"+suffix)
	build := exec.Command(filepath.Join(runtime.GOROOT(), "bin", "go"+suffix), "build", "-o", cli, ".")
	build.Dir = "../poc1"
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("offline build: %v %s", err, out)
	}
	plugin := newShadowPlugin(nil).(*shadowPlugin)
	var sealed []*capture
	var seals [][]byte
	for _, name := range []string{"A", "B"} {
		var first *actionReceipt
		for repeat := 0; repeat < 3; repeat++ {
			t.Run(fmt.Sprintf("%s_repeat_%d_G1_G6", name, repeat), func(t *testing.T) {
				armDir := filepath.Join(dir, fmt.Sprintf("%s-%d", name, repeat))
				if err := os.MkdirAll(armDir, 0700); err != nil {
					t.Fatal(err)
				}
				off, offCache, _ := actionFixture(t, name)
				on, onCache, _ := actionFixture(t, name)
				if off.session == on.session || off.session.Cache == on.session.Cache || off.job == on.job || off.task == on.task || off.task.Pod == on.task.Pod || off.task.VectorMap == on.task.VectorMap || off.session.ClusterInfo == on.session.ClusterInfo {
					t.Fatal("shared mutable fixture")
				}
				for _, node := range []string{"node0", "node1"} {
					a, b := off.session.ClusterInfo.Nodes[node], on.session.ClusterInfo.Nodes[node]
					if a == b || &a.IdleVector[0] == &b.IdleVector[0] {
						t.Fatal("shared node/vector")
					}
				}
				if plugin.sessionCapture(off.session) != nil {
					t.Fatal("OFF observer attached")
				}
				baseline := actionRun(t, off, offCache, name)
				plugin.OnSessionOpen(on.session)
				t.Cleanup(func() { plugin.OnSessionClose(on.session) })
				c := plugin.sessionCapture(on.session)
				if c == nil {
					t.Fatal("observer did not attach")
				}
				observed := actionRun(t, on, onCache, name)
				nativeSave(t, armDir, "off.json", baseline)
				nativeSave(t, armDir, "on.json", observed)
				if !reflect.DeepEqual(baseline, observed) {
					t.Fatal("native Action OFF/ON difference")
				}
				if first == nil {
					first = &baseline
				} else if !reflect.DeepEqual(*first, baseline) {
					t.Fatal("bounded repeat not deterministic")
				}
				plugin.OnSessionClose(on.session)
				if plugin.sessionCapture(on.session) != nil {
					t.Fatal("closed capture retained")
				}
				audit, err := c.Snapshot()
				if err != nil || !audit.Complete || len(audit.Events) == 0 {
					t.Fatal("native callback missing/incomplete", err)
				}
				for _, e := range audit.Events {
					if e.Kind != "ALLOCATE_OBSERVED" || e.Task != string(on.task.UID) || e.Node != observed.Choice {
						t.Fatal("unexpected native event", e)
					}
				}
				nativeSave(t, armDir, "capture-audit.json", audit)
				trace, err := c.MarshalTrace()
				if err != nil {
					t.Fatal(err)
				}
				actionOffline(t, armDir, cli, on, trace)
				// Re-execute the native Action after close (no new Pending work), and check
				// all earlier sessions remain sealed as subsequent independent Actions run.
				allocate.New().Execute(on.session)
				after, err := c.MarshalTrace()
				if err != nil || !bytes.Equal(trace, after) {
					t.Fatal("post-close mutation", err)
				}
				sealed = append(sealed, c)
				seals = append(seals, trace)
				for i, old := range sealed {
					b, err := old.MarshalTrace()
					if err != nil || !bytes.Equal(b, seals[i]) {
						t.Fatal("session leakage", err)
					}
				}
				t.Logf("G1-G6 %s repetition=%d choice=%s scores=%v feasible=%v exact fake Bind calls OFF=1 ON=1; no real binding assertion", name, repeat, observed.Choice, observed.Scores, observed.Feasible)
			})
		}
	}
	nativeSave(t, dir, "scope.json", map[string]any{"claim": "ACTION_LEVEL_DECISION_INVARIANT (tested fixtures only)", "full_scheduler": "NOT_VERIFIED", "gpu_numa": "zero GPU; no NUMA topology", "snapshot": "TEST_ONLY; native resource capacities/requests and IDs; value/future/shape/family are explicit synthetic metadata; pod capacity nonbinding, not lossless adapter", "normalization": "B3b public Task/Job/Node resource projection, timestamps/pointers/private caches excluded; accepted fields retained; actual resolved configuration and native scores compared", "cleanup": "CloseSession releases scoring pool and exact mock status/worker expectations; official helper's Background informers have no public cancellation handle, remain bounded to 12 Sessions until test-process exit; no real Kubernetes or persistent workers", "rollback": "B3b cold AcceptedResourceVector/ResourceReceivedType latch remains unchanged; no rollback restoration claim", "test_failed": t.Failed()})
}
