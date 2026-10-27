package main

// B3b reuses the public fixture pattern in pinned KAI framework/statement_test.go.
// No private Statement fields, fake handler calls, native OpenSession/CloseSession,
// Commit, RunApp/Run, clients, or scheduler process. All GPU/NUMA fields are CPU-
// fixture bookkeeping (zero GPUs / no NUMA topology), not device validation.
import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"testing"

	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/common_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/pod_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/pod_status"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/podgroup_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/resource_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/framework"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/test_utils/jobs_fake"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/test_utils/nodes_fake"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/test_utils/tasks_fake"
)

const nativeGiB = 1024 * 1024 * 1024

type nativeFixture struct {
	session *framework.Session
	job     *podgroup_info.PodGroupInfo
	task    *pod_info.PodInfo
}

func nativeFixtureNew(t *testing.T) nativeFixture {
	t.Helper()
	vm := resource_info.NewResourceVectorMap()
	jobs, toNodes, _ := jobs_fake.BuildJobsAndTasksMaps([]*jobs_fake.TestJobBasic{{
		Name: "native-job", QueueName: "queue0", RequiredGPUsPerTask: 0,
		RequiredCPUsPerTask: 2, RequiredMemoryPerTask: nativeGiB,
		Tasks: []*tasks_fake.TestTaskBasic{{State: pod_status.Pending}},
	}}, vm)
	// Despite its CPUMillis name this upstream fake passes a plain CPU quantity;
	// input8 is eight cores. Verify actual vectors instead of assuming units.
	nodes := nodes_fake.BuildNodesInfoMap(map[string]nodes_fake.TestNodeBasic{
		"node0": {CPUMillis: 8, CPUMemory: 8 * nativeGiB},
		"node1": {CPUMillis: 8, CPUMemory: 8 * nativeGiB},
	}, toNodes, nil, vm)
	job := jobs[common_info.NewPodGroupID("", "native-job")]
	if job == nil {
		t.Fatal("official job builder did not expose expected job")
	}
	task := job.GetAllPodsMap()[common_info.PodID("native-job-0")]
	if task == nil {
		t.Fatal("official task builder did not expose expected task")
	}
	s := &framework.Session{ID: "native-statement-session", ClusterInfo: &api.ClusterInfo{PodGroupInfos: jobs, Nodes: nodes}}
	if task.ResReqVector.Get(resource_info.CPUIndex) != 2000 || task.ResReqVector.Get(resource_info.MemoryIndex) != nativeGiB || task.ResReqVector.Get(resource_info.GPUIndex) != 0 || nodes["node0"].AllocatableVector.Get(resource_info.CPUIndex) != 8000 || nodes["node0"].AllocatableVector.Get(resource_info.MemoryIndex) != 8*nativeGiB {
		t.Fatal("unexpected official fixture resource units")
	}
	return nativeFixture{s, job, task}
}

func nativeJSON(t *testing.T, v any) json.RawMessage {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func nativeGPU(g *resource_info.GpuResourceRequirement) map[string]any {
	// The requirement's storage is private: JSON-marshalling it directly would
	// silently produce {}. Use its public accessors, never reflect private state.
	return map[string]any{"gpus": g.GPUs(), "memory": g.GpuMemory(), "devices": g.GetNumOfGpuDevices(), "portion": g.GpuFractionalPortion(), "mig": g.MigResources(), "dra": g.DraGpuCounts()}
}

func nativeTaskState(p *pod_info.PodInfo) map[string]any {
	return map[string]any{
		"uid": string(p.UID), "name": p.Name, "namespace": p.Namespace, "job": string(p.Job), "status": p.Status.String(), "node": p.NodeName,
		"virtual": p.IsVirtualStatus, "request_type": p.ResourceRequestType, "received_type": p.ResourceReceivedType,
		"request": p.ResReqVector.Clone(), "accepted": p.AcceptedResourceVector.Clone(),
		"gpu": nativeGPU(&p.GpuRequirement), "accepted_gpu": nativeGPU(&p.AcceptedGpuRequirement),
		"gpu_groups": p.GPUGroupIDs(), "fractional_groups": p.FractionalGpuGroups, "numa": p.NUMAPlacement.Clone(),
		"pod_uid": string(p.Pod.UID), "pod_node": p.Pod.Spec.NodeName,
	}
}

func nativeState(t *testing.T, f nativeFixture) json.RawMessage {
	t.Helper()
	status := map[string][]string{}
	for state, tasks := range f.job.PodStatusIndex {
		ids := []string{}
		for id := range tasks {
			ids = append(ids, string(id))
		}
		sort.Strings(ids)
		status[state.String()] = ids
	}
	tasks := map[string]any{}
	for id, p := range f.job.GetAllPodsMap() {
		tasks[string(id)] = nativeTaskState(p)
	}
	nodes := map[string]any{}
	for name, n := range f.session.ClusterInfo.Nodes {
		members := map[string]any{}
		for id, p := range n.PodInfos {
			members[string(id)] = nativeTaskState(p)
		}
		nodes[name] = map[string]any{"name": n.Name, "members": members, "allocatable": n.AllocatableVector.Clone(), "idle": n.IdleVector.Clone(), "used": n.UsedVector.Clone(), "releasing": n.ReleasingVector.Clone(), "gpu_sharing": n.GpuSharingNodeInfo, "legacy_mig": n.LegacyMIGTasks, "numa_topology": n.NumaTopology, "node_resource_topology": n.NodeResourceTopology}
	}
	labels := []string{}
	for i := 0; i < f.task.VectorMap.Len(); i++ {
		labels = append(labels, string(f.task.VectorMap.ResourceAt(i)))
	}
	// Exclude only fixture wall-clock timestamps, object addresses and private
	// lazy caches. All observed task fields, including accepted bookkeeping, stay.
	return nativeJSON(t, map[string]any{"vector_labels": labels, "tasks": tasks, "job": map[string]any{"uid": string(f.job.UID), "allocated": f.job.AllocatedVector.Clone(), "statuses": status}, "nodes": nodes})
}

type nativeRoundtrip struct {
	Before        json.RawMessage `json:"before"`
	During        json.RawMessage `json:"during"`
	After         json.RawMessage `json:"after"`
	Checkpoint    int             `json:"checkpoint"`
	AllocateError *string         `json:"allocate_error"`
	RollbackError *string         `json:"rollback_error"`
	RawRestored   bool            `json:"raw_restored"`
}

func nativeError(err error) *string {
	if err == nil {
		return nil
	}
	s := err.Error()
	return &s
}

func nativeExercise(t *testing.T, f nativeFixture) nativeRoundtrip {
	t.Helper()
	before := nativeState(t, f)
	statement := f.session.Statement()
	cp := statement.Checkpoint()
	alloc := statement.Allocate(f.task, "node0")
	during := nativeState(t, f)
	rollback := statement.Rollback(cp)
	after := nativeState(t, f)
	return nativeRoundtrip{before, during, after, int(cp), nativeError(alloc), nativeError(rollback), bytes.Equal(before, after)}
}

func nativeSuccess(t *testing.T, r nativeRoundtrip) {
	t.Helper()
	if r.AllocateError != nil || r.RollbackError != nil {
		t.Fatalf("native errors: %s", nativeJSON(t, r))
	}
}

func nativeSave(t *testing.T, dir, name string, v any) {
	t.Helper()
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(dir, name), append(b, '\n'), 0600); err != nil {
		t.Fatal(err)
	}
}

func TestNativeSessionDifferential(t *testing.T) {
	if runtime.Version() != "go1.26.8" {
		t.Fatal("exact go1.26.8 required")
	}
	dir := t.TempDir()
	if root := os.Getenv("POC1B_NATIVE_EVIDENCE_DIR"); root != "" {
		dir = root
		if err := os.MkdirAll(dir, 0700); err != nil {
			t.Fatal(err)
		}
	}
	proof := map[string]any{"claim": "STATEMENT_STATE_INVARIANT (tested cold and prepared-Pending OFF/ON paths); exact rollback restoration only on prepared-Pending", "FULL_SCHEDULER_DECISION_INVARIANT": "NOT_VERIFIED: node0 is specified, no action/scoring decision tested", "normalization": "documented public task/job/node/resource projection, not entire Go object memory; no wall-clock timestamps, pointer identity or private lazy caches compared; accepted bookkeeping explicitly included", "vector_units": "CPU millicores, memory bytes, GPU count, pod count", "fixture_imports": "official jobs_fake/tasks_fake/nodes_fake; no local substitutes", "precondition": "Primary exact-restoration pair: Pending task after a prior native Allocate/Rollback; additional fresh cold OFF/ON pair below has no preparation"}
	defer func() { proof["test_failed"] = t.Failed(); nativeSave(t, dir, "differential-proof.json", proof) }()
	off, on := nativeFixtureNew(t), nativeFixtureNew(t)
	if off.session == on.session || off.session.ClusterInfo == on.session.ClusterInfo || off.task == on.task || off.task.Pod == on.task.Pod || off.job == on.job || off.task.VectorMap == on.task.VectorMap || off.session.ClusterInfo.Nodes["node0"] == on.session.ClusterInfo.Nodes["node0"] || off.session.ClusterInfo.Nodes["node1"] == on.session.ClusterInfo.Nodes["node1"] || &off.task.ResReqVector[0] == &on.task.ResReqVector[0] || &off.job.AllocatedVector[0] == &on.job.AllocatedVector[0] || &off.session.ClusterInfo.Nodes["node0"].IdleVector[0] == &on.session.ClusterInfo.Nodes["node0"].IdleVector[0] {
		t.Fatal("shared fixture pointers")
	}
	initialOff, initialOn := nativeState(t, off), nativeState(t, on)
	if !bytes.Equal(initialOff, initialOn) {
		t.Fatal("fresh fixtures are not equivalent")
	}
	// Native cold allocation latches AcceptedResourceVector/ReceivedType in
	// upstream NodeInfo. Preserve the first roundtrip rather than hiding those
	// fields or patching state. Establish identical valid prepared-Pending inputs
	// through native public methods only, before installing either observer.
	warmOff := nativeExercise(t, off)
	proof["cold_preparation_off"] = warmOff
	nativeSuccess(t, warmOff)
	if !bytes.Equal(nativeState(t, on), initialOn) {
		t.Fatal("OFF operation changed independent ON fixture")
	}
	warmOn := nativeExercise(t, on)
	proof["cold_preparation_on"] = warmOn
	nativeSuccess(t, warmOn)
	if !reflect.DeepEqual(warmOff, warmOn) {
		t.Fatal("cold baseline preparations differ")
	}
	proof["cold_raw_restore"] = warmOff.RawRestored
	if !bytes.Equal(nativeState(t, off), nativeState(t, on)) {
		t.Fatal("prepared fixtures differ")
	}
	baseline := nativeExercise(t, off)
	proof["off"] = baseline
	nativeSuccess(t, baseline)
	if !baseline.RawRestored {
		t.Fatal("prepared OFF baseline does not restore original state")
	}
	plugin := newShadowPlugin(nil).(*shadowPlugin)
	plugin.OnSessionOpen(on.session)
	defer plugin.OnSessionClose(on.session)
	c := plugin.sessionCapture(on.session)
	if c == nil {
		t.Fatal("missing installed native capture")
	}
	if !bytes.Equal(nativeState(t, on), baseline.Before) {
		t.Fatal("observer open mutated native state")
	}
	// OFF native events while ON's observer exists must not enter its buffer.
	offIsolated := nativeExercise(t, off)
	proof["off_while_on_open"] = offIsolated
	nativeSuccess(t, offIsolated)
	observed := nativeExercise(t, on)
	proof["on"] = observed
	nativeSuccess(t, observed)
	plugin.OnSessionClose(on.session)
	audit, err := c.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	nativeSave(t, dir, "native-capture-audit.json", audit)
	trace, err := c.MarshalTrace()
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(dir, "native-generated-trace.json"), trace, 0600); err != nil {
		t.Fatal(err)
	}
	t.Run("G1_native_fixture", func(t *testing.T) {
		if off.session.Cache != nil || on.session.Cache != nil || off.task.ResReqVector.Get(resource_info.GPUIndex) != 0 {
			t.Fatal("not a CPU-only no-client fixture")
		}
		if baseline.Checkpoint != 0 || observed.Checkpoint != 0 {
			t.Fatal("unexpected public checkpoint")
		}
	})
	t.Run("G2_native_callbacks", func(t *testing.T) {
		if !audit.Complete || audit.Received != 2 || len(audit.Events) != 2 {
			t.Fatal("native callbacks missing, duplicated or cross-session leaked", audit)
		}
		for i, kind := range []string{"ALLOCATE_OBSERVED", "DEALLOCATE_OBSERVED"} {
			e := audit.Events[i]
			if e.Seq != uint64(i+1) || e.Kind != kind || e.Task != string(on.task.UID) || e.Node != "node0" {
				t.Fatal(e)
			}
		}
		// This file never invokes a handler: only Statement.Allocate/Rollback did.
	})
	t.Run("G3_source_integrity", func(t *testing.T) {
		for _, stage := range []struct{ a, b json.RawMessage }{{baseline.Before, observed.Before}, {baseline.During, observed.During}, {baseline.After, observed.After}} {
			if !bytes.Equal(stage.a, stage.b) {
				t.Fatal("OFF/ON native state differs")
			}
		}
		if bytes.Equal(observed.Before, observed.During) {
			t.Fatal("vacuous comparison: Allocate did not change state")
		}
	})
	t.Run("G4_statement_state_invariant", func(t *testing.T) {
		if !reflect.DeepEqual(baseline, observed) || !observed.RawRestored {
			t.Fatal("prepared Statement roundtrip parity failed")
		}
		var during struct {
			Tasks map[string]struct {
				Status  string `json:"status"`
				Node    string `json:"node"`
				Virtual bool   `json:"virtual"`
			} `json:"tasks"`
		}
		if err := json.Unmarshal(observed.During, &during); err != nil {
			t.Fatal(err)
		}
		p := during.Tasks[string(on.task.UID)]
		if p.Status != "Allocated" || p.Node != "node0" || !p.Virtual {
			t.Fatal("not a native virtual allocation", p)
		}
		// Also compare a fresh COLD OFF/ON pair, so the accepted-cache finding is
		// not used to avoid testing initial native allocation with the observer.
		coldOff, coldOn := nativeFixtureNew(t), nativeFixtureNew(t)
		coldBaseline := nativeExercise(t, coldOff)
		nativeSuccess(t, coldBaseline)
		coldPlugin := newShadowPlugin(nil).(*shadowPlugin)
		coldPlugin.OnSessionOpen(coldOn.session)
		defer coldPlugin.OnSessionClose(coldOn.session)
		coldCapture := coldPlugin.sessionCapture(coldOn.session)
		coldObserved := nativeExercise(t, coldOn)
		nativeSuccess(t, coldObserved)
		coldPlugin.OnSessionClose(coldOn.session)
		proof["cold_observed_off"], proof["cold_observed_on"] = coldBaseline, coldObserved
		if !reflect.DeepEqual(coldBaseline, coldObserved) {
			t.Fatal("cold native OFF/ON source-state difference")
		}
		if coldBaseline.RawRestored || coldObserved.RawRestored {
			t.Fatal("expected pinned native cold accepted-bookkeeping latch changed")
		}
		coldAudit, err := coldCapture.Snapshot()
		if err != nil || !coldAudit.Complete || len(coldAudit.Events) != 2 {
			t.Fatal("cold native capture incomplete", err)
		}
		coldTrace, err := coldCapture.MarshalTrace()
		if err != nil || !bytes.Equal(coldTrace, trace) {
			t.Fatal("cold callback trace differs from the trace submitted to frozen CLI", err)
		}
		nativeSave(t, dir, "cold-native-capture-audit.json", coldAudit)
		if err = os.WriteFile(filepath.Join(dir, "cold-native-generated-trace.json"), coldTrace, 0600); err != nil {
			t.Fatal(err)
		}
		proof["cold_observed_trace_matches_prepared"] = true
		t.Log("STATEMENT_STATE_INVARIANT: cold/prepared OFF/ON parity; exact rollback restoration only prepared-Pending; FULL_SCHEDULER_DECISION_INVARIANT NOT VERIFIED")
	})
	t.Run("G5_lifecycle", func(t *testing.T) {
		if plugin.sessionCapture(on.session) != nil {
			t.Fatal("closed session retained")
		}
		postOff, postOn := nativeExercise(t, off), nativeExercise(t, on)
		nativeSuccess(t, postOff)
		nativeSuccess(t, postOn)
		proof["post_close_off"] = postOff
		proof["post_close_on"] = postOn
		after, err := c.Snapshot()
		if err != nil || !reflect.DeepEqual(audit, after) || !postOff.RawRestored || !postOn.RawRestored {
			t.Fatal("post-close native events mutated capture or source", err)
		}
	})
	t.Run("G6_offline_handoff", func(t *testing.T) {
		// Independently built TEST-ONLY snapshot from the in-memory CPU fixture,
		// not a real KAI snapshot adapter/forecast. Future/value/Shape/family are
		// explicit schema-required test choices; no real-world authenticity claim.
		nodes := []any{}
		for _, name := range []string{"node0", "node1"} {
			n := on.session.ClusterInfo.Nodes[name]
			nodes = append(nodes, map[string]any{"id": name, "gpu_family": "CPU_ONLY_TEST", "shape": "test-cpu", "capacity": []int{0, int(n.AllocatableVector.Get(resource_info.CPUIndex) / 1000), int(n.AllocatableVector.Get(resource_info.MemoryIndex) / nativeGiB)}})
		}
		demand := []int{0, int(on.task.ResReqVector.Get(resource_info.CPUIndex) / 1000), int(on.task.ResReqVector.Get(resource_info.MemoryIndex) / nativeGiB)}
		snapshot := map[string]any{"schema_version": "0.1", "units": []string{"whole GPUs", "integer CPU cores", "integer RAM GiB"}, "nodes": nodes, "tasks": []any{map[string]any{"id": string(on.task.UID), "demand": demand, "value": 1, "eligible": []string{"node0", "node1"}, "gpu_family": nil}}, "future": []any{map[string]any{"id": "test-only-future", "demand": demand, "weight": 1, "eligible": []string{"node0", "node1"}, "gpu_family": nil}}, "realized_future": []any{}, "constraints": map[string]any{}}
		nativeSave(t, dir, "test-only-snapshot.json", snapshot)
		suffix := ""
		if runtime.GOOS == "windows" {
			suffix = ".exe"
		}
		goTool := filepath.Join(runtime.GOROOT(), "bin", "go"+suffix)
		cli := filepath.Join(dir, "frozen-poc1-offline"+suffix)
		build := exec.Command(goTool, "build", "-o", cli, ".")
		build.Dir = "../poc1"
		t.Logf("offline CLI build: %q cwd=%s", build.Args, build.Dir)
		if out, err := build.CombinedOutput(); err != nil {
			t.Fatalf("frozen CLI build: %v: %s", err, out)
		}
		outdir := filepath.Join(dir, "offline-bundle")
		cmd := exec.Command(cli, "-trace", filepath.Join(dir, "native-generated-trace.json"), "-snapshot", filepath.Join(dir, "test-only-snapshot.json"), "-out", outdir)
		cmd.Dir = dir
		t.Logf("offline CLI run: %q", cmd.Args)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("frozen offline CLI: %v: %s", err, out)
		}
		result, err := os.ReadFile(filepath.Join(outdir, "trace-result.json"))
		if err != nil {
			t.Fatal(err)
		}
		var v struct {
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
		if err = json.Unmarshal(result, &v); err != nil {
			t.Fatal(err)
		}
		if v.Mode != "SAVED_UNVERIFIED" || v.Authority != "UNVERIFIED" || v.Status != "TRACE_INCOMPLETE" || v.Commit || v.Bind || len(v.Statements) != 0 || len(v.Bindings) != 0 || len(v.Observations) != 2 {
			t.Fatalf("unsafe interpretation: %s", result)
		}
		for _, o := range v.Observations {
			if o.State != "VIRTUAL_OBSERVED" || o.Association != "UNKNOWN" {
				t.Fatal(o)
			}
		}
		a, err := os.ReadFile(filepath.Join(dir, "test-only-snapshot.json"))
		if err != nil {
			t.Fatal(err)
		}
		b, err := os.ReadFile(filepath.Join(outdir, "snapshot.json"))
		if err != nil || !bytes.Equal(a, b) {
			t.Fatal("offline CLI changed test snapshot", err)
		}
		proof["offline_cli_exit"] = 0
		proof["snapshot_relation"] = "TEST_ONLY 0GPU/8cores/8GiB nodes and 0GPU/2cores/1GiB task; UID/node IDs match official native builders; future/value/shape/family are declared test metadata; no real forecast/eligibility integration claim"
	})
}
