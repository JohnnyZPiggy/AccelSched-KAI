package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"

	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/actions/allocate"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/common_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/node_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/pod_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/pod_status"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/cache"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/framework"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/test_utils"
	"go.uber.org/mock/gomock"
)

func probeRefresh(t *testing.T, f nativeFixture) nativeFixture {
	t.Helper()
	f.job = f.session.ClusterInfo.PodGroupInfos[common_info.NewPodGroupID("", "native-job")]
	if f.job == nil {
		t.Fatal("missing authoritative job")
	}
	f.task = f.job.GetAllPodsMap()[common_info.PodID("native-job-0")]
	if f.task == nil {
		t.Fatal("missing authoritative task")
	}
	return f
}
func probeDomain(t *testing.T, f nativeFixture) json.RawMessage {
	f = probeRefresh(t, f)
	return reclaimState(t, preemptFixture{session: f.session, jobs: map[string]nativeFixture{"native-job": f}})
}
func probeHooks(t *testing.T, s *framework.Session) map[string]int {
	t.Helper()
	out := map[string]int{}
	v := reflect.ValueOf(s).Elem()
	typ := v.Type()
	for i := 0; i < v.NumField(); i++ {
		field := typ.Field(i)
		value := v.Field(i)
		if field.PkgPath == "" && value.Kind() == reflect.Slice && value.Type().Elem().Kind() == reflect.Func {
			out[field.Name] = value.Len()
		}
	}
	// Read-only length of a pinned private slice (no values/unsafe/mutation).
	// KAI exposes AddEventHandler but no public handler-count diagnostic.
	handlers := v.FieldByName("eventHandlers")
	if !handlers.IsValid() || handlers.Kind() != reflect.Slice {
		t.Fatal("pinned handler metadata changed")
	}
	out["eventHandlers_count_only"] = handlers.Len()
	return out
}
func probeCounts(a nodeProbeAudit) map[string]uint64 {
	out := map[string]uint64{}
	for _, e := range a.Entries {
		out[e.Task+"/"+e.Node] = e.Calls
	}
	return out
}
func probeDelta(t *testing.T, a, b nodeProbeAudit) map[string]uint64 {
	t.Helper()
	x, y := probeCounts(a), probeCounts(b)
	out := map[string]uint64{}
	for k, n := range y {
		if n < x[k] {
			t.Fatal("counter regressed")
		}
		if n > x[k] {
			out[k] = n - x[k]
		}
	}
	return out
}
func probeNames(nodes []*node_info.NodeInfo) []string {
	out := []string{}
	for _, n := range nodes {
		out = append(out, n.Name)
	}
	return out
}
func probeAuditValid(t *testing.T, a nodeProbeAudit) {
	t.Helper()
	if !a.Complete || a.Diagnostics != 0 || a.Peak < 1 {
		t.Fatal("incomplete native scoring evidence", a)
	}
	var total uint64
	for _, e := range a.Entries {
		total += e.Calls
		if e.Score != 0 || e.Error != nil {
			t.Fatal("not neutral zero/nil", e)
		}
	}
	if total != a.Total {
		t.Fatal("unaccounted callback")
	}
}

type probeScoringReceipt struct {
	Before          json.RawMessage    `json:"before"`
	After           json.RawMessage    `json:"after"`
	Eligible        []string           `json:"eligible_nodes"`
	PredicateErrors map[string]*string `json:"predicate_errors"`
	Ordered         []string           `json:"native_ordered_nodes"`
	Scores          map[string]float64 `json:"native_aggregate_scores"`
	NativeErrors    map[string]*string `json:"native_score_errors"`
	StressOrder     []string           `json:"native_stress_order"`
}

func probeScoring(t *testing.T, f nativeFixture, r *nodeProbeRecord, dir, arm string) probeScoringReceipt {
	t.Helper()
	s := f.session
	q := s.ClusterInfo.Queues["queue0"]
	if len(s.ClusterInfo.Nodes) != 2 || len(s.ClusterInfo.PodGroupInfos) != 1 || f.job.Priority != 0 || string(f.job.Queue) != "queue0" || q == nil || q.Resources.GPU.Quota != 0 || q.Resources.CPU.Quota != -1 || q.Resources.Memory.Quota != -1 {
		t.Fatal("frozen B4 B queue/priority/topology changed")
	}
	out := probeScoringReceipt{Before: probeDomain(t, f), Eligible: []string{}, PredicateErrors: map[string]*string{}, Scores: map[string]float64{}, NativeErrors: map[string]*string{}}
	if err := s.PrePredicateFn(f.task, f.job); err != nil {
		t.Fatal(err)
	}
	candidates := []*node_info.NodeInfo{}
	for _, name := range []string{"node1", "node0"} {
		n := s.ClusterInfo.Nodes[name]
		err := s.PredicateFn(f.task, f.job, n)
		out.PredicateErrors[name] = nil
		if err != nil {
			msg := err.Error()
			out.PredicateErrors[name] = &msg
			t.Fatal("B4 B predicate rejected node", name, err)
		}
		if !s.FittingNode(f.task, n, false) {
			t.Fatal("B4 B native resource check rejected node", name)
		}
		candidates = append(candidates, n)
	}
	// Canonical eligible-set evidence only; NEVER sort the native ordered output.
	out.Eligible = []string{"node0", "node1"}
	out.Ordered = probeNames(s.OrderedNodesByTask(candidates, f.task))
	if !reflect.DeepEqual(out.Ordered, []string{"node0", "node1"}) {
		t.Fatal("native strict-score order changed", out.Ordered)
	}
	if r != nil {
		audit := r.audit()
		probeAuditValid(t, audit)
		if !reflect.DeepEqual(probeCounts(audit), map[string]uint64{"native-job-0/node0": 1, "native-job-0/node1": 1}) || audit.Total != 2 {
			t.Fatal("controlled OrderedNodesByTask must call each passed node exactly once", audit)
		}
		nativeSave(t, dir, arm+"-controlled-probe.json", audit)
	}
	for _, n := range candidates {
		score, err := s.NodeOrderFn(f.task, n)
		out.NativeErrors[n.Name] = nil
		if err != nil {
			msg := err.Error()
			out.NativeErrors[n.Name] = &msg
			t.Fatal("native score error", err)
		}
		out.Scores[n.Name] = score
	}
	if out.Scores["node0"] != 1019 || out.Scores["node1"] != 1010 {
		t.Fatal("original strict B4 B scores changed", out.Scores)
	}
	if r != nil {
		a := r.audit()
		probeAuditValid(t, a)
		nativeSave(t, dir, arm+"-aggregate-queries-probe.json", a)
	}
	// Small concurrency exercise over 16 references to the SAME two eligible
	// nodes, not 16 distinct nodes or an allocator candidate-set experiment.
	repeated := []*node_info.NodeInfo{}
	for i := 0; i < 8; i++ {
		repeated = append(repeated, candidates...)
	}
	out.StressOrder = probeNames(s.OrderedNodesByTask(repeated, f.task))
	want := []string{}
	for i := 0; i < 8; i++ {
		want = append(want, "node0")
	}
	for i := 0; i < 8; i++ {
		want = append(want, "node1")
	}
	if !reflect.DeepEqual(out.StressOrder, want) {
		t.Fatal("native ordered stress output changed", out.StressOrder)
	}
	if r != nil {
		a := r.audit()
		probeAuditValid(t, a)
		if !reflect.DeepEqual(probeCounts(a), map[string]uint64{"native-job-0/node0": 10, "native-job-0/node1": 10}) || a.Total != 20 {
			t.Fatal("native phase callback accounting mismatch", a)
		}
		nativeSave(t, dir, arm+"-concurrent-probe.json", a)
	}
	out.After = probeDomain(t, f)
	if !bytes.Equal(out.Before, out.After) {
		t.Fatal("controlled native scoring mutated public domain state")
	}
	return out
}

type probeActionReceipt struct {
	Before   json.RawMessage   `json:"before"`
	After    json.RawMessage   `json:"after"`
	Config   json.RawMessage   `json:"configuration"`
	Bind     []json.RawMessage `json:"actual_bind_calls"`
	Pipeline []json.RawMessage `json:"actual_pipeline_calls"`
	Evict    []json.RawMessage `json:"actual_evict_calls"`
	UID      string            `json:"task_uid"`
	Status   string            `json:"task_status"`
	Node     string            `json:"chosen_node"`
	Result   string            `json:"action_result"`
}

func probeAllocate(t *testing.T, f nativeFixture, mock *cache.MockCache) probeActionReceipt {
	t.Helper()
	out := probeActionReceipt{Before: probeDomain(t, f), Config: nativeJSON(t, f.session.Config), Bind: []json.RawMessage{}, Pipeline: []json.RawMessage{}, Evict: []json.RawMessage{}, Result: "Execute returns void; native state and exact nil-return fake Bind checked; no real transaction certificate"}
	mock.EXPECT().Bind(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).DoAndReturn(func(p *pod_info.PodInfo, node string, annotations map[string]string, numa cache.NUMAPrediction) error {
		out.Bind = append(out.Bind, nativeJSON(t, map[string]any{"task": nativeTaskState(p), "node": node, "annotations": annotations, "numa": numa, "returned_error": nil}))
		return nil
	}).Times(1)
	mock.EXPECT().TaskPipelined(gomock.Any(), gomock.Any()).Times(0)
	mock.EXPECT().Evict(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Times(0)
	allocate.New().Execute(f.session)
	// Authoritative post-solver indexes, including a possible published clone.
	f.job = f.session.ClusterInfo.PodGroupInfos[common_info.NewPodGroupID("", "native-job")]
	if f.job == nil {
		t.Fatal("lost job")
	}
	f.task = f.job.GetAllPodsMap()[common_info.PodID("native-job-0")]
	if f.task == nil {
		t.Fatal("lost task")
	}
	out.After = probeDomain(t, f)
	out.UID = string(f.task.UID)
	out.Status = f.task.Status.String()
	out.Node = f.task.NodeName
	if f.task.Status != pod_status.Binding || out.Node != "node0" || len(out.Bind) != 1 {
		t.Fatal("native Allocate oracle failed", out.Status, out.Node)
	}
	return out
}

func TestNeutralNodeScoringProbe(t *testing.T) {
	if runtime.Version() != "go1.26.8" {
		t.Fatal("exact Go 1.26.8 required")
	}
	test_utils.InitTestingInfrastructure()
	// Serial test-only registration. No public unregister API exists: the unique
	// builder remains until this bounded test process exits, no deployment claim.
	builders := map[string]uintptr{}
	for _, name := range []string{"predicates", "nodeplacement", shadowName} {
		b, ok := framework.GetPluginBuilder(name)
		if ok {
			builders[name] = reflect.ValueOf(b).Pointer()
		}
	}
	if _, exists := framework.GetPluginBuilder(nodeProbeName); exists {
		t.Fatal("unique probe registry name already occupied")
	}
	framework.RegisterPluginBuilder(nodeProbeName, newNodeScoringProbe)
	builder, ok := framework.GetPluginBuilder(nodeProbeName)
	if !ok || builder == nil {
		t.Fatal("registry lost builder")
	}
	args := framework.PluginArguments{"lambda": "0"}
	plugin := builder(args).(*nodeScoringProbe)
	if plugin.Name() != nodeProbeName || !reflect.DeepEqual(args, framework.PluginArguments{"lambda": "0"}) {
		t.Fatal("constructor/identity/argument mutation")
	}
	for name, want := range builders {
		b, ok := framework.GetPluginBuilder(name)
		if !ok || reflect.ValueOf(b).Pointer() != want {
			t.Fatal("existing builder overwritten", name)
		}
	}
	dir := t.TempDir()
	if root := os.Getenv("POC1B_NODE_PROBE_EVIDENCE_DIR"); root != "" {
		dir = root
		if err := os.MkdirAll(dir, 0700); err != nil {
			t.Fatal(err)
		}
	}
	nativeSave(t, dir, "registry.json", map[string]any{"name": plugin.Name(), "test_only": true, "old_builders_unchanged": true, "unregister": "not available; process lifetime"})
	var firstScore *probeScoringReceipt
	var firstAction *probeActionReceipt
	var sealed []*nodeProbeRecord
	var seals []nodeProbeAudit
	for repeat := 0; repeat < 3; repeat++ {
		t.Run(fmt.Sprintf("pair_%d_G1_G6", repeat), func(t *testing.T) {
			d := filepath.Join(dir, fmt.Sprintf("pair-%d", repeat))
			if err := os.MkdirAll(d, 0700); err != nil {
				t.Fatal(err)
			}
			off, offCache, offCtrl := actionFixture(t, "B")
			on, onCache, onCtrl := actionFixture(t, "B")
			if off.session == on.session || offCache == onCache || offCtrl == onCtrl || off.session.Cache.KubeClient() == on.session.Cache.KubeClient() || off.session.ClusterInfo == on.session.ClusterInfo || off.task == on.task || off.job == on.job || off.task.Pod == on.task.Pod || off.task.VectorMap == on.task.VectorMap || &off.task.ResReqVector[0] == &on.task.ResReqVector[0] || &off.job.AllocatedVector[0] == &on.job.AllocatedVector[0] {
				t.Fatal("shared mutable arm state")
			}
			for _, name := range []string{"node0", "node1"} {
				a, b := off.session.ClusterInfo.Nodes[name], on.session.ClusterInfo.Nodes[name]
				if a == b || &a.IdleVector[0] == &b.IdleVector[0] {
					t.Fatal("shared native nodes/vectors")
				}
			}
			if plugin.record(off.session) != nil {
				t.Fatal("OFF attached")
			}
			hooks := probeHooks(t, on.session)
			state := probeDomain(t, on)
			config := nativeJSON(t, on.session.Config)
			plugin.OnSessionOpen(on.session)
			t.Cleanup(func() { plugin.OnSessionClose(on.session) })
			record := plugin.record(on.session)
			if record == nil {
				t.Fatal("no session-local record")
			}
			for _, old := range sealed {
				if old == record {
					t.Fatal("reused scoring record")
				}
			}
			expected := map[string]int{}
			for k, n := range hooks {
				expected[k] = n
			}
			expected["NodeOrderFns"]++
			if !reflect.DeepEqual(probeHooks(t, on.session), expected) || !bytes.Equal(state, probeDomain(t, on)) || !bytes.Equal(config, nativeJSON(t, on.session.Config)) {
				t.Fatal("open added side effects beyond one node scoring callback")
			}
			nativeSave(t, d, "hook-counts.json", map[string]any{"before": hooks, "after": probeHooks(t, on.session)})
			a, b := probeScoring(t, off, nil, d, "off"), probeScoring(t, on, record, d, "on")
			nativeSave(t, d, "off-scoring.json", a)
			nativeSave(t, d, "on-scoring.json", b)
			if !reflect.DeepEqual(a, b) {
				t.Fatal("native eligible set/order/score/state difference")
			}
			if firstScore == nil {
				firstScore = &a
			} else if !reflect.DeepEqual(*firstScore, a) {
				t.Fatal("scoring repetition difference")
			}
			start := record.audit()
			baseline, observed := probeAllocate(t, off, offCache), probeAllocate(t, on, onCache)
			finish := record.audit()
			probeAuditValid(t, finish)
			delta := probeDelta(t, start, finish)
			var actual uint64
			for key, n := range delta {
				if key != "native-job-0/node0" && key != "native-job-0/node1" {
					t.Fatal("unexpected Action scoring key", key)
				}
				actual += n
			}
			if actual == 0 || finish.Total-start.Total != actual {
				t.Fatal("native Allocate did not call probe")
			}
			nativeSave(t, d, "native-action-probe.json", map[string]any{"before": start, "after": finish, "actual_native_action_counts": delta})
			nativeSave(t, d, "off-action.json", baseline)
			nativeSave(t, d, "on-action.json", observed)
			if !reflect.DeepEqual(baseline, observed) {
				t.Fatal("native Allocate/cache/state difference")
			}
			if firstAction == nil {
				firstAction = &baseline
			} else if !reflect.DeepEqual(*firstAction, baseline) {
				t.Fatal("Action repetition difference")
			}
			plugin.OnSessionClose(on.session)
			if plugin.record(on.session) != nil {
				t.Fatal("closed session retained")
			}
			seal := record.audit()
			probeAuditValid(t, seal)
			if !seal.Closed {
				t.Fatal("unsealed record")
			}
			off, on = probeRefresh(t, off), probeRefresh(t, on)
			// Active native scoring after close (not a no-work Action) still returns
			// native nodes, but the sealed probe record must never change.
			nodesOff := []*node_info.NodeInfo{off.session.ClusterInfo.Nodes["node1"], off.session.ClusterInfo.Nodes["node0"]}
			nodesOn := []*node_info.NodeInfo{on.session.ClusterInfo.Nodes["node1"], on.session.ClusterInfo.Nodes["node0"]}
			beforeOff, beforeOn := probeDomain(t, off), probeDomain(t, on)
			postOff, postOn := probeNames(off.session.OrderedNodesByTask(nodesOff, off.task)), probeNames(on.session.OrderedNodesByTask(nodesOn, on.task))
			if !reflect.DeepEqual(postOff, postOn) || len(postOn) != 2 || !reflect.DeepEqual(record.audit(), seal) || !bytes.Equal(beforeOff, probeDomain(t, off)) || !bytes.Equal(beforeOn, probeDomain(t, on)) {
				t.Fatal("active post-close scoring mutation or node drop")
			}
			sealed = append(sealed, record)
			seals = append(seals, seal)
			for i, old := range sealed {
				if !reflect.DeepEqual(old.audit(), seals[i]) {
					t.Fatal("cross-session contamination")
				}
			}
			nativeSave(t, d, "sealed-probe.json", seal)
			nativeSave(t, d, "post-close-native-order.json", map[string]any{"off": postOff, "on": postOn, "active_score_callback_invoked": true})
			t.Logf("G1-G6 pair=%d eligible=2 order=%v native scores=%v controlled probe calls=2 aggregate-query calls=2 stress calls=16 actual Allocate counts=%v observed peak=%d Bind=1 Pipeline=0 Evict=0", repeat, b.Ordered, b.Scores, delta, seal.Peak)
		})
	}
	// Diagnostic fallback uses public native NodeOrderFn, no failing native plugin
	// or fake callback. No pool is created for this isolated invalid-input check.
	s := &framework.Session{ID: "probe-diagnostic-fallback"}
	p := builder(nil).(*nodeScoringProbe)
	p.OnSessionOpen(s)
	r := p.record(s)
	if score, err := s.NodeOrderFn(nil, nil); score != 0 || err != nil {
		t.Fatal("diagnostic became node-dropping error", score, err)
	}
	task := &pod_info.PodInfo{UID: common_info.PodID("diagnostic-task")}
	for i := 0; i <= nodeProbeCapacity; i++ {
		if score, err := s.NodeOrderFn(task, &node_info.NodeInfo{Name: fmt.Sprintf("diagnostic-node-%d", i)}); score != 0 || err != nil {
			t.Fatal("bounded metadata overflow changed score/error")
		}
	}
	p.OnSessionClose(s)
	audit := r.audit()
	if !audit.Closed || audit.Complete || audit.Diagnostics != 2 || len(audit.Entries) != nodeProbeCapacity || audit.Total != nodeProbeCapacity+2 {
		t.Fatal("fallback evidence not bounded", audit)
	}
	if score, err := s.NodeOrderFn(task, &node_info.NodeInfo{Name: "post-close"}); score != 0 || err != nil || !reflect.DeepEqual(audit, r.audit()) {
		t.Fatal("closed diagnostic record changed")
	}
	nativeSave(t, dir, "diagnostic-fallback.json", audit)
	nativeSave(t, dir, "scope.json", map[string]any{"claim": "NEUTRAL_NODE_SCORING_DECISION_INVARIANT (tested fixtures only)", "test_failed": t.Failed(), "fixture": "frozen B4 B: native-job-0 requests 2cores/1GiB/0GPU, node0=4cores/8GiB and node1=8cores/8GiB, both native predicates/fit pass, original scores1019>1010", "metadata": "sorted count entries deterministic; observed peak is actual scheduling-dependent instrumentation, raw only, not a required overlap/performance proof; no forced waits/yields", "stress": "one bounded native OrderedNodesByTask call over16 references to the same2 eligible nodes; not16 distinct candidates", "close": "active OrderedNodesByTask after sealing invokes existing neutral callback without metadata writes; no unregister/detach API claim", "event_handler_count": "read-only reflection length for pinned private slice only; no private handler values, unsafe or mutation", "trace": "probe scores are separate metadata, not B3a Statement Trace events; no new trace schema", "limits": "no production registration/entrypoint/deployment, nonzero lambda, LP/PGFP, real K8s/GPU/NUMA, full Scheduler or algorithm performance claim; frozen modfile/informer/race/cold Rollback limits retained"})
}
