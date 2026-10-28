package main

// M4-B Stage A: native multi-task policy differential on the pinned KAI Fake Session.
// NATIVE_OFF, NATIVE_DYNAMIC_PGFP (alpha=0) and NATIVE_DYNAMIC_PGFP_LP (alpha=.5) each build
// a fresh Session from the same immutable trace, call the actual OrderedNodesByTask /
// NodeOrderFn / allocate.New().Execute with the fake Cache, and carry forward their own
// native state. Every dynamic arrival gets a fresh certified M3-A artifact and a one-use
// grant (the only factory is here, in _test.go). Held-out future arrivals are replayed
// through the same native Allocate only after all current decisions are frozen.
// Controlled local test harness; not a production path, signature or Kubernetes binding.

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strconv"
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
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/test_utils/jobs_fake"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/test_utils/nodes_fake"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/test_utils/tasks_fake"
	"go.uber.org/mock/gomock"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

const (
	m4bLambda = 40.0
	m4bBF     = 4.0
	m4bBP     = 8.0
)

type m4bForecast struct {
	ID       string   `json:"id"`
	Demand   []int    `json:"demand"`
	Weight   int      `json:"weight"`
	Eligible []string `json:"eligible"`
}

// m4bTrace is the immutable common trace. Cores are whole CPU cores; every task asks 1 GiB.
type m4bTrace struct {
	Name     string        `json:"name"`
	Cores    [2]float64    `json:"node_cores"`
	Arrivals []int         `json:"arrival_cores"`
	Forecast []m4bForecast `json:"forecast"`
	Future   []int         `json:"held_out_future_cores"`
}

var m4bBoth = []string{"node0", "node1"}

// Declared before any run; never selected or tuned on outcomes.
var m4bTraces = []m4bTrace{
	{"matched", [2]float64{6, 8}, []int{2, 2, 2}, []m4bForecast{{"F0", []int{0, 4, 1}, 2, m4bBoth}}, []int{4, 4}},
	{"shifted", [2]float64{6, 8}, []int{2, 2, 2}, []m4bForecast{{"F0", []int{0, 4, 1}, 2, m4bBoth}}, []int{2, 2, 2}},
	{"equal_nodes", [2]float64{8, 8}, []int{2, 2}, []m4bForecast{{"F0", []int{0, 4, 1}, 2, m4bBoth}}, []int{4, 4, 4}},
	// Added after the first dev run showed the three traces above all at admission ceiling (disclosed
	// in the PR): designed from the mechanism (hole preservation), not found by outcome search.
	{"fragmentation", [2]float64{6, 6}, []int{2, 2}, []m4bForecast{{"F0", []int{0, 4, 1}, 2, m4bBoth}}, []int{4, 4}},
	{"shift_harm", [2]float64{6, 6}, []int{2, 2}, []m4bForecast{{"F0", []int{0, 4, 1}, 2, m4bBoth}}, []int{6}},
}

var m4bArms = []string{"native_off", "native_dynamic_pgfp", "native_dynamic_pgfp_lp"}

type m4bTimes struct {
	Fixture, Producer, Begin, NativeScore, Allocate float64
}

type m4bEnv struct {
	f      nativeFixture
	mock   *cache.MockCache
	binds  *[]json.RawMessage
	plugin *artifactBridgeProbe
	trace  m4bTrace
}

func m4bFixture(t *testing.T, tr m4bTrace, arm string) *m4bEnv {
	t.Helper()
	ctrl := gomock.NewController(t)
	metadata := test_utils.TestTopologyBasic{
		Name:   "M4B-" + tr.Name,
		Jobs:   []*jobs_fake.TestJobBasic{{Name: "native-job", QueueName: "queue0", RequiredCPUsPerTask: float64(tr.Arrivals[0]), RequiredMemoryPerTask: nativeGiB, Tasks: []*tasks_fake.TestTaskBasic{{State: pod_status.Pending}}}},
		Nodes:  map[string]nodes_fake.TestNodeBasic{"node0": {CPUMillis: tr.Cores[0], CPUMemory: 8 * nativeGiB}, "node1": {CPUMillis: tr.Cores[1], CPUMemory: 8 * nativeGiB}},
		Queues: []test_utils.TestQueueBasic{{Name: "queue0"}},
		Mocks:  &test_utils.TestMock{CacheRequirements: &test_utils.CacheMocking{}},
	}
	s := test_utils.BuildSession(metadata, ctrl)
	s.ID = "m4b-" + tr.Name + "-" + arm
	mock, ok := s.Cache.(*cache.MockCache)
	if !ok {
		t.Fatal("not official cache mock")
	}
	job := s.ClusterInfo.PodGroupInfos[common_info.NewPodGroupID("", "native-job")]
	task := job.GetAllPodsMap()[common_info.PodID("native-job-0")]
	if task == nil || task.ResReqVector.Get(resource_info.CPUIndex) != float64(tr.Arrivals[0])*1000 {
		t.Fatal("official fake resource units changed")
	}
	binds := &[]json.RawMessage{}
	mock.EXPECT().Bind(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).DoAndReturn(func(p *pod_info.PodInfo, n string, a map[string]string, numa cache.NUMAPrediction) error {
		*binds = append(*binds, nativeJSON(t, map[string]any{"task": p.UID, "node": n, "annotations": a, "numa": numa, "returned_error": nil}))
		return nil
	}).AnyTimes()
	mock.EXPECT().TaskPipelined(gomock.Any(), gomock.Any()).Times(0)
	mock.EXPECT().Evict(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Times(0)
	mock.EXPECT().RecordJobStatusEvent(gomock.Any(), gomock.Any()).Return(nil).Times(1)
	mock.EXPECT().WaitForWorkers(gomock.Any()).Times(1)
	t.Cleanup(func() { framework.CloseSession(s); ctrl.Finish() })
	e := &m4bEnv{f: nativeFixture{s, job, task}, mock: mock, binds: binds, plugin: newArtifactBridge(nil).(*artifactBridgeProbe), trace: tr}
	// The identical hook set is installed in every arm; NATIVE_OFF never activates a phase.
	e.plugin.OnSessionOpen(s)
	t.Cleanup(func() { e.plugin.OnSessionClose(s) })
	return e
}

func (e *m4bEnv) arrive(t *testing.T, index, cores int) {
	t.Helper()
	if index == 0 {
		return
	}
	req := e.f.job.GetAllPodsMap()[common_info.PodID("native-job-0")].Pod.Spec.Containers[0].Resources.Requests.DeepCopy()
	req[v1.ResourceCPU] = *resource.NewQuantity(int64(cores), resource.DecimalSI)
	name := fmt.Sprintf("native-job-%d", index)
	pod := tasks_fake.BuildPod(name, "", &tasks_fake.TestTaskBasic{State: pod_status.Pending}, v1.PodPending, req, "", 0, nil, "native-job")
	next := pod_info.NewTaskInfo(pod, e.f.job.GetAllPodsMap()[common_info.PodID("native-job-0")].VectorMap)
	e.f.job.AddTaskInfo(next)
	if next.Status != pod_status.Pending || next.NodeName != "" {
		t.Fatal("arrival")
	}
	e.f.task = next
}

type m4bProduced struct {
	raw      []byte
	grant    *bridgeGrant
	artifact []byte
	controls []byte
	envelope string
	seconds  float64
}

// m4bProduce is the only grant factory for M4-B. The controlled Python process, committed
// sources and byte-equal stdout bind the grant; the forecast is the trace's, never realized.
func m4bProduce(t *testing.T, e *m4bEnv, phase int, alpha float64, dir string) m4bProduced {
	t.Helper()
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	root, _ := filepath.Abs("..")
	paths := []string{"src/accelsched_kai/m4b_producer.py", "src/accelsched_kai/bridge_envelope.py", "src/accelsched_kai/bounded_advice.py", "src/accelsched_kai/adapter.py", "src/accelsched_kai/oracle.py", "src/accelsched_kai/advisor.py", "src/accelsched_kai/schema.py"}
	var producerHash string
	for _, name := range paths {
		b, err := os.ReadFile(filepath.Join(root, name))
		if err != nil {
			t.Fatal(err)
		}
		b = bytes.ReplaceAll(b, []byte("\r\n"), []byte("\n"))
		expected, err := exec.Command("git", "-C", root, "show", "HEAD:"+name).Output()
		if err != nil || !bytes.Equal(b, expected) {
			t.Fatal("uncommitted/mismatched controlled producer source ", name, err)
		}
		if name == paths[0] {
			producerHash = bridgeHash(b)
		}
	}
	praw, err := bridgeProjection(t, e.f)
	if err != nil {
		t.Fatal(err)
	}
	request := nativeJSON(t, map[string]any{"projection_b64": base64.StdEncoding.EncodeToString(praw), "phase": phase, "alpha": alpha, "BF": m4bBF, "BP": m4bBP, "forecast": e.trace.Forecast})
	input := filepath.Join(dir, "request.json")
	if err = os.WriteFile(input, request, 0600); err != nil {
		t.Fatal(err)
	}
	python := os.Getenv("M3B_PYTHON")
	if python == "" {
		t.Fatal("M3B_PYTHON must identify the existing frozen environment")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	code := "import runpy,sys;sys.path.insert(0,sys.argv[1]);sys.argv=sys.argv[2:];runpy.run_module('accelsched_kai.m4b_producer',run_name='__main__')"
	output := filepath.Join(dir, "producer")
	cmd := exec.CommandContext(ctx, python, "-I", "-B", "-c", code, filepath.Join(root, "src"), "m4b_producer", input, output)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	start := time.Now()
	raw, runErr := cmd.Output()
	seconds := time.Since(start).Seconds()
	_ = os.WriteFile(filepath.Join(dir, "producer.stderr.txt"), stderr.Bytes(), 0600)
	if runErr != nil {
		t.Fatal("controlled producer failed ", runErr, stderr.String())
	}
	file, err := os.ReadFile(filepath.Join(output, "envelope.json"))
	if err != nil || !bytes.Equal(file, raw) {
		t.Fatal("producer handoff mismatch")
	}
	artifact, _ := os.ReadFile(filepath.Join(output, "artifact.json"))
	controls, _ := os.ReadFile(filepath.Join(output, "controls.json"))
	hooks := []uintptr{}
	for _, fn := range e.f.session.PredicateFns {
		hooks = append(hooks, reflect.ValueOf(fn).Pointer())
	}
	grant := &bridgeGrant{owner: e.f.session, task: e.f.task, nodes: map[string]*node_info.NodeInfo{}, phase: phase, envelopeHash: bridgeHash(raw), projectionHash: bridgeHash(praw), producerHash: producerHash}
	for name, n := range e.f.session.ClusterInfo.Nodes {
		grant.nodes[name] = n
	}
	f := e.f
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
	return m4bProduced{raw: raw, grant: grant, artifact: artifact, controls: controls, envelope: bridgeHash(raw), seconds: seconds}
}

func m4bFeasible(e *m4bEnv) (map[string]bool, map[string]string) {
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

func m4bResources(t *testing.T, s *framework.Session) map[string]any {
	t.Helper()
	out := map[string]any{}
	for _, name := range m4bBoth {
		n := s.ClusterInfo.Nodes[name]
		entry := map[string]any{}
		for key, vec := range map[string]interface{ Get(int) float64 }{"allocatable": n.AllocatableVector, "idle": n.IdleVector, "used": n.UsedVector, "releasing": n.ReleasingVector} {
			x := []int64{}
			for _, i := range []int{resource_info.GPUIndex, resource_info.CPUIndex, resource_info.MemoryIndex} {
				x = append(x, int64(vec.Get(i)))
			}
			entry[key] = x
		}
		ids := []string{}
		for id := range n.PodInfos {
			ids = append(ids, string(id))
		}
		sortStrings(ids)
		entry["members"] = ids
		out[name] = entry
	}
	return out
}

func sortStrings(a []string) {
	for i := 1; i < len(a); i++ {
		for j := i; j > 0 && a[j] < a[j-1]; j-- {
			a[j], a[j-1] = a[j-1], a[j]
		}
	}
}

func m4bConserved(before, after map[string]any, node string, millis, mem int64) error {
	for _, name := range m4bBoth {
		b, a := before[name].(map[string]any), after[name].(map[string]any)
		for _, key := range []string{"allocatable", "releasing"} {
			if !reflect.DeepEqual(b[key], a[key]) {
				return fmt.Errorf("%s %s changed", name, key)
			}
		}
		bi, ai := b["idle"].([]int64), a["idle"].([]int64)
		bu, au := b["used"].([]int64), a["used"].([]int64)
		for k := 0; k < 3; k++ {
			if bi[k]+bu[k] != ai[k]+au[k] {
				return fmt.Errorf("%s idle+used not conserved", name)
			}
		}
		dm, dr := int64(0), int64(0)
		if name == node {
			dm, dr = millis, mem
		}
		if bi[0] != ai[0] || bi[1]-ai[1] != dm || bi[2]-ai[2] != dr {
			return fmt.Errorf("%s idle delta wrong", name)
		}
	}
	return nil
}

func m4bScores(e *m4bEnv, t *testing.T) (map[string]float64, []string, float64) {
	t.Helper()
	s := e.f.session
	nodes := []*node_info.NodeInfo{s.ClusterInfo.Nodes["node1"], s.ClusterInfo.Nodes["node0"]}
	start := time.Now()
	order := probeNames(s.OrderedNodesByTask(nodes, e.f.task))
	scores := map[string]float64{}
	for _, n := range nodes {
		v, err := s.NodeOrderFn(e.f.task, n)
		if err != nil {
			t.Fatal(err)
		}
		scores[n.Name] = v
	}
	return scores, order, time.Since(start).Seconds()
}

func m4bHash(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }

// runArm drives one arm through every current arrival and then the held-out future.
func m4bRunArm(t *testing.T, tr m4bTrace, arm, dir string) (map[string]any, map[string]any) {
	t.Helper()
	times := m4bTimes{}
	start := time.Now()
	e := m4bFixture(t, tr, arm)
	times.Fixture = time.Since(start).Seconds()
	alpha := 0.0
	if arm == "native_dynamic_pgfp_lp" {
		alpha = 0.5
	}
	steps := []any{}
	phase := 0
	for i, cores := range tr.Arrivals {
		e.arrive(t, i, cores)
		stepDir := filepath.Join(dir, fmt.Sprintf("arrival-%d", i))
		if err := os.MkdirAll(stepDir, 0700); err != nil {
			t.Fatal(err)
		}
		beforeRes := m4bResources(t, e.f.session)
		beforeDomain := probeDomain(t, e.f)
		feasible, perr := m4bFeasible(e)
		s0, order0, secs := m4bScores(e, t) // plugin inactive: exactly the native score S0
		times.NativeScore += secs
		step := map[string]any{"index": i, "task": string(e.f.task.UID), "request_cores": cores, "before_resources": beforeRes, "before_domain_sha256": m4bHash(beforeDomain),
			"feasible": feasible, "predicate_errors": perr, "native_scores_S0": s0, "native_order_S0": order0}
		var produced m4bProduced
		active := false
		if arm != "native_off" {
			phase++
			produced = m4bProduce(t, e, phase, alpha, filepath.Join(stepDir, "handoff"))
			times.Producer += produced.seconds
			begin := time.Now()
			active = e.plugin.begin(produced.raw, produced.grant, m4bLambda)
			times.Begin += time.Since(begin).Seconds()
			if !active {
				t.Fatal("fresh certified artifact did not activate at arrival ", i)
			}
			step["lambda"], step["alpha"], step["BF"], step["BP"] = m4bLambda, alpha, m4bBF, m4bBP
			step["envelope_sha256"], step["artifact_sha256"] = produced.envelope, m4bHash(produced.artifact)
			step["controls"] = json.RawMessage(produced.controls)
			step["artifact_pairs"] = m4bPairs(t, produced.artifact)
		}
		scores, order, secs := m4bScores(e, t)
		times.NativeScore += secs
		step["native_scores_S"], step["native_order_S"] = scores, order
		if active {
			audit := e.plugin.record.audit()
			step["advice_entries_returned"] = audit.Entries
			step["score_calls_before_action"] = audit.Total
			// Attribution: the native total is exactly the original native S0 plus the returned lambda*A.
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
		job := e.f.session.ClusterInfo.PodGroupInfos[e.f.job.UID]
		p := job.GetAllPodsMap()[e.f.task.UID]
		if p == nil || p.Status != pod_status.Binding || len(*e.binds)-binds != 1 {
			t.Fatal("native Allocate did not bind exactly one task at arrival ", i, " status ", p.Status)
		}
		afterRes := m4bResources(t, e.f.session)
		millis := int64(cores) * 1000
		if err := m4bConserved(beforeRes, afterRes, p.NodeName, millis, nativeGiB); err != nil {
			t.Fatal("resource conservation: ", err)
		}
		afterDomain := probeDomain(t, e.f)
		step["chosen_node"], step["status_after"], step["bind"] = p.NodeName, p.Status.String(), (*e.binds)[len(*e.binds)-1]
		step["after_resources"], step["after_domain_sha256"] = afterRes, m4bHash(afterDomain)
		steps = append(steps, step)
	}
	// Held-out future: replayed only now, identically for every arm, through native Allocate with no advice.
	future := []any{}
	admitted, admittedCores := 0, 0
	for k, cores := range tr.Future {
		e.arrive(t, len(tr.Arrivals)+k, cores)
		feasible, perr := m4bFeasible(e)
		before := m4bResources(t, e.f.session)
		binds := len(*e.binds)
		allocStart := time.Now()
		allocate.New().Execute(e.f.session)
		times.Allocate += time.Since(allocStart).Seconds()
		p := e.f.session.ClusterInfo.PodGroupInfos[e.f.job.UID].GetAllPodsMap()[e.f.task.UID]
		row := map[string]any{"index": k, "task": string(e.f.task.UID), "request_cores": cores, "feasible": feasible, "predicate_errors": perr, "before_resources": before}
		if len(*e.binds)-binds == 1 && p.Status == pod_status.Binding {
			admitted++
			admittedCores += cores
			row["outcome"], row["node"] = "ACCEPTED", p.NodeName
			if err := m4bConserved(before, m4bResources(t, e.f.session), p.NodeName, int64(cores)*1000, nativeGiB); err != nil {
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
	final := m4bResources(t, e.f.session)
	record := map[string]any{"arm": arm, "trace": tr, "steps": steps, "future": future, "future_admitted": admitted, "future_admitted_cores": admittedCores,
		"future_total": len(tr.Future), "final_resources": final, "hooks": probeHooks(t, e.f.session), "bind_count": len(*e.binds),
		"native_future_admission": "NATIVE_ALLOCATE_NO_ADVICE_IDENTICAL_FOR_ALL_ARMS"}
	timing := map[string]any{"arm": arm, "trace": tr.Name, "seconds": times}
	return record, timing
}

func m4bPairs(t *testing.T, artifact []byte) any {
	var a struct {
		Normalization any `json:"normalization"`
		Pairs         any `json:"pairs"`
		Lp            struct {
			Prices any `json:"prices"`
			Hash   any `json:"price_sha256"`
		} `json:"lp"`
		Placement struct {
			Audit any `json:"independent_audit"`
		} `json:"placement"`
		Sha any `json:"artifact_sha256"`
	}
	if json.Unmarshal(artifact, &a) != nil {
		t.Fatal("artifact bytes")
	}
	return map[string]any{"normalization": a.Normalization, "pairs": a.Pairs, "prices": a.Lp.Prices, "price_sha256": a.Lp.Hash, "independent_audit": a.Placement.Audit, "artifact_sha256": a.Sha}
}

func m4bEvidenceDir(t *testing.T) string {
	if d := os.Getenv("M4B_EVIDENCE_DIR"); d != "" {
		if err := os.MkdirAll(d, 0700); err != nil {
			t.Fatal(err)
		}
		return d
	}
	return t.TempDir()
}

func TestM4BStageANativeDifferential(t *testing.T) {
	test_utils.InitTestingInfrastructure()
	dir := m4bEvidenceDir(t)
	normalized := map[string]any{"version": "M4B-A1", "mode": "NATIVE_FAKE_SESSION_CPU_PROBE_ONLY", "lambda": m4bLambda, "BF": m4bBF, "BP": m4bBP, "traces": map[string]any{}}
	timings := []any{}
	for _, tr := range m4bTraces {
		perTrace := map[string]any{}
		for _, arm := range m4bArms {
			t.Run(tr.Name+"/"+strconv.Quote(arm), func(t *testing.T) {
				d := filepath.Join(dir, tr.Name, arm)
				if err := os.MkdirAll(d, 0700); err != nil {
					t.Fatal(err)
				}
				rec, tim := m4bRunArm(t, tr, arm, d)
				perTrace[arm] = rec
				timings = append(timings, tim)
			})
		}
		// G1: identical authoritative native state and native scores at the first arrival in every arm.
		ref := perTrace["native_off"].(map[string]any)["steps"].([]any)[0].(map[string]any)
		for _, arm := range m4bArms[1:] {
			got := perTrace[arm].(map[string]any)["steps"].([]any)[0].(map[string]any)
			for _, key := range []string{"before_domain_sha256", "before_resources", "native_scores_S0", "native_order_S0", "feasible"} {
				if !reflect.DeepEqual(ref[key], got[key]) {
					t.Fatalf("%s/%s first-arrival %s differs from NATIVE_OFF", tr.Name, arm, key)
				}
			}
		}
		normalized["traces"].(map[string]any)[tr.Name] = perTrace
	}
	raw, err := json.Marshal(normalized)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(dir, "normalized.json"), raw, 0600); err != nil {
		t.Fatal(err)
	}
	nativeSave(t, dir, "timings.json", timings)
	t.Log("normalized_sha256 ", m4bHash(raw))
}

// G3: every arrival needs its own certified artifact and one-use grant. Stale, reused,
// mutated, foreign-Session and same-batch attempts must fall back to neutral (0, nil).
func TestM4BPerArrivalFreshnessAndTrust(t *testing.T) {
	test_utils.InitTestingInfrastructure()
	dir := filepath.Join(m4bEvidenceDir(t), "freshness")
	tr := m4bTraces[3] // fragmentation
	e := m4bFixture(t, tr, "native_dynamic_pgfp")
	receipt := map[string]any{}

	first := m4bProduce(t, e, 1, 0, filepath.Join(dir, "arrival0"))
	if !e.plugin.begin(first.raw, first.grant, m4bLambda) {
		t.Fatal("arrival 0 fresh grant")
	}
	allocate.New().Execute(e.f.session)
	e.plugin.endPhase()
	// The next arrival: the old grant is spent, and an unspent copy still fails the state gate.
	e.arrive(t, 1, tr.Arrivals[1])
	reuse := *first.grant
	if e.plugin.begin(first.raw, &reuse, m4bLambda) {
		t.Fatal("spent grant reactivated")
	}
	unspent := *first.grant
	unspent.used = false
	unspent.phase = 2
	current, err := bridgeProjection(t, e.f)
	if err != nil || bridgeHash(current) == first.grant.projectionHash {
		t.Fatal("state evolution not measured", err)
	}
	if e.plugin.begin(first.raw, &unspent, m4bLambda) {
		t.Fatal("stale projection accepted with unspent copy")
	}
	receipt["spent_grant_rejected"], receipt["stale_projection_rejected"] = true, true
	// Neutral fallback: while no phase is active the callback returns exactly S0 and counts it.
	neutral, _, _ := m4bScores(e, t)
	if neutral["node0"] != 1019 || neutral["node1"] != 1010 || e.plugin.inactive.Load() == 0 {
		t.Fatal("neutral fallback score", neutral)
	}
	receipt["neutral_fallback_scores"] = neutral
	// Resource mutation between production and activation is detected (projection gate), then a
	// fresh artifact for the unchanged-again state activates and drives the native flip.
	mutated := m4bProduce(t, e, 2, 0, filepath.Join(dir, "arrival1-mutated"))
	e.f.session.ClusterInfo.Nodes["node0"].IdleVector[resource_info.CPUIndex] -= 1000
	if e.plugin.begin(mutated.raw, mutated.grant, m4bLambda) {
		t.Fatal("resource-mutated state accepted")
	}
	e.f.session.ClusterInfo.Nodes["node0"].IdleVector[resource_info.CPUIndex] += 1000
	receipt["resource_mutation_rejected"] = true
	// A grant minted for another Session cannot activate this one (re-session attempt).
	other := m4bFixture(t, tr, "native_dynamic_pgfp_foreign")
	foreign := m4bProduce(t, other, 1, 0, filepath.Join(dir, "foreign"))
	if e.plugin.begin(foreign.raw, foreign.grant, m4bLambda) {
		t.Fatal("foreign-Session grant accepted")
	}
	receipt["foreign_session_rejected"] = true
	fresh := m4bProduce(t, e, 2, 0, filepath.Join(dir, "arrival1-fresh"))
	if !e.plugin.begin(fresh.raw, fresh.grant, m4bLambda) {
		t.Fatal("fresh arrival-1 grant")
	}
	scores, order, _ := m4bScores(e, t)
	allocate.New().Execute(e.f.session)
	e.plugin.endPhase()
	if order[0] != "node1" || scores["node0"] >= scores["node1"] || e.f.session.ClusterInfo.PodGroupInfos[e.f.job.UID].GetAllPodsMap()[e.f.task.UID].NodeName != "node1" {
		t.Fatal("state-dependent second decision not observed", scores, order)
	}
	if m4bHash(first.artifact) == m4bHash(fresh.artifact) {
		t.Fatal("second arrival reused the first artifact")
	}
	receipt["second_arrival_scores"], receipt["second_arrival_order"] = scores, order
	receipt["artifact_digests_differ"] = true
	// Same-batch attempt: P1 refuses two simultaneous Pending targets, so the batch-start
	// scoring design cannot be expressed through the bridge at all.
	e.arrive(t, 2, 2)
	e.arrive(t, 3, 2)
	if _, err = bridgeProjection(t, e.f); err == nil {
		t.Fatal("two Pending targets accepted into one projection")
	}
	receipt["same_batch_two_pending_refused"] = err.Error()
	nativeSave(t, dir, "receipt.json", receipt)
}
