package main

// M5 Stage A: strict P2-WholeGPU-1 projection over genuine official-KAI whole-GPU Fake Sessions.
// Nodes and Pods are built with the pinned fake builders (nodes_fake GPUs, jobs_fake whole-GPU
// requests); no CPU-only task is relabelled and no ResourceVector is mutated to fake a GPU.
// P1 (CPU_ONLY, bridgeProjection) stays unchanged. The unmodified M3-B artifactBridgeProbe is
// reused: its grant carries a P2 projection closure and the controlled Python P2 producer.

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
	"strconv"
	"testing"
	"time"

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
	commonconstants "github.com/kai-scheduler/api/constants"
	"go.uber.org/mock/gomock"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

const (
	p2Version   = "P2-WholeGPU-1"
	p2Semantics = "AUDITED_WHOLE_GPU_SINGLE_PENDING"
	p2Family    = "WHOLE_GPU_TEST"
	m5MemGiB    = 64
)

type m5Node struct {
	GPUs  int     `json:"gpus"`
	Cores float64 `json:"cores"`
}
type m5Job struct {
	GPUs  int `json:"gpus"`
	Cores int `json:"cores"`
}

type m5Env struct {
	f      nativeFixture
	mock   *cache.MockCache
	binds  *[]json.RawMessage
	plugin *artifactBridgeProbe
	nodes  [2]m5Node
	jobs   int
}

// m5Session builds one official Fake Session from a topology; the first Pending task is native-job-0.
func m5Session(t *testing.T, id string, md test_utils.TestTopologyBasic) *m5Env {
	t.Helper()
	ctrl := gomock.NewController(t)
	s := test_utils.BuildSession(md, ctrl)
	s.ID = id
	mock, ok := s.Cache.(*cache.MockCache)
	if !ok {
		t.Fatal("not official cache mock")
	}
	job := s.ClusterInfo.PodGroupInfos[common_info.NewPodGroupID("", "native-job")]
	if job == nil {
		t.Fatal("missing job")
	}
	task := job.GetAllPodsMap()[common_info.PodID("native-job-0")]
	if task == nil {
		t.Fatal("missing task")
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
	e := &m5Env{f: nativeFixture{s, job, task}, mock: mock, binds: binds, plugin: newArtifactBridge(nil).(*artifactBridgeProbe), jobs: 1}
	e.plugin.OnSessionOpen(s)
	t.Cleanup(func() { e.plugin.OnSessionClose(s) })
	return e
}

func m5Topology(name string, nodes [2]m5Node, first m5Job, mutate func(*test_utils.TestTopologyBasic)) test_utils.TestTopologyBasic {
	md := test_utils.TestTopologyBasic{
		Name: "M5-" + name,
		Jobs: []*jobs_fake.TestJobBasic{{Name: "native-job", QueueName: "queue0", RequiredGPUsPerTask: float64(first.GPUs), RequiredCPUsPerTask: float64(first.Cores), RequiredMemoryPerTask: nativeGiB, Tasks: []*tasks_fake.TestTaskBasic{{State: pod_status.Pending}}}},
		Nodes: map[string]nodes_fake.TestNodeBasic{
			"node0": {GPUs: nodes[0].GPUs, CPUMillis: nodes[0].Cores, CPUMemory: m5MemGiB * nativeGiB},
			"node1": {GPUs: nodes[1].GPUs, CPUMillis: nodes[1].Cores, CPUMemory: m5MemGiB * nativeGiB},
		},
		Queues: []test_utils.TestQueueBasic{{Name: "queue0"}},
		Mocks:  &test_utils.TestMock{CacheRequirements: &test_utils.CacheMocking{}},
	}
	if mutate != nil {
		mutate(&md)
	}
	return md
}

// arrive adds a genuine whole-GPU Pod (copy of the first Pod's request list with the new GPU/CPU quantities).
func (e *m5Env) arrive(t *testing.T, index int, j m5Job) {
	t.Helper()
	zero := e.f.job.GetAllPodsMap()[common_info.PodID("native-job-0")]
	req := zero.Pod.Spec.Containers[0].Resources.Requests.DeepCopy()
	req[v1.ResourceCPU] = *resource.NewQuantity(int64(j.Cores), resource.DecimalSI)
	req[resource_info.GPUResourceName] = *resource.NewQuantity(int64(j.GPUs), resource.DecimalSI)
	pod := tasks_fake.BuildPod(fmt.Sprintf("native-job-%d", index), "", &tasks_fake.TestTaskBasic{State: pod_status.Pending}, v1.PodPending, req, "", 0, nil, "native-job")
	next := pod_info.NewTaskInfo(pod, zero.VectorMap)
	e.f.job.AddTaskInfo(next)
	if next.Status != pod_status.Pending || next.NodeName != "" {
		t.Fatal("arrival")
	}
	e.f.task = next
}

// ---- P2 native vector helpers --------------------------------------------------------------

// p2Raw returns the native [CPU milli, memory bytes, whole GPUs, pod slots] as exact integers.
func p2Raw(v resource_info.ResourceVector) ([]int64, error) {
	if len(v) != 4 {
		return nil, fmt.Errorf("extra native resource coordinate: vector length %d", len(v))
	}
	out := []int64{}
	for _, i := range []int{resource_info.CPUIndex, resource_info.MemoryIndex, resource_info.GPUIndex, resource_info.PodsIndex} {
		x := v.Get(i)
		if math.IsNaN(x) || math.IsInf(x, 0) || x < 0 || x != math.Trunc(x) || x > 1e15 {
			return nil, fmt.Errorf("noninteger/negative/nonfinite native coordinate %d", i)
		}
		out = append(out, int64(x))
	}
	return out, nil
}

// p2Model converts raw native to [whole GPU, CPU cores, RAM GiB] with no rounding at all.
func p2Model(raw []int64) ([]int64, error) {
	if raw[0]%1000 != 0 || raw[1]%nativeGiB != 0 {
		return nil, fmt.Errorf("CPU milli/1000 or RAM bytes/2^30 not exact")
	}
	return []int64{raw[2], raw[0] / 1000, raw[1] / nativeGiB}, nil
}

func p2Shape(model []int64) string { return fmt.Sprintf("G%dC%dM%d", model[0], model[1], model[2]) }

type p2Elig struct {
	Native      bool   `json:"native"`
	Model       bool   `json:"model"`
	FittingGPUs int    `json:"fitting_gpus"`
	PreErr      string `json:"-"`
	PredErr     string `json:"-"`
}

// p2EligibilityFor calls the pinned native PrePredicateFn/PredicateFn/FittingNode/FittingGPUs and the
// independent model rule (idle >= request in all three model coordinates and a free pod slot).
func p2EligibilityFor(e *m5Env) (map[string]p2Elig, error) {
	s, task, job := e.f.session, e.f.task, e.f.job
	out := map[string]p2Elig{}
	pre := s.PrePredicateFn(task, job)
	reqRaw, err := p2Raw(task.ResReqVector)
	if err != nil {
		return nil, err
	}
	req, err := p2Model(reqRaw)
	if err != nil {
		return nil, err
	}
	for _, name := range m4bBoth {
		n := s.ClusterInfo.Nodes[name]
		idleRaw, err := p2Raw(n.IdleVector)
		if err != nil {
			return nil, err
		}
		idle, err := p2Model(idleRaw)
		if err != nil {
			return nil, err
		}
		model := idleRaw[3] >= 1
		for k := 0; k < 3; k++ {
			model = model && idle[k] >= req[k]
		}
		el := p2Elig{Model: model, FittingGPUs: len(s.FittingGPUs(n, task))}
		if pre != nil {
			el.PreErr = pre.Error()
		}
		if perr := s.PredicateFn(task, job, n); perr != nil {
			el.PredErr = perr.Error()
		}
		el.Native = pre == nil && el.PredErr == "" && s.FittingNode(task, n, false)
		out[name] = el
	}
	return out, nil
}

func p2TaskRaw(p *pod_info.PodInfo) ([]int64, error) { return p2Raw(p.ResReqVector) }

// p2Projection is the strict, versioned P2-WholeGPU-1 projection of the CURRENT native Session.
func p2Projection(t *testing.T, e *m5Env) ([]byte, error) {
	t.Helper()
	f := e.f
	s := f.session
	if resource_info.CPUIndex != 0 || resource_info.MemoryIndex != 1 || resource_info.GPUIndex != 2 || resource_info.PodsIndex != 3 {
		return nil, fmt.Errorf("pinned native index map changed")
	}
	if s == nil || s.ClusterInfo == nil || len(s.ClusterInfo.PodGroupInfos) != 1 || len(s.ClusterInfo.Nodes) != 2 || len(s.ClusterInfo.Topologies) != 0 || len(s.ClusterInfo.ResourceClaims) != 0 || len(s.ClusterInfo.ResourceSlices) != 0 {
		return nil, fmt.Errorf("unsupported native membership/features")
	}
	job := s.ClusterInfo.PodGroupInfos[common_info.NewPodGroupID("", "native-job")]
	if job == nil || string(job.Queue) != "queue0" || job.Priority != 0 || len(job.GetAllPodsMap()) > 4 {
		return nil, fmt.Errorf("unsupported job")
	}
	for _, set := range job.PodSets {
		if set.GetMinAvailable() != 1 {
			return nil, fmt.Errorf("gang unsupported")
		}
	}
	nodes := []any{}
	for _, name := range m4bBoth {
		n := s.ClusterInfo.Nodes[name]
		if n == nil || n.Name != name || n.Node == nil || n.Node.Name != name || n.NumaTopology != nil || n.NodeResourceTopology != nil || n.HasDRAGPUs || len(n.LegacyMIGTasks) != 0 || len(n.Node.Spec.Taints) != 0 || len(n.UsedSharedGPUsMemory) != 0 || n.MaxTaskNum < 4 {
			return nil, fmt.Errorf("unsupported node semantics (NUMA/DRA/MIG/taint/shared GPU/pod slots)")
		}
		raw := map[string][]int64{}
		// Fixed order: the first failing coordinate must be reported deterministically.
		for _, kv := range []struct {
			key string
			vec resource_info.ResourceVector
		}{{"allocatable", n.AllocatableVector}, {"idle", n.IdleVector}, {"used", n.UsedVector}, {"releasing", n.ReleasingVector}} {
			x, err := p2Raw(kv.vec)
			if err != nil {
				return nil, fmt.Errorf("%s %s: %w", name, kv.key, err)
			}
			raw[kv.key] = x
		}
		for _, x := range raw["releasing"] {
			if x != 0 {
				return nil, fmt.Errorf("releasing/preemption unsupported")
			}
		}
		for k := 0; k < 4; k++ {
			if raw["allocatable"][k] != raw["idle"][k]+raw["used"][k] {
				return nil, fmt.Errorf("native idle+used != allocatable at coordinate %d", k)
			}
		}
		model := map[string][]int64{}
		for _, key := range []string{"allocatable", "idle", "used"} {
			m, err := p2Model(raw[key])
			if err != nil {
				return nil, fmt.Errorf("%s %s: %w", name, key, err)
			}
			model[key] = m
		}
		if raw["idle"][3] < 4 {
			return nil, fmt.Errorf("pod-slot bottleneck")
		}
		labels := map[string]string{commonconstants.GpuCountLabel: strconv.FormatInt(raw["allocatable"][2], 10), node_info.GpuMemoryLabel: "100", "node-role.kubernetes.io/mig-enabled": "false", commonconstants.MigStrategyLabel: "", tasks_fake.NodeAffinityKey: name}
		if !reflect.DeepEqual(n.Node.Labels, labels) {
			return nil, fmt.Errorf("unknown node labels/family or GPU count label mismatch")
		}
		members := []string{}
		for uid := range n.PodInfos {
			members = append(members, string(uid))
		}
		sort.Strings(members)
		nodes = append(nodes, map[string]any{"name": name, "uid": string(n.Node.UID), "shape": p2Shape(model["allocatable"]), "family": p2Family,
			"allocatable": model["allocatable"], "idle": model["idle"], "used": model["used"], "releasing": []int64{0, 0, 0},
			"native_raw": raw, "members": members,
			"native": map[string]any{"spec": n.Node.Spec, "labels": n.Node.Labels, "annotations": n.Node.Annotations, "status": n.Node.Status}})
	}
	ids := []string{}
	for id := range job.GetAllPodsMap() {
		ids = append(ids, string(id))
	}
	sort.Strings(ids)
	tasks, pending := []any{}, 0
	for _, id := range ids {
		p := job.GetAllPodsMap()[common_info.PodID(id)]
		if p == nil || p.Pod == nil || string(p.Pod.UID) != id || p.ResourceRequestType != pod_info.RequestTypeRegular || !p.NUMAPlacement.IsEmpty() ||
			p.GpuRequirement.GpuMemory() != 0 || len(p.GpuRequirement.MigResources()) != 0 || len(p.GpuRequirement.DraGpuCounts()) != 0 || p.GpuRequirement.IsFractionalRequest() ||
			len(p.GPUGroupIDs()) != 0 || p.IsLegacyMIGtask || p.VectorMap.Len() != 4 {
			return nil, fmt.Errorf("unsupported task/GPU semantics: %s", id)
		}
		if p.Pod.Annotations[commonconstants.GpuFraction] != "" {
			return nil, fmt.Errorf("fractional GPU annotation unsupported")
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
			if k != v1.ResourceCPU && k != v1.ResourceMemory && k != resource_info.GPUResourceName && k != v1.ResourcePods {
				return nil, fmt.Errorf("extra resource %s", k)
			}
		}
		if pods := req[v1.ResourcePods]; pods.Value() != 1 || pods.MilliValue() != 1000 {
			return nil, fmt.Errorf("exactly one pod slot required")
		}
		if p.Pod.Spec.Containers[0].Name != "main" || p.Pod.Spec.SchedulerName != "kai-scheduler" {
			return nil, fmt.Errorf("unsupported container/predicate semantics")
		}
		gpus := req[resource_info.GPUResourceName]
		if gpus.MilliValue() != gpus.Value()*1000 || float64(gpus.Value()) != p.GpuRequirement.GPUs() || p.GpuRequirement.GPUs() != p.ResReqVector.Get(resource_info.GPUIndex) {
			return nil, fmt.Errorf("Pod GPU request, GpuRequirement and ResourceVector disagree")
		}
		if float64(req.Cpu().MilliValue()) != p.ResReqVector.Get(resource_info.CPUIndex) || float64(req.Memory().Value()) != p.ResReqVector.Get(resource_info.MemoryIndex) {
			return nil, fmt.Errorf("native request/Pod unit mismatch")
		}
		rr, err := p2TaskRaw(p)
		if err != nil {
			return nil, err
		}
		rm, err := p2Model(rr)
		if err != nil {
			return nil, err
		}
		ar, err := p2Raw(p.AcceptedResourceVector)
		if err != nil {
			return nil, err
		}
		am, err := p2Model([]int64{ar[0], ar[1], ar[2], 0})
		if err != nil {
			return nil, err
		}
		if p.Status == pod_status.Pending {
			pending++
			if p.NodeName != "" {
				return nil, fmt.Errorf("Pending node assignment")
			}
		}
		tasks = append(tasks, map[string]any{"uid": id, "pod_uid": string(p.Pod.UID), "status": p.Status.String(), "node": p.NodeName, "request": rm, "accepted": am, "native_request": rr,
			"native": map[string]any{"spec": p.Pod.Spec, "labels": p.Pod.Labels, "annotations": p.Pod.Annotations, "gpu": nativeGPU(&p.GpuRequirement), "job": p.Job}})
	}
	if pending != 1 {
		return nil, fmt.Errorf("one Pending target required")
	}
	elig, err := p2EligibilityFor(e)
	if err != nil {
		return nil, err
	}
	eligOut := map[string]any{}
	for name, el := range elig {
		if el.PreErr != "" {
			return nil, fmt.Errorf("native prepredicate: %s", el.PreErr)
		}
		if el.Native != el.Model {
			return nil, fmt.Errorf("native/model candidate disagreement at %s (native=%v model=%v pred=%q)", name, el.Native, el.Model, el.PredErr)
		}
		eligOut[name] = el
	}
	out := map[string]any{"version": p2Version, "session": s.ID, "semantics": p2Semantics, "units": []string{"whole GPUs", "integer CPU cores", "integer RAM GiB"},
		"index_map": map[string]int{"cpu": resource_info.CPUIndex, "memory": resource_info.MemoryIndex, "gpu": resource_info.GPUIndex, "pods": resource_info.PodsIndex},
		"nodes":     nodes, "tasks": tasks, "eligibility": eligOut, "domain": probeDomain(t, f), "config": s.Config, "hooks": probeHooks(t, s)}
	return json.Marshal(out)
}

// ---- controlled Python P2 producer ----------------------------------------------------------

var m5Sources = []string{"src/accelsched_kai/m5_producer.py", "src/accelsched_kai/bridge_envelope.py", "src/accelsched_kai/bounded_advice.py", "src/accelsched_kai/adapter.py", "src/accelsched_kai/oracle.py", "src/accelsched_kai/advisor.py", "src/accelsched_kai/schema.py"}

type m5Produced struct {
	raw, artifact, controls []byte
	grant                   *bridgeGrant
	envelope                string
	seconds                 float64
	stderr                  string
}

// m5Run runs the controlled producer over exact projection bytes after checking committed sources.
func m5Run(t *testing.T, praw []byte, phase int, alpha float64, forecast []m4bForecast, dir string) (m5Produced, string, error) {
	t.Helper()
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	root, _ := filepath.Abs("..")
	var producerHash string
	for _, name := range m5Sources {
		b, err := os.ReadFile(filepath.Join(root, name))
		if err != nil {
			t.Fatal(err)
		}
		b = bytes.ReplaceAll(b, []byte("\r\n"), []byte("\n"))
		expected, err := exec.Command("git", "-C", root, "show", "HEAD:"+name).Output()
		if err != nil || !bytes.Equal(b, expected) {
			t.Fatal("uncommitted/mismatched controlled producer source ", name, err)
		}
		if name == m5Sources[0] {
			producerHash = bridgeHash(b)
		}
	}
	request := nativeJSON(t, map[string]any{"projection_b64": base64.StdEncoding.EncodeToString(praw), "phase": phase, "alpha": alpha, "BF": m4bBF, "BP": m4bBP, "forecast": forecast})
	input := filepath.Join(dir, "request.json")
	if err := os.WriteFile(input, request, 0600); err != nil {
		t.Fatal(err)
	}
	python := os.Getenv("M3B_PYTHON")
	if python == "" {
		t.Fatal("M3B_PYTHON must identify the existing frozen environment")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	code := "import runpy,sys;sys.path.insert(0,sys.argv[1]);sys.argv=sys.argv[2:];runpy.run_module('accelsched_kai.m5_producer',run_name='__main__')"
	output := filepath.Join(dir, "producer")
	cmd := exec.CommandContext(ctx, python, "-I", "-B", "-c", code, filepath.Join(root, "src"), "m5_producer", input, output)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	start := time.Now()
	raw, runErr := cmd.Output()
	seconds := time.Since(start).Seconds()
	_ = os.WriteFile(filepath.Join(dir, "producer.stderr.txt"), stderr.Bytes(), 0600)
	if runErr != nil {
		return m5Produced{seconds: seconds, stderr: stderr.String()}, producerHash, runErr
	}
	file, err := os.ReadFile(filepath.Join(output, "envelope.json"))
	if err != nil || !bytes.Equal(file, raw) {
		t.Fatal("producer handoff mismatch")
	}
	artifact, _ := os.ReadFile(filepath.Join(output, "artifact.json"))
	controls, _ := os.ReadFile(filepath.Join(output, "controls.json"))
	return m5Produced{raw: raw, artifact: artifact, controls: controls, envelope: bridgeHash(raw), seconds: seconds}, producerHash, nil
}

// m5Produce is the only grant factory for M5. It binds exact bytes, committed sources, the owner
// Session pointer, the current P2 projection callback and the Task/Node identities.
func m5Produce(t *testing.T, e *m5Env, phase int, alpha float64, forecast []m4bForecast, dir string) m5Produced {
	t.Helper()
	praw, err := p2Projection(t, e)
	if err != nil {
		t.Fatal("P2 projection rejected a state that must be supported: ", err)
	}
	p, producerHash, runErr := m5Run(t, praw, phase, alpha, forecast, dir)
	if runErr != nil {
		t.Fatal("controlled P2 producer failed ", runErr, p.stderr)
	}
	hooks := []uintptr{}
	for _, fn := range e.f.session.PredicateFns {
		hooks = append(hooks, reflect.ValueOf(fn).Pointer())
	}
	grant := &bridgeGrant{owner: e.f.session, task: e.f.task, nodes: map[string]*node_info.NodeInfo{}, phase: phase, envelopeHash: bridgeHash(p.raw), projectionHash: bridgeHash(praw), producerHash: producerHash}
	for name, n := range e.f.session.ClusterInfo.Nodes {
		grant.nodes[name] = n
	}
	grant.project = func(s *framework.Session) ([]byte, error) {
		if s != e.f.session {
			return nil, fmt.Errorf("foreign Session")
		}
		now := []uintptr{}
		for _, fn := range s.PredicateFns {
			now = append(now, reflect.ValueOf(fn).Pointer())
		}
		if !reflect.DeepEqual(now, hooks) {
			return nil, fmt.Errorf("predicate implementation changed")
		}
		return p2Projection(t, e)
	}
	nativeSave(t, dir, "projection.json", json.RawMessage(praw))
	p.grant = grant
	return p
}
