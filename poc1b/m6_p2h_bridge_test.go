package main

// M6 P2H-1: NEW versioned test-only admission boundary for 2-to-4 heterogeneous whole-GPU nodes.
// The frozen M3-B receiver (artifactBridgeProbe/decodeBridge: exactly two pairs, phases 1..3) and the
// P1/P2 projections are NOT modified; this file adds a separate receiver (p2hBridge/decodeP2H) that
// reuses only the unchanged M2 advice record (makeAdviceRecord) and the strict JSON helpers.
// Explicit caps: 2..4 nodes, <=8 tasks, phases 1..8, one exact score pair per native candidate.
// Controlled local test harness; not a signature, not a production activation path.

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
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
	p2hVersion   = "P2H-1"
	p2hSemantics = "AUDITED_WHOLE_GPU_MULTI_NODE_SINGLE_PENDING"
	p2hMinNodes  = 2
	p2hMaxNodes  = 4
	p2hMaxTasks  = 8
	p2hMaxPhase  = 8
)

type m6Node struct {
	GPUs   int     `json:"gpus"`
	Cores  float64 `json:"cores"`
	MemGiB float64 `json:"mem_gib"`
}
type m6Job struct {
	GPUs  int `json:"gpus"`
	Cores int `json:"cores"`
	GiB   int `json:"gib"`
}

func m6Names(n int) []string {
	out := []string{}
	for i := 0; i < n; i++ {
		out = append(out, fmt.Sprintf("node%d", i))
	}
	return out
}

type m6Env struct {
	f      nativeFixture
	mock   *cache.MockCache
	binds  *[]json.RawMessage
	plugin *p2hBridge
	nodes  []m6Node
	names  []string
}

func m6Topology(name string, nodes []m6Node, first m6Job, mutate func(*test_utils.TestTopologyBasic)) test_utils.TestTopologyBasic {
	nm := map[string]nodes_fake.TestNodeBasic{}
	for i, n := range nodes {
		nm[fmt.Sprintf("node%d", i)] = nodes_fake.TestNodeBasic{GPUs: n.GPUs, CPUMillis: n.Cores, CPUMemory: n.MemGiB * nativeGiB}
	}
	md := test_utils.TestTopologyBasic{
		Name:   "M6-" + name,
		Jobs:   []*jobs_fake.TestJobBasic{{Name: "native-job", QueueName: "queue0", RequiredGPUsPerTask: float64(first.GPUs), RequiredCPUsPerTask: float64(first.Cores), RequiredMemoryPerTask: float64(first.GiB) * nativeGiB, Tasks: []*tasks_fake.TestTaskBasic{{State: pod_status.Pending}}}},
		Nodes:  nm,
		Queues: []test_utils.TestQueueBasic{{Name: "queue0"}},
		Mocks:  &test_utils.TestMock{CacheRequirements: &test_utils.CacheMocking{}},
	}
	if mutate != nil {
		mutate(&md)
	}
	return md
}

func m6Session(t *testing.T, id string, nodes []m6Node, md test_utils.TestTopologyBasic) *m6Env {
	t.Helper()
	ctrl := gomock.NewController(t)
	s := test_utils.BuildSession(md, ctrl)
	s.ID = id
	mock, ok := s.Cache.(*cache.MockCache)
	if !ok {
		t.Fatal("not official cache mock")
	}
	job := s.ClusterInfo.PodGroupInfos[common_info.NewPodGroupID("", "native-job")]
	task := job.GetAllPodsMap()[common_info.PodID("native-job-0")]
	if job == nil || task == nil {
		t.Fatal("missing job/task")
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
	e := &m6Env{f: nativeFixture{s, job, task}, mock: mock, binds: binds, plugin: newP2HBridge(), nodes: nodes, names: m6Names(len(nodes))}
	e.plugin.OnSessionOpen(s)
	t.Cleanup(func() { e.plugin.OnSessionClose(s) })
	return e
}

func (e *m6Env) arrive(t *testing.T, index int, j m6Job) {
	t.Helper()
	zero := e.f.job.GetAllPodsMap()[common_info.PodID("native-job-0")]
	req := zero.Pod.Spec.Containers[0].Resources.Requests.DeepCopy()
	req[v1.ResourceCPU] = *resource.NewQuantity(int64(j.Cores), resource.DecimalSI)
	req[v1.ResourceMemory] = *resource.NewQuantity(int64(j.GiB)*nativeGiB, resource.DecimalSI)
	req[resource_info.GPUResourceName] = *resource.NewQuantity(int64(j.GPUs), resource.DecimalSI)
	pod := tasks_fake.BuildPod(fmt.Sprintf("native-job-%d", index), "", &tasks_fake.TestTaskBasic{State: pod_status.Pending}, v1.PodPending, req, "", 0, nil, "native-job")
	next := pod_info.NewTaskInfo(pod, zero.VectorMap)
	e.f.job.AddTaskInfo(next)
	if next.Status != pod_status.Pending || next.NodeName != "" {
		t.Fatal("arrival")
	}
	e.f.task = next
}

// ---- receiver ---------------------------------------------------------------------------------

type p2hGrant struct {
	owner                                      *framework.Session
	task                                       *pod_info.PodInfo
	nodes                                      map[string]*node_info.NodeInfo
	names, candidates                          []string
	envelopeHash, projectionHash, producerHash string
	phase                                      int
	project                                    func(*framework.Session) ([]byte, error)
	used                                       bool // accessed only under the owning bridge mutex
}

type p2hEnvelope struct {
	Version      string       `json:"version"`
	Producer     string       `json:"producer"`
	ProducerHash string       `json:"producer_sha256"`
	ArtifactHash string       `json:"artifact_sha256"`
	Artifact     string       `json:"artifact_b64"`
	Prediction   string       `json:"prediction_sha256"`
	Projection   string       `json:"projection_sha256"`
	Session      string       `json:"session"`
	Phase        int          `json:"phase"`
	Task         string       `json:"task_uid"`
	Candidates   []string     `json:"candidates"`
	Nodes        []bridgeNode `json:"nodes"`
	Pairs        []bridgePair `json:"pairs"`
}

func decodeP2H(raw []byte, g *p2hGrant) (p2hEnvelope, error) {
	var v p2hEnvelope
	if g == nil || len(raw) == 0 || len(raw) > 131072 || !bytes.Equal(bytes.TrimSpace(raw), raw) || bridgeHash(raw) != g.envelopeHash {
		return v, fmt.Errorf("no controlled canonical handoff or byte mismatch")
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	d.UseNumber()
	if err := bridgeUnique(d); err != nil {
		return v, err
	}
	if _, err := d.Token(); err != io.EOF {
		return v, fmt.Errorf("trailing JSON")
	}
	d = json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	if err := d.Decode(&v); err != nil {
		return v, err
	}
	if err := bridgeStringFields(raw, []string{"version", "producer", "producer_sha256", "artifact_sha256", "artifact_b64", "prediction_sha256", "projection_sha256", "session", "task_uid"}, 13); err != nil {
		return v, err
	}
	var object map[string]json.RawMessage
	_ = json.Unmarshal(raw, &object)
	var candidates []json.RawMessage
	if json.Unmarshal(object["candidates"], &candidates) != nil {
		return v, fmt.Errorf("candidate array required")
	}
	for _, c := range candidates {
		if len(c) == 0 || c[0] != '"' {
			return v, fmt.Errorf("exact string candidate required")
		}
	}
	for key, names := range map[string][]string{"nodes": {"name", "uid"}, "pairs": {"task_uid", "node", "score_bits"}} {
		var children []json.RawMessage
		if json.Unmarshal(object[key], &children) != nil {
			return v, fmt.Errorf("array required")
		}
		for _, child := range children {
			if err := bridgeStringFields(child, names, len(names)); err != nil {
				return v, err
			}
		}
	}
	if v.Version != p2hVersion || v.Producer != "SAE_CONTROLLED_OFFLINE_TEST_HARNESS" || v.ProducerHash != g.producerHash || v.Projection != g.projectionHash ||
		v.Phase != g.phase || v.Phase < 1 || v.Phase > p2hMaxPhase || v.Session != g.owner.ID || v.Task != string(g.task.UID) {
		return v, fmt.Errorf("provenance/version/phase mismatch")
	}
	artifact, err := base64.StdEncoding.Strict().DecodeString(v.Artifact)
	if err != nil || base64.StdEncoding.EncodeToString(artifact) != v.Artifact || bridgeHash(artifact) != v.ArtifactHash {
		return v, fmt.Errorf("artifact byte digest mismatch")
	}
	var a struct {
		Version    string `json:"version"`
		Status     string `json:"status"`
		Prediction string `json:"prediction_sha256"`
		Pairs      []struct {
			Task  string  `json:"task_id"`
			Node  string  `json:"node_id"`
			Score float64 `json:"score"`
		} `json:"pairs"`
	}
	n := len(g.candidates)
	if json.Unmarshal(artifact, &a) != nil || a.Version != "M3A-1" || a.Status != "CERTIFIED_OFFLINE_ADVICE" || a.Prediction != v.Prediction ||
		n < 1 || n > p2hMaxNodes || len(a.Pairs) != n || len(v.Pairs) != n || len(v.Candidates) != n || len(v.Nodes) != len(g.names) {
		return v, fmt.Errorf("artifact coverage/status")
	}
	if !reflect.DeepEqual(v.Candidates, g.candidates) {
		return v, fmt.Errorf("candidate set/order differs from the native candidate list")
	}
	for i, nd := range v.Nodes {
		actual := g.nodes[g.names[i]]
		if nd.Name != g.names[i] || actual == nil || actual.Node == nil || nd.UID != string(actual.Node.UID) {
			return v, fmt.Errorf("node identity/order")
		}
	}
	for i, p := range v.Pairs {
		bits, e := strconv.ParseUint(p.Bits, 16, 64)
		score := math.Float64frombits(bits)
		if e != nil || len(p.Bits) != 16 || fmt.Sprintf("%016x", bits) != p.Bits || !finiteAdvice(score) || score < -1 || score > 1 ||
			p.Task != v.Task || p.Node != v.Candidates[i] || a.Pairs[i].Task != p.Task || a.Pairs[i].Node != p.Node || math.Float64bits(a.Pairs[i].Score) != bits {
			return v, fmt.Errorf("pair identity/score mismatch")
		}
	}
	return v, nil
}

type p2hBridge struct {
	mu                  sync.RWMutex
	owner               *framework.Session
	record              *adviceRecord
	phaseActive, closed bool
	target              *pod_info.PodInfo
	nodes               map[string]*node_info.NodeInfo
	inactive            atomic.Uint64
	lastPhase           int
}

func newP2HBridge() *p2hBridge { return &p2hBridge{} }

func (p *p2hBridge) OnSessionOpen(s *framework.Session) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if s == nil || p.owner != nil || p.closed {
		return
	}
	p.owner = s
	p.record = makeAdviceRecord(s, nil)
	s.AddNodeOrderFn(p.score)
}

func (p *p2hBridge) score(t *pod_info.PodInfo, n *node_info.NodeInfo) (float64, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	if p.closed {
		return 0, nil
	}
	if !p.phaseActive || t == nil || n == nil || t.Pod == nil || p.target == nil || t.UID != p.target.UID || t.Job != p.target.Job || t.Pod.UID != p.target.Pod.UID || p.nodes[n.Name] == nil || n.Node != p.nodes[n.Name].Node {
		p.inactive.Add(1)
		return 0, nil
	}
	return p.record.score(t, n)
}

// begin: the caller holds exclusive ownership of the Session for the single target's scoring+Allocate.
func (p *p2hBridge) begin(raw []byte, g *p2hGrant, lambda float64) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.phaseActive = false
	if p.closed || g == nil || g.used || g.owner != p.owner || g.project == nil || g.phase <= p.lastPhase || g.phase > p2hMaxPhase || !finiteAdvice(lambda) || lambda < 0 || lambda > adviceLambdaLimit {
		return false
	}
	g.used = true
	current, err := g.project(p.owner)
	if err != nil || bridgeHash(current) != g.projectionHash {
		return false
	}
	v, err := decodeP2H(raw, g)
	if err != nil || g.owner.ClusterInfo == nil || len(g.nodes) != len(g.names) {
		return false
	}
	for name, n := range g.nodes {
		if g.owner.ClusterInfo.Nodes[name] != n {
			return false
		}
	}
	found := false
	for _, j := range g.owner.ClusterInfo.PodGroupInfos {
		for _, t := range j.GetAllPodsMap() {
			if t == g.task {
				found = true
			}
		}
	}
	if !found {
		return false
	}
	snapshot := adviceSnapshot{Version: 1, Session: v.Session, Lambda: lambda}
	for _, pair := range v.Pairs {
		bits, _ := strconv.ParseUint(pair.Bits, 16, 64)
		snapshot.Entries = append(snapshot.Entries, adviceValue{pair.Task, pair.Node, math.Float64frombits(bits)})
	}
	if p.record != nil {
		p.record.metadata.close()
	}
	p.record = makeAdviceRecord(p.owner, &snapshot)
	p.target = g.task
	p.nodes = make(map[string]*node_info.NodeInfo)
	for name, n := range g.nodes {
		p.nodes[name] = n
	}
	p.phaseActive = p.record.active
	if p.phaseActive {
		p.lastPhase = g.phase
	}
	return p.phaseActive
}

func (p *p2hBridge) endPhase() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.phaseActive = false
	if p.record != nil {
		p.record.metadata.close()
	}
}

func (p *p2hBridge) OnSessionClose(s *framework.Session) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.owner == s {
		p.phaseActive, p.closed = false, true
		if p.record != nil {
			p.record.metadata.close()
		}
		p.target, p.nodes = nil, nil
	}
}

// ---- native helpers for N nodes ------------------------------------------------------------------

type p2hElig struct {
	Native      bool   `json:"native"`
	Model       bool   `json:"model"`
	FittingGPUs int    `json:"fitting_gpus"`
	PreErr      string `json:"-"`
	PredErr     string `json:"-"`
}

func p2hEligibility(e *m6Env) (map[string]p2hElig, []string, error) {
	s, task, job := e.f.session, e.f.task, e.f.job
	out, cands := map[string]p2hElig{}, []string{}
	pre := s.PrePredicateFn(task, job)
	reqRaw, err := p2Raw(task.ResReqVector)
	if err != nil {
		return nil, nil, err
	}
	req, err := p2Model(reqRaw)
	if err != nil {
		return nil, nil, err
	}
	for _, name := range e.names {
		n := s.ClusterInfo.Nodes[name]
		idleRaw, err := p2Raw(n.IdleVector)
		if err != nil {
			return nil, nil, err
		}
		idle, err := p2Model(idleRaw)
		if err != nil {
			return nil, nil, err
		}
		model := idleRaw[3] >= 1
		for k := 0; k < 3; k++ {
			model = model && idle[k] >= req[k]
		}
		el := p2hElig{Model: model, FittingGPUs: len(s.FittingGPUs(n, task))}
		if pre != nil {
			el.PreErr = pre.Error()
		}
		if perr := s.PredicateFn(task, job, n); perr != nil {
			el.PredErr = perr.Error()
		}
		el.Native = pre == nil && el.PredErr == "" && s.FittingNode(task, n, false)
		out[name] = el
		if el.Native {
			cands = append(cands, name)
		}
	}
	return out, cands, nil
}

// p2hProjection is the strict, versioned P2H-1 projection of the CURRENT native Session.
func p2hProjection(t *testing.T, e *m6Env) ([]byte, error) {
	t.Helper()
	f, s := e.f, e.f.session
	if resource_info.CPUIndex != 0 || resource_info.MemoryIndex != 1 || resource_info.GPUIndex != 2 || resource_info.PodsIndex != 3 {
		return nil, fmt.Errorf("pinned native index map changed")
	}
	if len(e.names) < p2hMinNodes || len(e.names) > p2hMaxNodes {
		return nil, fmt.Errorf("P2H supports 2..4 nodes")
	}
	if s == nil || s.ClusterInfo == nil || len(s.ClusterInfo.PodGroupInfos) != 1 || len(s.ClusterInfo.Nodes) != len(e.names) || len(s.ClusterInfo.Topologies) != 0 || len(s.ClusterInfo.ResourceClaims) != 0 || len(s.ClusterInfo.ResourceSlices) != 0 {
		return nil, fmt.Errorf("unsupported native membership/features")
	}
	job := s.ClusterInfo.PodGroupInfos[common_info.NewPodGroupID("", "native-job")]
	if job == nil || string(job.Queue) != "queue0" || job.Priority != 0 || len(job.GetAllPodsMap()) > p2hMaxTasks {
		return nil, fmt.Errorf("unsupported job")
	}
	for _, set := range job.PodSets {
		if set.GetMinAvailable() != 1 {
			return nil, fmt.Errorf("gang unsupported")
		}
	}
	nodes := []any{}
	for _, name := range e.names {
		n := s.ClusterInfo.Nodes[name]
		if n == nil || n.Name != name || n.Node == nil || n.Node.Name != name || n.NumaTopology != nil || n.NodeResourceTopology != nil || n.HasDRAGPUs || len(n.LegacyMIGTasks) != 0 || len(n.Node.Spec.Taints) != 0 || len(n.UsedSharedGPUsMemory) != 0 || n.MaxTaskNum < 4 {
			return nil, fmt.Errorf("unsupported node semantics (NUMA/DRA/MIG/taint/shared GPU/pod slots) at %s", name)
		}
		raw := map[string][]int64{}
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
		rr, err := p2Raw(p.ResReqVector)
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
	elig, cands, err := p2hEligibility(e)
	if err != nil {
		return nil, err
	}
	eligOut := map[string]any{}
	for _, name := range e.names {
		el := elig[name]
		if el.PreErr != "" {
			return nil, fmt.Errorf("native prepredicate: %s", el.PreErr)
		}
		if el.Native != el.Model {
			return nil, fmt.Errorf("native/model candidate disagreement at %s", name)
		}
		eligOut[name] = el
	}
	out := map[string]any{"version": p2hVersion, "session": s.ID, "semantics": p2hSemantics, "units": []string{"whole GPUs", "integer CPU cores", "integer RAM GiB"},
		"index_map":  map[string]int{"cpu": resource_info.CPUIndex, "memory": resource_info.MemoryIndex, "gpu": resource_info.GPUIndex, "pods": resource_info.PodsIndex},
		"bounds":     map[string]int{"max_nodes": p2hMaxNodes, "max_phase": p2hMaxPhase, "max_tasks": p2hMaxTasks},
		"node_names": e.names, "nodes": nodes, "tasks": tasks, "eligibility": eligOut, "candidates": cands,
		"domain": probeDomain(t, f), "config": s.Config, "hooks": probeHooks(t, s)}
	return json.Marshal(out)
}

// ---- controlled Python P2H producer ----------------------------------------------------------

var m6Sources = []string{"src/accelsched_kai/p2h_producer.py", "src/accelsched_kai/bounded_advice.py", "src/accelsched_kai/adapter.py", "src/accelsched_kai/oracle.py", "src/accelsched_kai/advisor.py", "src/accelsched_kai/schema.py"}

type m6Produced struct {
	raw, artifact, controls []byte
	grant                   *p2hGrant
	envelope                string
	seconds                 float64
	stderr                  string
}

func m6Run(t *testing.T, praw []byte, phase int, alpha float64, forecast []m4bForecast, dir string) (m6Produced, string, error) {
	t.Helper()
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	root, _ := filepath.Abs("..")
	var producerHash string
	for _, name := range m6Sources {
		b, err := os.ReadFile(filepath.Join(root, name))
		if err != nil {
			t.Fatal(err)
		}
		b = bytes.ReplaceAll(b, []byte("\r\n"), []byte("\n"))
		expected, err := exec.Command("git", "-C", root, "show", "HEAD:"+name).Output()
		if err != nil || !bytes.Equal(b, expected) {
			t.Fatal("uncommitted/mismatched controlled producer source ", name, err)
		}
		if name == m6Sources[0] {
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
	code := "import runpy,sys;sys.path.insert(0,sys.argv[1]);sys.argv=sys.argv[2:];runpy.run_module('accelsched_kai.p2h_producer',run_name='__main__')"
	output := filepath.Join(dir, "producer")
	cmd := exec.CommandContext(ctx, python, "-I", "-B", "-c", code, filepath.Join(root, "src"), "p2h_producer", input, output)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	start := time.Now()
	raw, runErr := cmd.Output()
	seconds := time.Since(start).Seconds()
	_ = os.WriteFile(filepath.Join(dir, "producer.stderr.txt"), stderr.Bytes(), 0600)
	if runErr != nil {
		return m6Produced{seconds: seconds, stderr: stderr.String()}, producerHash, runErr
	}
	file, err := os.ReadFile(filepath.Join(output, "envelope.json"))
	if err != nil || !bytes.Equal(file, raw) {
		t.Fatal("producer handoff mismatch")
	}
	artifact, _ := os.ReadFile(filepath.Join(output, "artifact.json"))
	controls, _ := os.ReadFile(filepath.Join(output, "controls.json"))
	return m6Produced{raw: raw, artifact: artifact, controls: controls, envelope: bridgeHash(raw), seconds: seconds}, producerHash, nil
}

// m6Produce is the only grant factory for M6: exact bytes, committed sources, owner Session pointer,
// current P2H projection callback, ordered Node identities and the Go-computed native candidate list.
func m6Produce(t *testing.T, e *m6Env, phase int, alpha float64, forecast []m4bForecast, dir string) m6Produced {
	t.Helper()
	praw, err := p2hProjection(t, e)
	if err != nil {
		t.Fatal("P2H projection rejected a state that must be supported: ", err)
	}
	p, producerHash, runErr := m6Run(t, praw, phase, alpha, forecast, dir)
	if runErr != nil {
		t.Fatal("controlled P2H producer failed ", runErr, p.stderr)
	}
	_, cands, _ := p2hEligibility(e)
	hooks := []uintptr{}
	for _, fn := range e.f.session.PredicateFns {
		hooks = append(hooks, reflect.ValueOf(fn).Pointer())
	}
	grant := &p2hGrant{owner: e.f.session, task: e.f.task, nodes: map[string]*node_info.NodeInfo{}, names: append([]string{}, e.names...), candidates: cands, phase: phase,
		envelopeHash: bridgeHash(p.raw), projectionHash: bridgeHash(praw), producerHash: producerHash}
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
		return p2hProjection(t, e)
	}
	nativeSave(t, dir, "projection.json", json.RawMessage(praw))
	p.grant = grant
	return p
}

// m6Resources captures every node's native [GPU, CPU milli, memory bytes] vectors plus pod slots.
func m6Resources(s *framework.Session, names []string) map[string]any {
	out := map[string]any{}
	for _, name := range names {
		n := s.ClusterInfo.Nodes[name]
		entry := map[string]any{}
		for key, vec := range map[string]resource_info.ResourceVector{"allocatable": n.AllocatableVector, "idle": n.IdleVector, "used": n.UsedVector, "releasing": n.ReleasingVector} {
			x := []int64{}
			for _, i := range []int{resource_info.GPUIndex, resource_info.CPUIndex, resource_info.MemoryIndex} {
				x = append(x, int64(vec.Get(i)))
			}
			entry[key] = x
			entry[key+"_pods"] = int64(vec.Get(resource_info.PodsIndex))
		}
		ids := []string{}
		for id := range n.PodInfos {
			ids = append(ids, string(id))
		}
		sort.Strings(ids)
		entry["members"] = ids
		out[name] = entry
	}
	return out
}

// m6Scores returns the original/active NodeOrderFn totals for the given nodes, the native
// OrderedNodesByTask order, and the wall-clock seconds spent in native scoring.
func m6Scores(e *m6Env, t *testing.T, names []string) (map[string]float64, []string, float64) {
	t.Helper()
	s := e.f.session
	nodes := []*node_info.NodeInfo{}
	for i := len(names) - 1; i >= 0; i-- { // deliberately reversed input so native ordering is observable
		nodes = append(nodes, s.ClusterInfo.Nodes[names[i]])
	}
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
