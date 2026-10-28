package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"strconv"
	"sync"
	"sync/atomic"

	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/node_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/pod_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/framework"
)

const artifactBridgeName = "accelsched-trusted-artifact-test-bridge"

type bridgeNode struct {
	Name string `json:"name"`
	UID  string `json:"uid"`
}
type bridgePair struct {
	Task string `json:"task_uid"`
	Node string `json:"node"`
	Bits string `json:"score_bits"`
}
type bridgeEnvelope struct {
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
	Nodes        []bridgeNode `json:"nodes"`
	Pairs        []bridgePair `json:"pairs"`
}

// No production constructor/deserializer exists for this capability. Only the
// controlled test harness may mint one after the pinned Python process succeeds.
// Trust assumes the test process/workspace is controlled; this is NOT a signature.
type bridgeGrant struct {
	owner                                      *framework.Session
	task                                       *pod_info.PodInfo
	nodes                                      map[string]*node_info.NodeInfo
	envelopeHash, projectionHash, producerHash string
	phase                                      int
	project                                    func(*framework.Session) ([]byte, error)
	used                                       bool // Accessed under the owning bridge mutex; no concurrent handoff.
}

func bridgeHash(raw []byte) string { s := sha256.Sum256(raw); return hex.EncodeToString(s[:]) }

func bridgeUnique(d *json.Decoder) error {
	t, err := d.Token()
	if err != nil {
		return err
	}
	if delim, ok := t.(json.Delim); ok {
		switch delim {
		case '{':
			seen := map[string]bool{}
			for d.More() {
				k, e := d.Token()
				if e != nil {
					return e
				}
				s, ok := k.(string)
				if !ok || seen[s] {
					return fmt.Errorf("duplicate/invalid key")
				}
				seen[s] = true
				if e = bridgeUnique(d); e != nil {
					return e
				}
			}
		case '[':
			for d.More() {
				if e := bridgeUnique(d); e != nil {
					return e
				}
			}
		default:
			return fmt.Errorf("unexpected delimiter")
		}
		_, err = d.Token()
	}
	return err
}
func decodeBridge(raw []byte, grant *bridgeGrant) (bridgeEnvelope, error) {
	var v bridgeEnvelope
	if grant == nil || len(raw) == 0 || len(raw) > 65536 || !bytes.Equal(bytes.TrimSpace(raw), raw) || bridgeHash(raw) != grant.envelopeHash {
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
	if v.Version != "M3B-1" || v.Producer != "SAE_CONTROLLED_OFFLINE_TEST_HARNESS" || v.ProducerHash != grant.producerHash || v.Projection != grant.projectionHash || v.Phase != grant.phase || v.Session != grant.owner.ID || v.Task != string(grant.task.UID) {
		return v, fmt.Errorf("provenance/version/phase mismatch")
	}
	artifact, err := base64.StdEncoding.Strict().DecodeString(v.Artifact)
	if err != nil || base64.StdEncoding.EncodeToString(artifact) != v.Artifact || bridgeHash(artifact) != v.ArtifactHash {
		return v, fmt.Errorf("artifact byte digest mismatch")
	}
	// Python already recertified this exact artifact. Check the mapping/score
	// handoff again on ORIGINAL bytes, never assume Go re-encoding equals Python.
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
	if json.Unmarshal(artifact, &a) != nil || a.Version != "M3A-1" || a.Status != "CERTIFIED_OFFLINE_ADVICE" || a.Prediction != v.Prediction || len(a.Pairs) != 2 || len(v.Pairs) != 2 || len(v.Nodes) != 2 {
		return v, fmt.Errorf("artifact coverage/status")
	}
	for i, n := range v.Nodes {
		name := fmt.Sprintf("node%d", i)
		actual := grant.nodes[name]
		if actual == nil || n.Name != name || actual.Node == nil || n.UID != string(actual.Node.UID) {
			return v, fmt.Errorf("node identity")
		}
	}
	for i, p := range v.Pairs {
		bits, e := strconv.ParseUint(p.Bits, 16, 64)
		score := math.Float64frombits(bits)
		if e != nil || len(p.Bits) != 16 || fmt.Sprintf("%016x", bits) != p.Bits || !finiteAdvice(score) || score < -1 || score > 1 || p.Task != v.Task || p.Node != v.Nodes[i].Name || a.Pairs[i].Task != p.Task || a.Pairs[i].Node != p.Node || math.Float64bits(a.Pairs[i].Score) != bits {
			return v, fmt.Errorf("pair identity/score mismatch")
		}
	}
	return v, nil
}

type artifactBridgeProbe struct {
	mu                  sync.RWMutex
	owner               *framework.Session
	record              *adviceRecord
	phaseActive, closed bool
	target              *pod_info.PodInfo
	nodes               map[string]*node_info.NodeInfo
	inactive            atomic.Uint64
	lastPhase           int
	phaseRecords        []*adviceRecord
}

var _ framework.Plugin = (*artifactBridgeProbe)(nil)
var _ framework.PluginBuilder = newArtifactBridge

func newArtifactBridge(_ framework.PluginArguments) framework.Plugin { return &artifactBridgeProbe{} }
func (*artifactBridgeProbe) Name() string                            { return artifactBridgeName }
func (p *artifactBridgeProbe) OnSessionOpen(s *framework.Session) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if s == nil || p.owner != nil || p.closed {
		return
	}
	p.owner = s
	// Reuse the frozen M2 record/counters/fallback. No active snapshot at open.
	p.record = makeAdviceRecord(s, nil)
	s.AddNodeOrderFn(p.score)
}
func (p *artifactBridgeProbe) score(t *pod_info.PodInfo, n *node_info.NodeInfo) (float64, error) {
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

// The caller holds EXCLUSIVE ownership of the fake Session: capture/validate P
// immediately before a single target's native scoring or Allocate, then endPhase.
// No concurrent Session mutation is supported; native scoring workers only read.
func (p *artifactBridgeProbe) begin(raw []byte, g *bridgeGrant, lambda float64) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.phaseActive = false
	if p.closed || g == nil || g.used || g.owner != p.owner || g.project == nil || g.phase <= p.lastPhase || g.phase > 3 || !finiteAdvice(lambda) || lambda < 0 || lambda > adviceLambdaLimit {
		return false
	}
	g.used = true
	currentProjection, err := g.project(p.owner)
	if err != nil || bridgeHash(currentProjection) != g.projectionHash {
		return false
	}
	v, err := decodeBridge(raw, g)
	if err != nil {
		return false
	}
	if g.owner.ClusterInfo == nil {
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
	// M2 validated/copy-owned snapshot logic; each phase gets a new sealed record.
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
		p.phaseRecords = append(p.phaseRecords, p.record)
	}
	return p.phaseActive
}
func (p *artifactBridgeProbe) endPhase() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.phaseActive = false
	if p.record != nil {
		p.record.metadata.close()
	}
}
func (p *artifactBridgeProbe) OnSessionClose(s *framework.Session) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.owner == s {
		p.phaseActive = false
		p.closed = true
		if p.record != nil {
			p.record.metadata.close()
		}
		p.target = nil
		p.nodes = nil
	}
}
