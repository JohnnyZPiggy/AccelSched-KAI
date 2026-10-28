package main

import (
	"math"
	"strings"
	"sync"

	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/node_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/pod_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/framework"
)

const adviceProbeName = "accelsched-fixed-advice-probe"
const adviceLambdaLimit = 1e6

type adviceValue struct {
	Task, Node string
	Advice     float64
}
type adviceSnapshot struct {
	Version int
	Session string
	Lambda  float64
	Entries []adviceValue
}
type adviceRecord struct {
	metadata nodeProbeRecord
	active   bool
	lambda   float64
	values   map[nodeProbeKey]float64 // Owned copy, immutable before callback publication.
	returned map[nodeProbeKey]float64 // Actual returned scalars, protected by metadata.mu.
}

func finiteAdvice(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }

func makeAdviceRecord(s *framework.Session, snapshot *adviceSnapshot) *adviceRecord {
	r := &adviceRecord{metadata: nodeProbeRecord{counts: make(map[nodeProbeKey]uint64)}, returned: make(map[nodeProbeKey]float64)}
	if validIdentity(s.ID) {
		r.metadata.session = strings.Clone(s.ID)
	}
	valid := s.ClusterInfo != nil && snapshot != nil && snapshot.Version == 1 && validIdentity(snapshot.Session) &&
		snapshot.Session == s.ID && finiteAdvice(snapshot.Lambda) && snapshot.Lambda >= 0 &&
		snapshot.Lambda <= adviceLambdaLimit && len(snapshot.Entries) > 0 && len(snapshot.Entries) <= nodeProbeCapacity
	values := make(map[nodeProbeKey]float64)
	if valid {
		for _, e := range snapshot.Entries {
			key := nodeProbeKey{e.Task, e.Node}
			_, duplicate := values[key]
			knownTask := false
			for _, job := range s.ClusterInfo.PodGroupInfos {
				for id := range job.GetAllPodsMap() {
					if string(id) == e.Task {
						knownTask = true
					}
				}
			}
			if !validIdentity(e.Task) || !validIdentity(e.Node) || duplicate || !knownTask ||
				s.ClusterInfo.Nodes[e.Node] == nil || !finiteAdvice(e.Advice) || math.Abs(e.Advice) > 1 ||
				!finiteAdvice(snapshot.Lambda*e.Advice) {
				valid = false
				break
			}
			values[nodeProbeKey{strings.Clone(e.Task), strings.Clone(e.Node)}] = e.Advice
		}
	}
	if valid {
		r.active, r.lambda, r.values = true, snapshot.Lambda, values
	} else {
		// Reject the entire snapshot; never retain a partially accepted Advice map.
		r.metadata.diagnostics = 1
	}
	return r
}

func (r *adviceRecord) score(task *pod_info.PodInfo, node *node_info.NodeInfo) (float64, error) {
	m := &r.metadata
	m.gate.RLock()
	defer m.gate.RUnlock()
	if m.closed {
		return 0, nil
	}
	n := m.active.Add(1)
	defer m.active.Add(-1)
	for peak := m.peak.Load(); n > peak; peak = m.peak.Load() {
		if m.peak.CompareAndSwap(peak, n) {
			break
		}
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.total++
	if task == nil || node == nil || !validIdentity(string(task.UID)) || !validIdentity(node.Name) {
		m.diagnostics++
		return 0, nil
	}
	key := nodeProbeKey{string(task.UID), node.Name}
	if _, exists := m.counts[key]; !exists {
		if len(m.counts) == nodeProbeCapacity {
			m.diagnostics++
			return 0, nil
		}
		key = nodeProbeKey{strings.Clone(key.Task), strings.Clone(key.Node)}
	}
	m.counts[key]++
	a, known := r.values[key]
	score := float64(0)
	if !r.active || !known {
		m.diagnostics++
	} else if r.lambda != 0 {
		score = r.lambda * a
	}
	r.returned[key] = score
	// |increment| <= 1e6: too small to overflow any finite binary64 aggregate.
	// Existing native errors/nonfinite aggregates are never intercepted or masked.
	return score, nil
}

func (r *adviceRecord) audit() nodeProbeAudit {
	a := r.metadata.audit()
	r.metadata.mu.Lock()
	defer r.metadata.mu.Unlock()
	for i := range a.Entries {
		a.Entries[i].Score = r.returned[nodeProbeKey{a.Entries[i].Task, a.Entries[i].Node}]
	}
	return a
}

type adviceScoringProbe struct {
	mu       sync.Mutex
	prepared bool
	pending  *adviceSnapshot
	sessions map[*framework.Session]*adviceRecord
}

var _ framework.Plugin = (*adviceScoringProbe)(nil)
var _ framework.PluginBuilder = newAdviceScoringProbe

// No registration/init/production wiring. Only tests prepare the one-shot snapshot.
func newAdviceScoringProbe(_ framework.PluginArguments) framework.Plugin {
	return &adviceScoringProbe{}
}
func (*adviceScoringProbe) Name() string { return adviceProbeName }
func (p *adviceScoringProbe) prepare(s adviceSnapshot) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.prepared || len(p.sessions) != 0 {
		return false
	}
	p.prepared = true
	// Bound copies even for malformed input. Open records a neutral diagnostic.
	if len(s.Entries) > nodeProbeCapacity || !validIdentity(s.Session) {
		return false
	}
	s.Session = strings.Clone(s.Session)
	entries := make([]adviceValue, len(s.Entries))
	for i, e := range s.Entries {
		if !validIdentity(e.Task) || !validIdentity(e.Node) {
			return false
		}
		entries[i] = adviceValue{strings.Clone(e.Task), strings.Clone(e.Node), e.Advice}
	}
	s.Entries = entries
	p.pending = &s
	return true
}
func (p *adviceScoringProbe) OnSessionOpen(s *framework.Session) {
	if s == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.sessions == nil {
		p.sessions = make(map[*framework.Session]*adviceRecord)
	}
	if p.sessions[s] != nil {
		return // Idempotent open cannot attach a second callback.
	}
	r := makeAdviceRecord(s, p.pending)
	p.pending = nil // A second Session, including one with the same label, is neutral.
	p.prepared = true
	p.sessions[s] = r
	s.AddNodeOrderFn(r.score)
}
func (p *adviceScoringProbe) OnSessionClose(s *framework.Session) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if r := p.sessions[s]; r != nil {
		r.metadata.close()
		delete(p.sessions, s)
	}
}
func (p *adviceScoringProbe) record(s *framework.Session) *adviceRecord {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.sessions[s]
}
