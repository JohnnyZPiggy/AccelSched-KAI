package main

import (
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/node_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/pod_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/framework"
)

const nodeProbeName = "accelsched-node-probe"
const nodeProbeCapacity = 256

type nodeProbeKey struct{ Task, Node string }
type nodeProbeEntry struct {
	Task  string  `json:"task_uid"`
	Node  string  `json:"node"`
	Calls uint64  `json:"calls"`
	Score float64 `json:"score"`
	Error *string `json:"error"`
}
type nodeProbeAudit struct {
	Session     string           `json:"session"`
	Closed      bool             `json:"closed"`
	Complete    bool             `json:"complete"`
	Total       uint64           `json:"total_calls"`
	Diagnostics uint64           `json:"diagnostics"`
	Peak        int64            `json:"observed_peak_callbacks"`
	Entries     []nodeProbeEntry `json:"entries"`
}

type nodeProbeRecord struct {
	gate               sync.RWMutex // A close drains admitted callbacks before sealing.
	mu                 sync.Mutex
	session            string
	closed             bool
	total, diagnostics uint64
	counts             map[nodeProbeKey]uint64
	active, peak       atomic.Int64
}

func (r *nodeProbeRecord) score(task *pod_info.PodInfo, node *node_info.NodeInfo) (float64, error) {
	r.gate.RLock()
	defer r.gate.RUnlock()
	if r.closed {
		return 0, nil
	}
	active := r.active.Add(1)
	defer r.active.Add(-1) // Runs before releasing gate; no late write after seal.
	for peak := r.peak.Load(); active > peak; peak = r.peak.Load() {
		if r.peak.CompareAndSwap(peak, active) {
			break
		}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.total++
	if task == nil || node == nil || !validIdentity(string(task.UID)) || !validIdentity(node.Name) {
		r.diagnostics++
		return 0, nil
	}
	key := nodeProbeKey{string(task.UID), node.Name}
	if count, ok := r.counts[key]; ok {
		r.counts[key] = count + 1
	} else if len(r.counts) < nodeProbeCapacity {
		r.counts[nodeProbeKey{strings.Clone(key.Task), strings.Clone(key.Node)}] = 1
	} else {
		r.diagnostics++
	}
	// Lambda is fixed at zero. Diagnostic loss never rejects a KAI candidate.
	return 0, nil
}

func (r *nodeProbeRecord) close() { r.gate.Lock(); r.closed = true; r.gate.Unlock() }
func (r *nodeProbeRecord) audit() nodeProbeAudit {
	r.gate.Lock() // Also supplies a stable phase boundary outside the hot path.
	r.mu.Lock()
	a := nodeProbeAudit{r.session, r.closed, r.diagnostics == 0, r.total, r.diagnostics, r.peak.Load(), make([]nodeProbeEntry, 0, len(r.counts))}
	for key, count := range r.counts {
		a.Entries = append(a.Entries, nodeProbeEntry{key.Task, key.Node, count, 0, nil})
	}
	r.mu.Unlock()
	r.gate.Unlock()
	sort.Slice(a.Entries, func(i, j int) bool {
		if a.Entries[i].Task != a.Entries[j].Task {
			return a.Entries[i].Task < a.Entries[j].Task
		}
		return a.Entries[i].Node < a.Entries[j].Node
	})
	return a
}

type nodeScoringProbe struct {
	mu       sync.Mutex
	sessions map[*framework.Session]*nodeProbeRecord
}

var _ framework.Plugin = (*nodeScoringProbe)(nil)
var _ framework.PluginBuilder = newNodeScoringProbe

// No init/registration or main.go changes: enabling this builder is test-only.
func newNodeScoringProbe(_ framework.PluginArguments) framework.Plugin { return &nodeScoringProbe{} }
func (*nodeScoringProbe) Name() string                                 { return nodeProbeName }
func (p *nodeScoringProbe) OnSessionOpen(s *framework.Session) {
	if s == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.sessions == nil {
		p.sessions = make(map[*framework.Session]*nodeProbeRecord)
	}
	if r := p.sessions[s]; r != nil {
		r.gate.Lock()
		r.mu.Lock()
		r.diagnostics++
		r.mu.Unlock()
		r.gate.Unlock()
		return
	}
	session, diagnostics := "", uint64(0)
	if validIdentity(s.ID) {
		session = strings.Clone(s.ID)
	} else {
		diagnostics = 1
	}
	r := &nodeProbeRecord{session: session, diagnostics: diagnostics, counts: make(map[nodeProbeKey]uint64)}
	p.sessions[s] = r
	s.AddNodeOrderFn(r.score)
}
func (p *nodeScoringProbe) OnSessionClose(s *framework.Session) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if r := p.sessions[s]; r != nil {
		r.close()
		delete(p.sessions, s)
	}
}
func (p *nodeScoringProbe) record(s *framework.Session) *nodeProbeRecord {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.sessions[s]
}
