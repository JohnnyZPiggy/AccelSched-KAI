package main

import (
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"unicode/utf8"

	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/framework"
)

const captureCapacity = 256

// One plugin normally belongs to one KAI Session. This also safely isolates
// overlapping lightweight Sessions in a harness. Entries are removed on close;
// obtain the capture handle before close to export its sealed audit afterwards.
// No global event storage, source Event/PodInfo pointers or background goroutines.
type captureSessions struct {
	mu     sync.Mutex
	active map[*framework.Session]*capture
}

func (p *captureSessions) openCapture(s *framework.Session) {
	if s == nil {
		return
	} // Preserve the earlier nil-session interface smoke test.
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.active == nil {
		p.active = make(map[*framework.Session]*capture)
	}
	if existing := p.active[s]; existing != nil {
		existing.lifecycleFailure()
		return // Do not duplicate handlers or silently replace an active capture.
	}
	c := newCapture(s.ID)
	p.active[s] = c
	s.AddEventHandler(c.handler)
}

func (p *captureSessions) closeCapture(s *framework.Session) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if c := p.active[s]; c != nil {
		c.Close()
		delete(p.active, s)
	}
}

func (p *captureSessions) sessionCapture(s *framework.Session) *capture {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.active[s]
}

type capturedRecord struct {
	Seq  uint64 `json:"seq"`
	Kind string `json:"kind"`
	Task string `json:"task_id"`
	Node string `json:"node_id"`
}

type capture struct {
	gate            sync.RWMutex // Close drains admitted callbacks; Snapshot only reads sealed data.
	mu              sync.Mutex   // Callback uses TryLock: contention is loss, never waiting.
	once            sync.Once
	closing         atomic.Bool
	failed          atomic.Bool
	closed          bool
	session         string
	sessionError    bool
	handler         *framework.EventHandler
	records         [captureCapacity]capturedRecord
	count           int
	nextSeq         uint64
	received        atomic.Uint64
	invalid         atomic.Uint64
	dropped         atomic.Uint64
	contended       atomic.Uint64
	lifecycleErrors atomic.Uint64
}

func validIdentity(s string) bool {
	// Bound all callback work before scanning Unicode, and clone valid strings so
	// a short substring cannot retain a large mutable source object's backing data.
	return len(s) <= 512 && utf8.ValidString(s) && utf8.RuneCountInString(s) <= 128 && strings.TrimSpace(s) != ""
}

func newCapture(session string) *capture {
	c := &capture{}
	if validIdentity(session) {
		c.session = strings.Clone(session)
	} else {
		c.sessionError = true
		c.failed.Store(true)
	}
	c.handler = &framework.EventHandler{
		AllocateFunc:   func(e *framework.Event) { c.observe("ALLOCATE_OBSERVED", e) },
		DeallocateFunc: func(e *framework.Event) { c.observe("DEALLOCATE_OBSERVED", e) },
	}
	return c
}

func (c *capture) observe(kind string, e *framework.Event) {
	// No blocking locks, serialization, hashing, I/O, source mutation or goroutines.
	// The only gate writer is Close. Arrivals after its closing cutoff are outside
	// the capture window, ignored without ANY buffer/counter writes.
	if c.closing.Load() || !c.gate.TryRLock() {
		return
	}
	defer c.gate.RUnlock()
	if c.closing.Load() {
		return
	}
	c.received.Add(1)
	if !c.mu.TryLock() {
		// No ordered slot can be assigned without this lock. Record the loss,
		// fail normal export permanently, and never invent/renumber an event.
		c.contended.Add(1)
		c.failed.Store(true)
		return
	}
	defer c.mu.Unlock()
	c.nextSeq++ // Serialized attempt order; invalid/overflow positions are not reused.
	if c.sessionError || e == nil || e.Task == nil || (kind != "ALLOCATE_OBSERVED" && kind != "DEALLOCATE_OBSERVED") {
		c.invalid.Add(1)
		c.failed.Store(true)
		return
	}
	// UID maps verbatim to offline task_id; NodeName maps verbatim to node_id.
	// No namespace/name fallback, eligibility/capacity inference or trimming.
	// The callback owner must not concurrently mutate these source fields.
	uid, node := string(e.Task.UID), e.Task.NodeName
	if !validIdentity(uid) || !validIdentity(node) {
		c.invalid.Add(1)
		c.failed.Store(true)
		return
	}
	if c.count == captureCapacity {
		c.dropped.Add(1)
		c.failed.Store(true)
		return
	}
	c.records[c.count] = capturedRecord{c.nextSeq, kind, strings.Clone(uid), strings.Clone(node)}
	c.count++
}

func (c *capture) lifecycleFailure() {
	c.gate.RLock()
	defer c.gate.RUnlock()
	if c.closing.Load() {
		return
	}
	c.lifecycleErrors.Add(1)
	c.failed.Store(true)
}

func (c *capture) Close() {
	c.once.Do(func() {
		c.closing.Store(true)
		c.gate.Lock() // Outside callback: wait for already-admitted work to finish.
		c.closed = true
		c.gate.Unlock()
	})
}

// captureSnapshot is a detached immutable-in-time audit, only available after
// Close. Its Events slice is copied. No live snapshot lock can force silent loss.
// Complete here means capture loss-free, NOT PoC-1A reducer TRACE completeness.
type captureSnapshot struct {
	Session         string           `json:"session_id"`
	Closed          bool             `json:"closed"`
	Complete        bool             `json:"capture_complete"`
	SessionError    bool             `json:"invalid_session_id"`
	Capacity        int              `json:"capacity"`
	Received        uint64           `json:"received"`
	Invalid         uint64           `json:"invalid"`
	Dropped         uint64           `json:"overflow_dropped"`
	Contended       uint64           `json:"contention_dropped"`
	LifecycleErrors uint64           `json:"lifecycle_errors"`
	NextSeq         uint64           `json:"last_serialized_attempt_seq"`
	Events          []capturedRecord `json:"events"`
}

func (c *capture) Snapshot() (captureSnapshot, error) {
	if c == nil {
		return captureSnapshot{}, errors.New("no capture handle")
	}
	c.gate.RLock()
	defer c.gate.RUnlock()
	if !c.closed {
		return captureSnapshot{}, errors.New("capture must be closed before snapshot/export")
	}
	records := make([]capturedRecord, c.count)
	copy(records, c.records[:c.count])
	return captureSnapshot{c.session, true, !c.failed.Load(), c.sessionError, captureCapacity,
		c.received.Load(), c.invalid.Load(), c.dropped.Load(), c.contended.Load(), c.lifecycleErrors.Load(), c.nextSeq, records}, nil
}
