package main

import "fmt"

type Observation struct {
	Event         Event   `json:"event"`
	Association   string  `json:"association"`
	State         string  `json:"state"`
	InvalidatedBy *uint64 `json:"invalidated_by_seq"`
}

type StatementState struct {
	Outcome     string         `json:"outcome"`
	Next        int            `json:"next_operation_index"`
	Checkpoints map[string]int `json:"checkpoints"`
	Poisoned    bool           `json:"uncertain"`
	Closed      bool           `json:"closed"`
	live        map[int]int
}

type BindingState struct {
	Task      string  `json:"task_id"`
	Node      string  `json:"node_id"`
	Statement *string `json:"statement_id"`
	Outcome   string  `json:"outcome"`
	Requested bool    `json:"request_observed"`
	Poisoned  bool    `json:"uncertain"`
}

type LedgerEntry struct {
	Event       Event  `json:"event"`
	Disposition string `json:"disposition"`
}

type Result struct {
	SchemaVersion       string                     `json:"schema_version"`
	Session             string                     `json:"session_id"`
	Mode                string                     `json:"mode"`
	Authority           string                     `json:"authority"`
	Status              string                     `json:"status"`
	RealCommitConfirmed bool                       `json:"real_commit_confirmed"`
	RealBindConfirmed   bool                       `json:"real_bind_confirmed"`
	Observations        []Observation              `json:"observations"`
	Statements          map[string]*StatementState `json:"statements"`
	Bindings            map[string]*BindingState   `json:"bindings"`
	Ledger              []LedgerEntry              `json:"ledger"`
	Issues              []string                   `json:"issues"`
}

// reduce consumes validated JSON, in input order. Real confirmation flags are
// ALWAYS false. Only SYNTHETIC_TEST facts can change simulated transactions.
// Unverified claims never reverse observations or establish success. Poisoning
// is sticky: later success cannot erase contradictory/missing historical evidence.
func reduce(t Trace) Result {
	r := Result{SchemaVersion: "0.1", Session: t.Session, Mode: t.Mode, Authority: t.Authority,
		Status: "SYNTHETIC_REDUCED", Observations: []Observation{}, Statements: map[string]*StatementState{},
		Bindings: map[string]*BindingState{}, Ledger: []LedgerEntry{}, Issues: []string{}}
	issue := func(e Event, why string) {
		r.Status = "TRACE_INCOMPLETE"
		r.Issues = append(r.Issues, fmt.Sprintf("seq %d: %s", e.Seq, why))
	}
	statement := func(id string) *StatementState {
		s := r.Statements[id]
		if s == nil {
			s = &StatementState{Outcome: "OPEN_SYNTHETIC", Checkpoints: map[string]int{}, live: map[int]int{}}
			r.Statements[id] = s
		}
		return s
	}
	poison := func(s *StatementState) { s.Poisoned = true; s.Outcome = "UNKNOWN" }
	var previous uint64
	gap := false
	for _, e := range t.Events {
		disposition := "OBSERVED_ONLY"
		trusted := t.Mode == "SYNTHETIC_TEST" && e.Evidence.Authority == "SYNTHETIC_TEST"
		if e.Seq != previous+1 {
			gap = true
			issue(e, "missing sequence evidence; no complete transaction inference")
			for _, s := range r.Statements {
				poison(s)
			}
			for _, b := range r.Bindings {
				b.Poisoned = true
				b.Outcome = "UNKNOWN"
			}
			// A gap before the first fact also prevents subsequent synthetic success.
		}
		previous = e.Seq
		trusted = trusted && !gap
		if e.Kind == "ALLOCATE_OBSERVED" || e.Kind == "DEALLOCATE_OBSERVED" {
			o := Observation{Event: e, Association: "UNKNOWN", State: "VIRTUAL_OBSERVED"}
			if e.Statement == nil {
				issue(e, "unassociated observation retained; statement_id stays null")
			} else {
				s := statement(*e.Statement)
				if !trusted {
					poison(s)
				}
				if !trusted || s.Poisoned || s.Closed || *e.OperationIndex != s.Next {
					poison(s)
					issue(e, "untrusted, duplicate, missing or closed statement operation")
				} else {
					o.Association = "ATTRIBUTED_SYNTHETIC"
					s.live[s.Next] = len(r.Observations)
					s.Next++
				}
			}
			r.Observations = append(r.Observations, o)
		} else if e.BindRequest != nil {
			id := *e.BindRequest
			b := r.Bindings[id]
			if b == nil {
				b = &BindingState{Task: *e.Task, Node: *e.Node, Statement: e.Statement, Outcome: "UNKNOWN"}
				r.Bindings[id] = b
			}
			sameStatement := (b.Statement == nil && e.Statement == nil) || (b.Statement != nil && e.Statement != nil && *b.Statement == *e.Statement)
			if !trusted || b.Poisoned || b.Task != *e.Task || b.Node != *e.Node || !sameStatement {
				b.Poisoned = true
				b.Outcome = "UNKNOWN"
				issue(e, "untrusted or contradictory Binder evidence")
			} else if e.Kind == "BIND_REQUEST_OBSERVED" {
				if b.Requested {
					b.Poisoned = true
					b.Outcome = "UNKNOWN"
					issue(e, "duplicate BindRequest identity")
				} else {
					b.Requested = true
					b.Outcome = "REQUESTED_SYNTHETIC"
					disposition = "REQUEST_ONLY_SYNTHETIC"
				}
			} else if !b.Requested || b.Outcome != "REQUESTED_SYNTHETIC" {
				b.Poisoned = true
				b.Outcome = "UNKNOWN"
				issue(e, "missing request or contradictory/duplicate Binder completion")
			} else {
				b.Outcome = e.Kind + "_SYNTHETIC"
				disposition = "BINDER_FACT_SYNTHETIC"
			}
		} else {
			s := statement(*e.Statement)
			if !trusted || s.Poisoned || s.Closed {
				poison(s)
				issue(e, "transaction untrusted, contradicted or already closed")
			} else if e.Kind == "CHECKPOINT_CREATED" {
				_, exists := s.Checkpoints[*e.Checkpoint]
				if exists || *e.OperationIndex != s.Next {
					poison(s)
					issue(e, "duplicate checkpoint or missing operation range")
				} else {
					s.Checkpoints[*e.Checkpoint] = s.Next
					disposition = "CHECKPOINT_SYNTHETIC"
				}
			} else {
				start, end := *e.RangeStart, *e.RangeEnd
				valid := end == s.Next
				if e.Kind == "ROLLBACK_CONFIRMED" {
					pos, ok := s.Checkpoints[*e.Checkpoint]
					valid = valid && ok && start == pos
				} else {
					valid = valid && start == 0
				}
				for i := start; i < end; i++ {
					if _, ok := s.live[i]; !ok {
						valid = false
					}
				}
				if !valid {
					poison(s)
					issue(e, "missing/mismatched attributable transaction range")
				} else {
					switch e.Kind {
					case "ROLLBACK_CONFIRMED", "DISCARD_CONFIRMED":
						for i := start; i < end; i++ {
							o := &r.Observations[s.live[i]]
							seq := e.Seq
							o.State = "REVERSED_SYNTHETIC"
							o.InvalidatedBy = &seq
							delete(s.live, i)
						}
						s.Next = start
						for id, pos := range s.Checkpoints {
							if pos >= start {
								delete(s.Checkpoints, id)
							}
						}
						if e.Kind == "DISCARD_CONFIRMED" {
							s.Closed = true
							s.Outcome = "DISCARDED_SYNTHETIC"
						}
						disposition = "RANGE_REVERSED_SYNTHETIC"
					case "COMMIT_SUCCEEDED":
						s.Outcome = "COMMIT_SUCCEEDED_SYNTHETIC"
						s.Closed = true
						disposition = "COMMIT_FACT_SYNTHETIC_NOT_BINDING"
					case "COMMIT_FAILED":
						poison(s)
						s.Closed = true
						for _, index := range s.live {
							r.Observations[index].State = "UNKNOWN_PARTIAL_EFFECTS"
						}
						issue(e, "commit failure may have partial effects; no atomic rollback assumed")
						disposition = "COMMIT_FAILURE_INDETERMINATE"
					}
				}
			}
		}
		r.Ledger = append(r.Ledger, LedgerEntry{e, disposition})
	}
	if t.Mode == "SAVED_UNVERIFIED" && r.Status != "TRACE_INCOMPLETE" {
		r.Status = "OBSERVATION_ONLY_UNVERIFIED"
	}
	return r
}
