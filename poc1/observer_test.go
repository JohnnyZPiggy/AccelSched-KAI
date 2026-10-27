package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
)

func str(s string) *string { return &s }
func integer(n int) *int   { return &n }
func event(kind string) Event {
	source := "STATEMENT"
	if kind == "ALLOCATE_OBSERVED" || kind == "DEALLOCATE_OBSERVED" {
		source = "EVENT_HANDLER"
	}
	if kind == "BIND_REQUEST_OBSERVED" {
		source = "BIND_REQUEST"
	}
	if kind == "BIND_SUCCEEDED" || kind == "BIND_FAILED" {
		source = "BINDER"
	}
	return Event{Kind: kind, Evidence: Evidence{Authority: "SYNTHETIC_TEST", Source: source, Locator: "synthetic-test-generator", SHA256: digest([]byte("synthetic test evidence; not external proof"))}}
}
func virtual(kind, stmt string, index int) Event {
	e := event(kind)
	e.Task = str("T0")
	e.Node = str("N0")
	if stmt != "" {
		e.Statement = str(stmt)
		e.OperationIndex = integer(index)
	}
	return e
}
func checkpoint(stmt, id string, index int) Event {
	e := event("CHECKPOINT_CREATED")
	e.Statement = str(stmt)
	e.Checkpoint = str(id)
	e.OperationIndex = integer(index)
	return e
}
func transaction(kind, stmt, cp string, start, end int) Event {
	e := event(kind)
	e.Statement = str(stmt)
	if cp != "" {
		e.Checkpoint = str(cp)
	}
	e.RangeStart = integer(start)
	e.RangeEnd = integer(end)
	return e
}
func binding(kind, req string) Event {
	e := event(kind)
	e.Task = str("T0")
	e.Node = str("N0")
	e.BindRequest = str(req)
	return e
}
func trace(events ...Event) Trace {
	for i := range events {
		events[i].Seq = uint64(i + 1)
	}
	return Trace{SchemaVersion: "0.1", KAIRef: kaiRef, Session: "synthetic-session", Mode: "SYNTHETIC_TEST", Authority: "SYNTHETIC_TEST", Events: events}
}
func raw(t *testing.T, v any) []byte {
	t.Helper()
	b, err := encoded(v)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
func snapshot(t *testing.T) []byte {
	t.Helper()
	b, err := os.ReadFile("../fixtures/01_gpu_family.json")
	if err != nil {
		t.Fatal(err)
	}
	return b
}
func checked(t *testing.T, tr Trace) Result {
	t.Helper()
	parsed, err := decodeTrace(raw(t, tr))
	if err != nil {
		t.Fatal(err)
	}
	r := reduce(parsed)
	if r.RealCommitConfirmed || r.RealBindConfirmed {
		t.Fatal("fabricated real-world confirmation")
	}
	if len(r.Ledger) != len(tr.Events) {
		t.Fatal("evidence lost")
	}
	for i, e := range tr.Events {
		if !reflect.DeepEqual(r.Ledger[i].Event, e) {
			t.Fatal("historical fact overwritten")
		}
	}
	if dir := os.Getenv("POC1_EVIDENCE_DIR"); dir != "" {
		if err = os.MkdirAll(dir, 0700); err != nil {
			t.Fatal(err)
		}
		name := strings.ReplaceAll(t.Name(), "/", "_")
		files, err := bundle(raw(t, tr), snapshot(t))
		if err != nil {
			t.Fatal(err)
		}
		if err = writeBundle(filepath.Join(dir, name), files); err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(filepath.Join(dir, name+"-trace-input.json"), raw(t, tr), 0600); err != nil {
			t.Fatal(err)
		}
	}
	return r
}

func TestT01VirtualAllocation(t *testing.T) {
	for _, stmt := range []string{"", "S"} {
		t.Run("statement_"+stmt, func(t *testing.T) {
			r := checked(t, trace(virtual("ALLOCATE_OBSERVED", stmt, 0)))
			if r.Observations[0].State != "VIRTUAL_OBSERVED" || len(r.Bindings) != 0 {
				t.Fatal(r)
			}
			if stmt == "" && (r.Observations[0].Event.Statement != nil || r.Observations[0].Association != "UNKNOWN" || r.Status != "TRACE_INCOMPLETE") {
				t.Fatal(r)
			}
			if stmt != "" && r.Statements[stmt].Outcome != "OPEN_SYNTHETIC" {
				t.Fatal(r)
			}
		})
	}
}

func TestT02CheckpointRollback(t *testing.T) {
	t.Run("only_attributed_range", func(t *testing.T) {
		r := checked(t, trace(virtual("ALLOCATE_OBSERVED", "S", 0), checkpoint("S", "CP", 1), virtual("ALLOCATE_OBSERVED", "", 0), virtual("DEALLOCATE_OBSERVED", "S", 1), transaction("ROLLBACK_CONFIRMED", "S", "CP", 1, 2)))
		if r.Observations[0].State != "VIRTUAL_OBSERVED" || r.Observations[1].State != "VIRTUAL_OBSERVED" || r.Observations[1].Association != "UNKNOWN" || r.Observations[2].State != "REVERSED_SYNTHETIC" {
			t.Fatal(r)
		}
		if r.Observations[2].InvalidatedBy == nil || *r.Observations[2].InvalidatedBy != 5 || r.Statements["S"].Next != 1 {
			t.Fatal(r)
		}
	})
	t.Run("truncated_index_reuse", func(t *testing.T) {
		r := checked(t, trace(checkpoint("S", "CP", 0), virtual("ALLOCATE_OBSERVED", "S", 0), transaction("ROLLBACK_CONFIRMED", "S", "CP", 0, 1), virtual("ALLOCATE_OBSERVED", "S", 0)))
		if r.Observations[0].State != "REVERSED_SYNTHETIC" || r.Observations[1].State != "VIRTUAL_OBSERVED" {
			t.Fatal(r)
		}
	})
	for _, scenario := range []string{"wrong_checkpoint", "wrong_range", "untrusted", "other_statement"} {
		t.Run(scenario, func(t *testing.T) {
			e := transaction("ROLLBACK_CONFIRMED", "S", "CP", 0, 1)
			switch scenario {
			case "wrong_checkpoint":
				e.Checkpoint = str("missing")
			case "wrong_range":
				e.RangeEnd = integer(2)
			case "untrusted":
				e.Evidence.Authority = "UNVERIFIED"
			case "other_statement":
				e.Statement = str("OTHER")
			}
			r := checked(t, trace(checkpoint("S", "CP", 0), virtual("ALLOCATE_OBSERVED", "S", 0), e))
			if r.Status != "TRACE_INCOMPLETE" || r.Observations[0].State != "VIRTUAL_OBSERVED" || r.Observations[0].InvalidatedBy != nil {
				t.Fatal(r)
			}
		})
	}
}

func TestT03DiscardRetry(t *testing.T) {
	for _, discard := range []bool{false, true} {
		t.Run(map[bool]string{false: "callback_pairs_only", true: "explicit_discard_and_fresh_retry"}[discard], func(t *testing.T) {
			events := []Event{virtual("ALLOCATE_OBSERVED", "S", 0), virtual("DEALLOCATE_OBSERVED", "S", 1), virtual("ALLOCATE_OBSERVED", "S", 2)}
			if discard {
				events = append(events, transaction("DISCARD_CONFIRMED", "S", "", 0, 3), virtual("ALLOCATE_OBSERVED", "RETRY", 0))
			}
			r := checked(t, trace(events...))
			want := "VIRTUAL_OBSERVED"
			if discard {
				want = "REVERSED_SYNTHETIC"
			}
			for _, o := range r.Observations[:3] {
				if o.State != want {
					t.Fatal(r)
				}
			}
			if !discard && r.Statements["S"].Outcome != "OPEN_SYNTHETIC" {
				t.Fatal("callback sequence inferred commit")
			}
			if discard && (r.Statements["S"].Outcome != "DISCARDED_SYNTHETIC" || r.Statements["RETRY"].Outcome != "OPEN_SYNTHETIC") {
				t.Fatal(r)
			}
		})
	}
	t.Run("untrusted_discard_preserves_facts", func(t *testing.T) {
		e := transaction("DISCARD_CONFIRMED", "S", "", 0, 1)
		e.Evidence.Authority = "UNVERIFIED"
		r := checked(t, trace(virtual("ALLOCATE_OBSERVED", "S", 0), e))
		if r.Observations[0].State != "VIRTUAL_OBSERVED" || r.Statements["S"].Outcome != "UNKNOWN" {
			t.Fatal(r)
		}
	})
}

func TestT04CommitSuccessFailure(t *testing.T) {
	for _, kind := range []string{"COMMIT_SUCCEEDED", "COMMIT_FAILED"} {
		t.Run(kind, func(t *testing.T) {
			r := checked(t, trace(virtual("ALLOCATE_OBSERVED", "S", 0), transaction(kind, "S", "", 0, 1)))
			if len(r.Bindings) != 0 {
				t.Fatal("Commit fabricated Binder fact")
			}
			if kind == "COMMIT_SUCCEEDED" && r.Statements["S"].Outcome != "COMMIT_SUCCEEDED_SYNTHETIC" {
				t.Fatal(r)
			}
			if kind == "COMMIT_FAILED" && (r.Statements["S"].Outcome != "UNKNOWN" || r.Observations[0].State != "UNKNOWN_PARTIAL_EFFECTS" || r.Observations[0].InvalidatedBy != nil) {
				t.Fatal("failure presumed atomic rollback", r)
			}
		})
	}
	t.Run("unsigned_saved_claim", func(t *testing.T) {
		tr := trace(virtual("ALLOCATE_OBSERVED", "S", 0), transaction("COMMIT_SUCCEEDED", "S", "", 0, 1))
		tr.Mode = "SAVED_UNVERIFIED"
		tr.Authority = "UNVERIFIED"
		for i := range tr.Events {
			tr.Events[i].Evidence.Authority = "UNVERIFIED"
		}
		r := checked(t, tr)
		if r.Statements["S"].Outcome != "UNKNOWN" || r.Status != "TRACE_INCOMPLETE" {
			t.Fatal(r)
		}
	})
	t.Run("untrusted_cannot_be_repaired_by_success", func(t *testing.T) {
		e := transaction("COMMIT_SUCCEEDED", "S", "", 0, 1)
		e.Evidence.Authority = "UNVERIFIED"
		r := checked(t, trace(virtual("ALLOCATE_OBSERVED", "S", 0), e, transaction("COMMIT_SUCCEEDED", "S", "", 0, 1)))
		if r.Statements["S"].Outcome != "UNKNOWN" {
			t.Fatal(r)
		}
	})
	t.Run("contradictory_success_failure", func(t *testing.T) {
		r := checked(t, trace(virtual("ALLOCATE_OBSERVED", "S", 0), transaction("COMMIT_SUCCEEDED", "S", "", 0, 1), transaction("COMMIT_FAILED", "S", "", 0, 1)))
		if r.Statements["S"].Outcome != "UNKNOWN" {
			t.Fatal(r)
		}
	})
	t.Run("missing_operation", func(t *testing.T) {
		r := checked(t, trace(transaction("COMMIT_SUCCEEDED", "S", "", 0, 1)))
		if r.Statements["S"].Outcome != "UNKNOWN" {
			t.Fatal(r)
		}
	})
}

func TestT05BindRequestBinder(t *testing.T) {
	for _, kind := range []string{"", "BIND_SUCCEEDED", "BIND_FAILED"} {
		t.Run("request_"+kind, func(t *testing.T) {
			events := []Event{binding("BIND_REQUEST_OBSERVED", "R")}
			want := "REQUESTED_SYNTHETIC"
			if kind != "" {
				events = append(events, binding(kind, "R"))
				want = kind + "_SYNTHETIC"
			}
			r := checked(t, trace(events...))
			if r.Bindings["R"].Outcome != want || r.Bindings["R"].Statement != nil || len(r.Statements) != 0 {
				t.Fatal(r)
			}
		})
	}
	for _, scenario := range []string{"missing_request", "different_node", "duplicate_request", "contradiction", "untrusted"} {
		t.Run(scenario, func(t *testing.T) {
			events := []Event{binding("BIND_REQUEST_OBSERVED", "R"), binding("BIND_SUCCEEDED", "R")}
			switch scenario {
			case "missing_request":
				events = events[1:]
			case "different_node":
				events[1].Node = str("N1")
			case "duplicate_request":
				events[1] = events[0]
			case "contradiction":
				events = append(events, binding("BIND_FAILED", "R"))
			case "untrusted":
				events[1].Evidence.Authority = "UNVERIFIED"
			}
			r := checked(t, trace(events...))
			if r.Bindings["R"].Outcome != "UNKNOWN" || r.Status != "TRACE_INCOMPLETE" {
				t.Fatal(r)
			}
		})
	}
}

func TestT06StrictSchemaReproducibility(t *testing.T) {
	t.Run("strict_trace", func(t *testing.T) {
		base := raw(t, trace(virtual("ALLOCATE_OBSERVED", "S", 0)))
		cases := map[string][]byte{
			"duplicate_key":              bytes.Replace(base, []byte(`"schema_version": "0.1"`), []byte(`"schema_version": "0.1", "schema_version":"0.1"`), 1),
			"unknown_field":              bytes.Replace(base, []byte(`"session_id"`), []byte(`"sessionID"`), 1),
			"case_folded_field":          bytes.Replace(base, []byte(`"mode"`), []byte(`"MODE"`), 1),
			"missing_nullable":           bytes.Replace(base, []byte("      \"checkpoint_id\": null,\n"), nil, 1),
			"unknown_event_field":        bytes.Replace(base, []byte(`"seq": 1`), []byte(`"seq":1,"trusted":true`), 1),
			"authority_forgery":          bytes.ReplaceAll(base, []byte(`"SYNTHETIC_TEST"`), []byte(`"VERIFIED_EXTERNAL"`)),
			"evidence_authority_forgery": bytes.Replace(base, []byte(`"source": "EVENT_HANDLER"`), []byte(`"source":"EVENT_HANDLER","verified":true`), 1),
			"missing_provenance":         bytes.Replace(base, []byte(`"locator": "synthetic-test-generator"`), []byte(`"locator": ""`), 1),
			"bad_hash":                   bytes.Replace(base, []byte(digest([]byte("synthetic test evidence; not external proof"))), []byte("bad"), 1),
			"wrong_pin":                  bytes.Replace(base, []byte(kaiRef), []byte(accelRef), 1),
			"malformed":                  []byte("{"), "trailing": append(append([]byte{}, base...), []byte("{}")...), "nonfinite": []byte(`{"x":NaN}`), "invalid_utf8": {0xff},
		}
		for name, data := range cases {
			t.Run(name, func(t *testing.T) {
				if bytes.Equal(data, base) {
					t.Fatal("negative fixture did not mutate input")
				}
				if _, err := decodeTrace(data); err == nil {
					t.Fatal("accepted malformed trace")
				}
			})
		}
		for _, name := range []string{"duplicate_seq", "out_of_order"} {
			t.Run(name, func(t *testing.T) {
				tr := trace(virtual("ALLOCATE_OBSERVED", "S", 0), virtual("DEALLOCATE_OBSERVED", "S", 1))
				if name == "duplicate_seq" {
					tr.Events[1].Seq = 1
				} else {
					tr.Events[0].Seq = 2
					tr.Events[1].Seq = 1
				}
				if _, err := decodeTrace(raw(t, tr)); err == nil {
					t.Fatal("accepted unordered evidence")
				}
			})
		}
	})
	t.Run("gaps_poison_new_identities", func(t *testing.T) {
		tr := trace(virtual("ALLOCATE_OBSERVED", "S", 0), transaction("COMMIT_SUCCEEDED", "S", "", 0, 1), binding("BIND_REQUEST_OBSERVED", "R"), binding("BIND_SUCCEEDED", "R"))
		for i := range tr.Events {
			tr.Events[i].Seq += 4
		}
		r := checked(t, tr)
		if r.Statements["S"].Outcome != "UNKNOWN" || r.Bindings["R"].Outcome != "UNKNOWN" {
			t.Fatal(r)
		}
	})
	t.Run("duplicate_semantic_operation", func(t *testing.T) {
		r := checked(t, trace(virtual("ALLOCATE_OBSERVED", "S", 0), virtual("ALLOCATE_OBSERVED", "S", 0)))
		if r.Statements["S"].Outcome != "UNKNOWN" {
			t.Fatal(r)
		}
	})
	t.Run("poc0_exact_handoff", func(t *testing.T) {
		for _, name := range []string{"01_gpu_family.json", "02_fragmentation.json", "03_prediction_shift.json", "04_dual_degeneracy.json"} {
			b, err := os.ReadFile(filepath.Join("../fixtures", name))
			if err != nil {
				t.Fatal(err)
			}
			if _, err = validateSnapshot(b); err != nil {
				t.Fatal(name, err)
			}
		}
		tr := raw(t, trace(virtual("ALLOCATE_OBSERVED", "", 0)))
		sn := snapshot(t)
		a, err := bundle(tr, sn)
		if err != nil {
			t.Fatal(err)
		}
		b, err := bundle(tr, sn)
		if err != nil || !reflect.DeepEqual(a, b) || !bytes.Equal(a["snapshot.json"], sn) {
			t.Fatal("nonreproducible or modified snapshot")
		}
		var manifest Manifest
		if err = json.Unmarshal(a["manifest.json"], &manifest); err != nil {
			t.Fatal(err)
		}
		if len(manifest.Outputs) != 2 || manifest.Outputs["snapshot.json"] != digest(sn) || manifest.Outputs["trace-result.json"] != digest(a["trace-result.json"]) || manifest.Inputs["trace"] != digest(tr) {
			t.Fatal("hash mismatch/self-reference")
		}
		var wg sync.WaitGroup
		for i := 0; i < 8; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				c, err := bundle(tr, sn)
				if err != nil || !reflect.DeepEqual(a, c) {
					t.Error("concurrent nondeterminism", err)
				}
			}()
		}
		wg.Wait()
	})
	t.Run("strict_snapshot", func(t *testing.T) {
		mutations := map[string]func(map[string]any){
			"missing_field":    func(s map[string]any) { delete(s, "future") },
			"event_pollution":  func(s map[string]any) { s["events"] = []any{} },
			"unsupported":      func(s map[string]any) { s["constraints"] = map[string]any{"gang": true} },
			"null_constraints": func(s map[string]any) { s["constraints"] = nil },
			"wrong_units":      func(s map[string]any) { s["units"] = []string{"GPU", "CPU", "RAM"} },
			"null_coordinate":  func(s map[string]any) { s["nodes"].([]any)[0].(map[string]any)["capacity"] = []any{nil, 8, 8} },
			"fractional_gpu":   func(s map[string]any) { s["tasks"].([]any)[0].(map[string]any)["demand"] = []any{0.5, 2, 2} },
			"boolean_value":    func(s map[string]any) { s["tasks"].([]any)[0].(map[string]any)["value"] = true },
			"null_value":       func(s map[string]any) { s["tasks"].([]any)[0].(map[string]any)["value"] = nil },
			"unknown_node":     func(s map[string]any) { s["tasks"].([]any)[0].(map[string]any)["eligible"] = []string{"missing"} },
			"duplicate_node":   func(s map[string]any) { s["nodes"].([]any)[1].(map[string]any)["id"] = "N0" },
		}
		for name, change := range mutations {
			t.Run(name, func(t *testing.T) {
				var s map[string]any
				if err := json.Unmarshal(snapshot(t), &s); err != nil {
					t.Fatal(err)
				}
				change(s)
				if _, err := validateSnapshot(raw(t, s)); err == nil {
					t.Fatal("accepted invalid PoC-0 snapshot")
				}
			})
		}
	})
	t.Run("cli_and_no_overwrite", func(t *testing.T) {
		dir := t.TempDir()
		tracePath := filepath.Join(dir, "trace.json")
		snPath := filepath.Join(dir, "supplied.json")
		out := filepath.Join(dir, "out")
		if err := os.WriteFile(tracePath, raw(t, trace(virtual("ALLOCATE_OBSERVED", "", 0))), 0600); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(snPath, snapshot(t), 0600); err != nil {
			t.Fatal(err)
		}
		args := []string{"-trace", tracePath, "-snapshot", snPath, "-out", out}
		var stderr bytes.Buffer
		if run(args, &stderr) != 0 {
			t.Fatal(stderr.String())
		}
		if run(args, &stderr) != 1 {
			t.Fatal("overwrote output")
		}
		if run(nil, &stderr) != 2 {
			t.Fatal("usage exit")
		}
		if err := os.WriteFile(tracePath, []byte("{"), 0600); err != nil {
			t.Fatal(err)
		}
		args[len(args)-1] = filepath.Join(dir, "not-created")
		if run(args, &stderr) != 1 {
			t.Fatal("invalid input exit")
		}
		if _, err := os.Stat(args[len(args)-1]); !os.IsNotExist(err) {
			t.Fatal("created bundle for invalid input")
		}
	})
}
