package main

// Trace v0.1 has EXACT keys shown by Trace/Event/Evidence JSON tags below; every
// key is required, including nullable identifiers/indices. Unknown/case-folded or
// duplicate keys, trailing values and unordered/repeated seq are rejected.
//
// Authority policy (machine checked, no trust flag or verification override):
// SYNTHETIC_TEST mode permits SYNTHETIC_TEST or UNVERIFIED events. SAVED_UNVERIFIED
// permits only UNVERIFIED. No JSON value grants real-world authority. SHA-256 and
// evidence locators are integrity/provenance claims, not signatures or verified
// external proofs. No external verifier is implemented in this offline PoC.
//
// Synthetic statement operation indices are zero-based contiguous identities,
// supplied by a test harness, NOT fields inferred from native EventHandler.Task.
// Checkpoint stores the next operation index. Rollback/Discard/Commit explicitly
// identify a half-open range [range_start,range_end). Statement IDs must be fresh
// after Commit/Discard. Missing/contradictory attribution fails closed to UNKNOWN.

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"strings"
	"unicode/utf8"
)

const poc0Ref = "be8e4073eba3199990f5bd7c93a70558f39d4bdd"
const accelRef = "9dc0c263d2aefeb151d565166b6b950e2ac9f6a5"
const kaiRef = "374cbb796ecff07c4a9140e59286f3e3a280e84b"

type Evidence struct {
	Authority string `json:"authority"`
	Source    string `json:"source"`
	Locator   string `json:"locator"`
	SHA256    string `json:"sha256"`
}

type Event struct {
	Seq            uint64   `json:"seq"`
	Kind           string   `json:"kind"`
	Task           *string  `json:"task_id"`
	Node           *string  `json:"node_id"`
	Statement      *string  `json:"statement_id"`
	Checkpoint     *string  `json:"checkpoint_id"`
	OperationIndex *int     `json:"operation_index"`
	RangeStart     *int     `json:"range_start"`
	RangeEnd       *int     `json:"range_end"`
	BindRequest    *string  `json:"bind_request_id"`
	Evidence       Evidence `json:"evidence"`
}

type Trace struct {
	SchemaVersion string  `json:"schema_version"`
	KAIRef        string  `json:"kai_ref"`
	Session       string  `json:"session_id"`
	Mode          string  `json:"mode"`
	Authority     string  `json:"authority"`
	Events        []Event `json:"events"`
}

// strictJSON checks duplicate keys recursively before any struct decoding, which
// otherwise silently accepts duplicates. No input is rewritten during validation.
func strictJSON(data []byte) error {
	if len(data) > 1<<20 || !utf8.Valid(data) {
		return fmt.Errorf("invalid UTF-8 or input exceeds 1 MiB")
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.UseNumber()
	var walk func(int) error
	walk = func(depth int) error {
		if depth > 32 {
			return fmt.Errorf("JSON exceeds depth 32")
		}
		token, err := d.Token()
		if err != nil {
			return err
		}
		delim, ok := token.(json.Delim)
		if !ok {
			return nil
		}
		switch delim {
		case '{':
			seen := map[string]bool{}
			for d.More() {
				token, err = d.Token()
				if err != nil {
					return err
				}
				key, ok := token.(string)
				if !ok || seen[key] {
					return fmt.Errorf("duplicate/invalid JSON key %v", token)
				}
				seen[key] = true
				if err = walk(depth + 1); err != nil {
					return err
				}
			}
		case '[':
			for d.More() {
				if err = walk(depth + 1); err != nil {
					return err
				}
			}
		default:
			return fmt.Errorf("unexpected delimiter")
		}
		_, err = d.Token()
		return err
	}
	if err := walk(0); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return fmt.Errorf("trailing JSON or malformed document")
	}
	return nil
}

func object(data []byte, keys ...string) (map[string]json.RawMessage, error) {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(data, &m); err != nil || m == nil || len(m) != len(keys) {
		return nil, fmt.Errorf("exact object keys required: %v", keys)
	}
	for _, key := range keys {
		if _, ok := m[key]; !ok {
			return nil, fmt.Errorf("missing/unknown field: %s", key)
		}
	}
	return m, nil
}

func text(s string) bool        { return strings.TrimSpace(s) != "" && utf8.RuneCountInString(s) <= 128 }
func identifier(s *string) bool { return s == nil || text(*s) }

func decodeTrace(data []byte) (Trace, error) {
	var t Trace
	if err := strictJSON(data); err != nil {
		return t, err
	}
	m, err := object(data, "schema_version", "kai_ref", "session_id", "mode", "authority", "events")
	if err != nil {
		return t, err
	}
	if err = json.Unmarshal(data, &t); err != nil {
		return t, err
	}
	if t.SchemaVersion != "0.1" || t.KAIRef != kaiRef || !text(t.Session) || t.Events == nil || len(t.Events) > 4096 {
		return t, fmt.Errorf("invalid trace metadata/size/pinned KAI reference")
	}
	if !((t.Mode == "SYNTHETIC_TEST" && t.Authority == "SYNTHETIC_TEST") || (t.Mode == "SAVED_UNVERIFIED" && t.Authority == "UNVERIFIED")) {
		return t, fmt.Errorf("JSON cannot establish trusted authority")
	}
	var events []json.RawMessage
	_ = json.Unmarshal(m["events"], &events)
	var previous uint64
	for i, e := range t.Events {
		v, err := object(events[i], "seq", "kind", "task_id", "node_id", "statement_id", "checkpoint_id", "operation_index", "range_start", "range_end", "bind_request_id", "evidence")
		if err != nil {
			return t, err
		}
		if _, err = object(v["evidence"], "authority", "source", "locator", "sha256"); err != nil {
			return t, err
		}
		if e.Seq <= previous || !identifier(e.Task) || !identifier(e.Node) || !identifier(e.Statement) || !identifier(e.Checkpoint) || !identifier(e.BindRequest) {
			return t, fmt.Errorf("invalid identifiers or non-increasing unique seq at event %d", i)
		}
		previous = e.Seq
		for _, n := range []*int{e.OperationIndex, e.RangeStart, e.RangeEnd} {
			if n != nil && (*n < 0 || *n > 4096) {
				return t, fmt.Errorf("invalid index/range")
			}
		}
		proof, err := hex.DecodeString(e.Evidence.SHA256)
		if err != nil || len(proof) != 32 || strings.ToLower(e.Evidence.SHA256) != e.Evidence.SHA256 || !text(e.Evidence.Locator) {
			return t, fmt.Errorf("invalid evidence provenance")
		}
		if e.Evidence.Authority != "UNVERIFIED" && !(t.Mode == "SYNTHETIC_TEST" && e.Evidence.Authority == "SYNTHETIC_TEST") {
			return t, fmt.Errorf("unverified file cannot claim independent authority")
		}
		virtual := e.Kind == "ALLOCATE_OBSERVED" || e.Kind == "DEALLOCATE_OBSERVED"
		txn := e.Kind == "CHECKPOINT_CREATED" || e.Kind == "ROLLBACK_CONFIRMED" || e.Kind == "DISCARD_CONFIRMED" || e.Kind == "COMMIT_SUCCEEDED" || e.Kind == "COMMIT_FAILED"
		binding := e.Kind == "BIND_REQUEST_OBSERVED" || e.Kind == "BIND_SUCCEEDED" || e.Kind == "BIND_FAILED"
		switch {
		case virtual:
			if e.Evidence.Source != "EVENT_HANDLER" || e.Task == nil || e.Node == nil || e.Checkpoint != nil || e.BindRequest != nil || e.RangeStart != nil || e.RangeEnd != nil || (e.Statement == nil) != (e.OperationIndex == nil) {
				return t, fmt.Errorf("malformed virtual observation")
			}
		case txn:
			if e.Evidence.Source != "STATEMENT" || e.Statement == nil || e.Task != nil || e.Node != nil || e.BindRequest != nil {
				return t, fmt.Errorf("malformed transaction evidence")
			}
			if e.Kind == "CHECKPOINT_CREATED" {
				if e.Checkpoint == nil || e.OperationIndex == nil || e.RangeStart != nil || e.RangeEnd != nil {
					return t, fmt.Errorf("checkpoint requires explicit next operation index")
				}
			} else if e.OperationIndex != nil || e.RangeStart == nil || e.RangeEnd == nil || *e.RangeStart > *e.RangeEnd || (e.Kind == "ROLLBACK_CONFIRMED") != (e.Checkpoint != nil) {
				return t, fmt.Errorf("transaction requires explicit attributable range")
			}
		case binding:
			source := "BINDER"
			if e.Kind == "BIND_REQUEST_OBSERVED" {
				source = "BIND_REQUEST"
			}
			if e.Evidence.Source != source || e.Task == nil || e.Node == nil || e.BindRequest == nil || e.Checkpoint != nil || e.OperationIndex != nil || e.RangeStart != nil || e.RangeEnd != nil {
				return t, fmt.Errorf("malformed Binder evidence")
			}
		default:
			return t, fmt.Errorf("unknown event kind %q", e.Kind)
		}
	}
	return t, nil
}

type Snapshot struct{ Nodes, Tasks map[string]bool }

func vector(raw json.RawMessage) error {
	var v []json.RawMessage
	if err := json.Unmarshal(raw, &v); err != nil || len(v) != 3 {
		return fmt.Errorf("three integer resource coordinates required")
	}
	for _, r := range v {
		var x int
		if bytes.Equal(bytes.TrimSpace(r), []byte("null")) || json.Unmarshal(r, &x) != nil || x < 0 || x > 1000000 {
			return fmt.Errorf("integer resource out of bounds")
		}
	}
	return nil
}

// validateSnapshot mirrors the frozen PoC-0 boundary with no Python import/call.
// Snapshot bytes are exported UNCHANGED; no event-derived fields or eligibility.
func validateSnapshot(data []byte) (Snapshot, error) {
	s := Snapshot{map[string]bool{}, map[string]bool{}}
	if err := strictJSON(data); err != nil {
		return s, err
	}
	m, err := object(data, "schema_version", "units", "nodes", "tasks", "future", "realized_future", "constraints")
	if err != nil {
		return s, err
	}
	var version string
	_ = json.Unmarshal(m["schema_version"], &version)
	var units []string
	_ = json.Unmarshal(m["units"], &units)
	if version != "0.1" || len(units) != 3 || units[0] != "whole GPUs" || units[1] != "integer CPU cores" || units[2] != "integer RAM GiB" {
		return s, fmt.Errorf("PoC-0 schema/units mismatch")
	}
	if _, err = object(m["constraints"]); err != nil {
		return s, fmt.Errorf("PoC-0 unsupported or malformed constraints")
	}
	counts := map[string]int{}
	for _, name := range []string{"nodes", "tasks", "future", "realized_future"} {
		var items []json.RawMessage
		if err = json.Unmarshal(m[name], &items); err != nil || items == nil {
			return s, fmt.Errorf("required array %s", name)
		}
		limit, min := 8, 1
		if name == "tasks" {
			limit = 16
		}
		if name == "realized_future" {
			min = 0
		}
		if len(items) < min || len(items) > limit {
			return s, fmt.Errorf("PoC-0 bounded size: %s", name)
		}
		counts[name] = len(items)
		seen := map[string]bool{}
		for _, item := range items {
			keys := []string{"id", "gpu_family", "shape", "capacity"}
			objective := "value"
			if name != "nodes" {
				if name == "future" {
					objective = "weight"
				}
				keys = []string{"id", "gpu_family", "eligible", "demand", objective}
			}
			v, err := object(item, keys...)
			if err != nil {
				return s, err
			}
			var id string
			if json.Unmarshal(v["id"], &id) != nil || !text(id) || seen[id] {
				return s, fmt.Errorf("invalid/duplicate ID")
			}
			seen[id] = true
			var family *string
			if json.Unmarshal(v["gpu_family"], &family) != nil || !identifier(family) || (name == "nodes" && family == nil) {
				return s, fmt.Errorf("invalid GPU family")
			}
			if name == "nodes" {
				var shape string
				if json.Unmarshal(v["shape"], &shape) != nil || !text(shape) {
					return s, fmt.Errorf("invalid Shape")
				}
				if err = vector(v["capacity"]); err != nil {
					return s, err
				}
				s.Nodes[id] = true
			} else {
				if err = vector(v["demand"]); err != nil {
					return s, err
				}
				var d []int
				_ = json.Unmarshal(v["demand"], &d)
				if d[0]+d[1]+d[2] == 0 {
					return s, fmt.Errorf("zero demand")
				}
				var value float64
				if bytes.Equal(bytes.TrimSpace(v[objective]), []byte("null")) || json.Unmarshal(v[objective], &value) != nil || math.IsNaN(value) || math.IsInf(value, 0) || value < 0 || value > 1000000 {
					return s, fmt.Errorf("invalid value/weight")
				}
				var eligible []string
				if json.Unmarshal(v["eligible"], &eligible) != nil || eligible == nil {
					return s, fmt.Errorf("required eligibility list")
				}
				es := map[string]bool{}
				for _, n := range eligible {
					if !s.Nodes[n] || es[n] {
						return s, fmt.Errorf("unknown/duplicate eligible node")
					}
					es[n] = true
				}
				if name == "tasks" {
					s.Tasks[id] = true
				}
			}
		}
	}
	bound := 1
	for i := 0; i < counts["realized_future"]; i++ {
		bound *= counts["nodes"] + 1
		if bound > 100000 {
			return s, fmt.Errorf("PoC-0 offline search bound")
		}
	}
	return s, nil
}
