package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
)

const captureKAIRef = "374cbb796ecff07c4a9140e59286f3e3a280e84b"

type captureEvidence struct {
	Authority string `json:"authority"`
	Source    string `json:"source"`
	Locator   string `json:"locator"`
	SHA256    string `json:"sha256"`
}

type captureEvent struct {
	Seq            uint64          `json:"seq"`
	Kind           string          `json:"kind"`
	Task           string          `json:"task_id"`
	Node           string          `json:"node_id"`
	Statement      *string         `json:"statement_id"`
	Checkpoint     *string         `json:"checkpoint_id"`
	OperationIndex *int            `json:"operation_index"`
	RangeStart     *int            `json:"range_start"`
	RangeEnd       *int            `json:"range_end"`
	BindRequest    *string         `json:"bind_request_id"`
	Evidence       captureEvidence `json:"evidence"`
}

type captureTrace struct {
	SchemaVersion string         `json:"schema_version"`
	KAIRef        string         `json:"kai_ref"`
	Session       string         `json:"session_id"`
	Mode          string         `json:"mode"`
	Authority     string         `json:"authority"`
	Events        []captureEvent `json:"events"`
}

// Integrity record canonical bytes = json.Marshal of THIS struct, in this exact
// field order, compact Go JSON escaping/UTF-8, no trailing newline. Session and
// seq bind context. Hash is computed outside the callback. It authenticates
// NOTHING: saved in-process observations always remain SAVED_UNVERIFIED.
type integrityRecord struct {
	SchemaVersion string `json:"schema_version"`
	KAIRef        string `json:"kai_ref"`
	Session       string `json:"session_id"`
	Seq           uint64 `json:"seq"`
	Kind          string `json:"kind"`
	Task          string `json:"task_id"`
	Node          string `json:"node_id"`
}

func (c *capture) MarshalTrace() ([]byte, error) {
	s, err := c.Snapshot()
	if err != nil {
		return nil, err
	}
	if !s.Complete {
		return nil, errors.New("capture incomplete: inspect separate Snapshot loss/error audit; normal Trace export refused")
	}
	t := captureTrace{"0.1", captureKAIRef, s.Session, "SAVED_UNVERIFIED", "UNVERIFIED", make([]captureEvent, 0, len(s.Events))}
	for _, r := range s.Events {
		canonical, err := json.Marshal(integrityRecord{"0.1", captureKAIRef, s.Session, r.Seq, r.Kind, r.Task, r.Node})
		if err != nil {
			return nil, err
		}
		sum := sha256.Sum256(canonical)
		h := hex.EncodeToString(sum[:])
		t.Events = append(t.Events, captureEvent{Seq: r.Seq, Kind: r.Kind, Task: r.Task, Node: r.Node,
			Evidence: captureEvidence{"UNVERIFIED", "EVENT_HANDLER", "kai-event-handler/record-sha256/" + h, h}})
	}
	b, err := json.MarshalIndent(t, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}
