package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
)

func digest(data []byte) string { h := sha256.Sum256(data); return hex.EncodeToString(h[:]) }

// Generated JSON = encoding/json.MarshalIndent (struct field order, sorted map
// keys, Go JSON escaping/number format), two-space indent, exactly one final LF.
// No timestamps/absolute paths; bytes reproduce under the recorded Go version.
// Snapshot is the original raw file, NOT canonicalized. Hashes cover raw bytes.
// Manifest hashes inputs and the other two outputs, NEVER itself.
func encoded(v any) ([]byte, error) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}

type Manifest struct {
	TraceSchema string            `json:"trace_schema"`
	GoVersion   string            `json:"go_version"`
	Refs        map[string]string `json:"pinned_refs"`
	Inputs      map[string]string `json:"input_sha256"`
	Outputs     map[string]string `json:"output_sha256"`
	Authority   string            `json:"authority"`
	Limitations []string          `json:"limitations"`
}

func bundle(traceRaw, snapshotRaw []byte) (map[string][]byte, error) {
	t, err := decodeTrace(traceRaw)
	if err != nil {
		return nil, err
	}
	s, err := validateSnapshot(snapshotRaw)
	if err != nil {
		return nil, err
	}
	for _, e := range t.Events {
		if e.Task != nil && !s.Tasks[*e.Task] {
			return nil, fmt.Errorf("trace task absent from supplied snapshot")
		}
		if e.Node != nil && !s.Nodes[*e.Node] {
			return nil, fmt.Errorf("trace node absent from supplied snapshot")
		}
	}
	result, err := encoded(reduce(t))
	if err != nil {
		return nil, err
	}
	m := Manifest{TraceSchema: "0.1", GoVersion: runtime.Version(),
		Refs:    map[string]string{"poc0": poc0Ref, "accelsched": accelRef, "kai_scheduler": kaiRef},
		Inputs:  map[string]string{"trace": digest(traceRaw), "snapshot": digest(snapshotRaw)},
		Outputs: map[string]string{"snapshot.json": digest(snapshotRaw), "trace-result.json": digest(result)}, Authority: t.Authority,
		Limitations: []string{
			"Offline shadow observer only; no native EventHandler transaction/Binder success callbacks exist.",
			"Unsigned JSON and evidence SHA/locator self-claims do not establish external authenticity; no verifier is provided.",
			"Real Commit/Binding confirmation is always false; only explicitly labeled synthetic outcomes can be represented.",
			"Null statement IDs stay unassociated. No inference from task/time proximity, allocation pairs or Commit to Binding.",
			"Checkpoint/range operations require contiguous synthetic attribution; fresh statement IDs after Commit/Discard; gaps poison subsequent reductions.",
			"Snapshot copied byte-for-byte after strict PoC-0 validation; no events change capacity, assignments or eligibility. This is not a physical-feasibility certificate.",
			"Raw-byte SHA-256, no self-hash; generated JSON uses Go MarshalIndent plus LF; no timestamp or path fields.",
			"Each input <=1 MiB, JSON depth<=32, trace<=4096 events; no Kubernetes/GPU/Colab/network/paid calls.",
		}}
	manifest, err := encoded(m)
	if err != nil {
		return nil, err
	}
	return map[string][]byte{"snapshot.json": append([]byte(nil), snapshotRaw...), "trace-result.json": result, "manifest.json": manifest}, nil
}

func writeBundle(dir string, files map[string][]byte) error {
	// Require a NEW directory. Never overwrite input/evidence or reuse stale output.
	// Manifest is written last; absence of manifest means incomplete filesystem output.
	if err := os.Mkdir(dir, 0700); err != nil {
		return err
	}
	for _, name := range []string{"snapshot.json", "trace-result.json", "manifest.json"} {
		if err := os.WriteFile(filepath.Join(dir, name), files[name], 0600); err != nil {
			return err
		}
	}
	return nil
}
