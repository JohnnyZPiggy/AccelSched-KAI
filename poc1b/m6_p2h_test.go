package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/test_utils"
)

func m6EvidenceDir(t *testing.T) string {
	if d := os.Getenv("M6_EVIDENCE_DIR"); d != "" {
		if err := os.MkdirAll(d, 0700); err != nil {
			t.Fatal(err)
		}
		return d
	}
	return t.TempDir()
}

// TestM6P2H is the hard gate: Stage B/C are NOT_RUN unless the whole of Stage A passed.
func TestM6P2H(t *testing.T) {
	test_utils.InitTestingInfrastructure()
	dir := m6EvidenceDir(t)
	var stageA map[string]any
	if !t.Run("StageA", func(t *testing.T) { stageA = m6StageA(t, filepath.Join(dir, "stage_a")) }) {
		raw, _ := json.Marshal(map[string]any{"status": "P2H_NATIVE_MAPPING_BLOCKED", "stage_b": "NOT_RUN", "stage_c": "NOT_RUN"})
		_ = os.WriteFile(filepath.Join(dir, "blocked.json"), raw, 0600)
		t.Fatal("P2H_NATIVE_MAPPING_BLOCKED: Stage B/C NOT_RUN")
	}
	rawA, err := json.Marshal(stageA)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(dir, "stage_a.json"), rawA, 0600); err != nil {
		t.Fatal(err)
	}
	var normalized map[string]any
	var timings []any
	var batch map[string]any
	if !t.Run("StageBC", func(t *testing.T) {
		batch = m6SameBatch(t, filepath.Join(dir, "same_batch"))
		normalized, timings = m6StageBC(t, filepath.Join(dir, "stage_bc"))
	}) {
		t.Fatal("Stage B/C failed")
	}
	normalized["stage_a_sha256"] = m4bHash(rawA)
	normalized["same_batch"] = batch
	raw, err := json.Marshal(normalized)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(dir, "normalized.json"), raw, 0600); err != nil {
		t.Fatal(err)
	}
	nativeSave(t, dir, "timings.json", timings)
	t.Log("stage_a_sha256 ", m4bHash(rawA), " normalized_sha256 ", m4bHash(raw))
}
