package main

// M7-B2: producer latency study. The SAME exact request bytes (captured from real native states of the
// cold PGFP arm) are replayed through (a) a fresh instrumented process per task and (b) a long-lived
// experimental worker; outputs must be byte-identical to the regular cold producer's outputs. Stage times
// are non-overlapping (Python-internal snapshot/build/load/other, with the LP solve reported as a nested
// part of build). Nothing here is wired to a plugin or weakens a grant.

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"
)

type m7Req struct {
	Path     string
	Phase    int
	Alpha    float64
	Praw     []byte
	Forecast []m4bForecast
	Ref      map[string][]byte
}

func m7LoadRequests(t *testing.T, nativeDir string, limit int) []m7Req {
	t.Helper()
	files, _ := filepath.Glob(filepath.Join(nativeDir, "scenes", "*", "0", "pgfp_l40", "arrival-*", "handoff", "request.json"))
	sort.Strings(files)
	out := []m7Req{}
	for _, f := range files {
		raw, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		var r struct {
			Projection string        `json:"projection_b64"`
			Phase      int           `json:"phase"`
			Alpha      float64       `json:"alpha"`
			Forecast   []m4bForecast `json:"forecast"`
		}
		if json.Unmarshal(raw, &r) != nil {
			t.Fatal("request file ", f)
		}
		praw, _ := base64.StdEncoding.DecodeString(r.Projection)
		ref := map[string][]byte{}
		for _, n := range []string{"envelope", "artifact", "snapshot", "controls"} {
			ref[n], _ = os.ReadFile(filepath.Join(filepath.Dir(f), n+".json"))
		}
		out = append(out, m7Req{f, r.Phase, r.Alpha, praw, r.Forecast, ref})
		if len(out) == limit {
			break
		}
	}
	if len(out) < 8 {
		t.Fatal("too few matched requests captured from the native stage: ", len(out))
	}
	return out
}

func m7Equal(t *testing.T, label string, p m7Produced, ref map[string][]byte) {
	t.Helper()
	for name, got := range map[string][]byte{"envelope": p.raw, "artifact": p.artifact, "controls": p.controls} {
		if string(got) != string(ref[name]) {
			t.Fatalf("%s: %s differs from the regular cold producer's bytes", label, name)
		}
	}
}

func m7LatencyB2(t *testing.T, dir, nativeDir string) map[string]any {
	t.Helper()
	reqs := m7LoadRequests(t, nativeDir, 24)
	timings := map[string]any{"cold_oneshot": []any{}, "worker": []any{}}
	norm := map[string]any{"requests": len(reqs)}
	// (a) one fresh instrumented process per task, 3 repetitions each (the very first run in the study is the OS/page-cache-coldest sample).
	for i, r := range reqs {
		for rep := 0; rep < 3; rep++ {
			p := m7Exec(t, "p2h", "oneshot", r.Praw, r.Phase, r.Alpha, r.Forecast, nil, filepath.Join(dir, "oneshot", fmt.Sprintf("r%02d-rep%d", i, rep)))
			m7Equal(t, fmt.Sprintf("oneshot r%d rep%d", i, rep), p, r.Ref)
			timings["cold_oneshot"] = append(timings["cold_oneshot"].([]any), map[string]any{"request": i, "rep": rep, "wall_seconds": p.seconds, "stages": p.stages})
		}
	}
	// (b) one long-lived worker; first request after start is the warm-up sample; 5 repetitions per request, interleaved order.
	w := &m7Worker{}
	w.start(t, false)
	timings["worker_boot"] = w.Boot
	for rep := 0; rep < 5; rep++ {
		for k := range reqs {
			i := (k*7 + rep*3) % len(reqs) // deterministic interleaving of different requests, sessions and phases
			r := reqs[i]
			p := m7ExecWith(t, w, r, filepath.Join(dir, "worker", fmt.Sprintf("r%02d-rep%d", i, rep)))
			m7Equal(t, fmt.Sprintf("worker r%d rep%d", i, rep), p, r.Ref)
			timings["worker"] = append(timings["worker"].([]any), map[string]any{"request": i, "rep": rep, "order": k, "wall_seconds": p.seconds, "stages": p.stages})
		}
	}
	w.stop()
	norm["outputs_identical_to_regular_cold_producer"] = true
	norm["oneshot_runs"], norm["worker_runs"] = len(timings["cold_oneshot"].([]any)), len(timings["worker"].([]any))
	// (c) bounded crash / timeout recovery and failure isolation.
	rec := &m7Worker{}
	rec.start(t, true)
	if _, err := rec.call([]byte(`{"op":"selftest_crash"}`), 20*time.Second); err == nil || !strings.Contains(err.Error(), "died") {
		t.Fatal("a worker crash must be reported, got ", err)
	}
	rec.restart(t, true)
	ok := m7ExecWith(t, rec, reqs[0], filepath.Join(dir, "recovery", "after-crash"))
	m7Equal(t, "after crash restart", ok, reqs[0].Ref)
	start := time.Now()
	if _, err := rec.call([]byte(`{"op":"selftest_hang"}`), 3*time.Second); err == nil || !strings.Contains(err.Error(), "timeout") {
		t.Fatal("a hung worker must time out, got ", err)
	}
	norm["hang_detected_within_bound"] = time.Since(start) < 10*time.Second
	rec.restart(t, true)
	bad, _ := json.Marshal(map[string]any{"op": "produce", "kind": "p2h", "request": map[string]any{"projection_b64": "AAAA", "phase": 1}})
	resp, err := rec.call(bad, 20*time.Second)
	if err != nil || resp["ok"] != false {
		t.Fatal("an invalid request must be reported without killing the worker: ", err, resp)
	}
	again := m7ExecWith(t, rec, reqs[1], filepath.Join(dir, "recovery", "after-bad-request"))
	m7Equal(t, "after invalid request", again, reqs[1].Ref)
	norm["restarts_for_two_injected_faults"], norm["invalid_request_isolated"] = rec.Restarts, true
	rec.stop()
	// Per-request identity of the normalized outputs (timing-free): hashes of the reference bytes.
	hashes := []string{}
	for _, r := range reqs {
		hashes = append(hashes, bridgeHash(r.Ref["envelope"])+":"+bridgeHash(r.Ref["artifact"]))
	}
	norm["request_output_sha256"] = hashes
	return map[string]any{"normalized": norm, "timings": timings}
}

// m7ExecWith sends one captured request to an already running worker.
func m7ExecWith(t *testing.T, w *m7Worker, r m7Req, dir string) m7Produced {
	t.Helper()
	_ = os.MkdirAll(dir, 0700)
	hash := m7SourceHashes(t, "src/accelsched_kai/p2h_producer.py", append([]string{"src/accelsched_kai/p2h_worker_experiment.py"}, m7Deps...))
	request := map[string]any{"projection_b64": base64.StdEncoding.EncodeToString(r.Praw), "phase": r.Phase, "alpha": r.Alpha, "BF": m4bBF, "BP": m4bBP, "forecast": r.Forecast}
	raw, _ := json.Marshal(map[string]any{"op": "produce", "kind": "p2h", "request": request})
	start := time.Now()
	resp, err := w.call(raw, 90*time.Second)
	seconds := time.Since(start).Seconds()
	if err != nil || resp["ok"] != true {
		t.Fatal("worker produce failed: ", err, resp)
	}
	get := func(n string) []byte { b, _ := base64.StdEncoding.DecodeString(resp[n].(string)); return b }
	return m7Produced{raw: get("envelope"), artifact: get("artifact"), controls: get("controls"), execution: get("execution"), envelope: bridgeHash(get("envelope")),
		seconds: seconds, producerHash: hash, stages: resp["timings"].(map[string]any)}
}

// TestM7LatencyStandalone re-runs only B2 against an existing native evidence directory (development aid;
// skipped unless M7_NATIVE_DIR is set).
func TestM7LatencyStandalone(t *testing.T) {
	native := os.Getenv("M7_NATIVE_DIR")
	if native == "" {
		t.Skip("M7_NATIVE_DIR not set")
	}
	out := m7LatencyB2(t, t.TempDir(), native)
	raw, _ := json.Marshal(out["normalized"])
	t.Log("latency_normalized ", string(raw))
}
