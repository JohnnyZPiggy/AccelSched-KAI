package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/common_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/pod_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/framework"
)

func taskEvent() *framework.Event {
	return &framework.Event{Task: &pod_info.PodInfo{UID: "T0", NodeName: "N0", Name: "task", Namespace: "test"}}
}

func sealed(t *testing.T, c *capture) captureSnapshot {
	t.Helper()
	c.Close()
	s, err := c.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if s.Received != uint64(len(s.Events))+s.Invalid+s.Dropped+s.Contended {
		t.Fatal("lost failure denominator", s)
	}
	var seq uint64
	for _, r := range s.Events {
		if r.Seq <= seq {
			t.Fatal("duplicate/reordered sequence")
		}
		seq = r.Seq
	}
	if dir := os.Getenv("POC1B_EVIDENCE_DIR"); dir != "" {
		sid := sha256.Sum256([]byte(s.Session))
		dir = filepath.Join(dir, strings.ReplaceAll(t.Name(), "/", "_"), "capture-"+hex.EncodeToString(sid[:4]))
		if err = os.MkdirAll(dir, 0700); err != nil {
			t.Fatal(err)
		}
		data, err := json.MarshalIndent(s, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(filepath.Join(dir, "capture-audit.json"), append(data, '\n'), 0600); err != nil {
			t.Fatal(err)
		}
		tr, err := c.MarshalTrace()
		if s.Complete {
			if err != nil {
				t.Fatal(err)
			}
			if err = os.WriteFile(filepath.Join(dir, "generated-trace.json"), tr, 0600); err != nil {
				t.Fatal(err)
			}
		} else if err == nil || tr != nil {
			t.Fatal("loss trace exported")
		}
	}
	return s
}

func TestCaptureG1Lifecycle(t *testing.T) {
	p := newShadowPlugin(nil).(*shadowPlugin)
	a, b := &framework.Session{ID: "session-A"}, &framework.Session{ID: "session-B"}
	p.OnSessionOpen(a)
	p.OnSessionOpen(b)
	ca, cb := p.sessionCapture(a), p.sessionCapture(b)
	if ca == nil || cb == nil || ca == cb {
		t.Fatal("sessions share capture")
	}
	for _, pair := range []struct {
		s *framework.Session
		c *capture
	}{{a, ca}, {b, cb}} {
		// Observe the actual registered pointer without unsafe access, changing the
		// Session or calling native OpenSession/Run. Handler invocation uses this
		// same stored pointer, not a second fake callback implementation.
		v := reflect.ValueOf(pair.s).Elem()
		handlers := v.FieldByName("eventHandlers")
		if handlers.Len() != 1 || handlers.Index(0).Pointer() != reflect.ValueOf(pair.c.handler).Pointer() {
			t.Fatal("wrong registered handler")
		}
		if pair.c.handler.AllocateFunc == nil || pair.c.handler.DeallocateFunc == nil {
			t.Fatal("missing virtual callbacks")
		}
		for i := 0; i < v.NumField(); i++ {
			f := v.Field(i)
			if v.Type().Field(i).Name != "ID" && f.CanInterface() && !f.IsZero() {
				t.Fatalf("mutated Session field %s", v.Type().Field(i).Name)
			}
		}
	}
	ca.handler.AllocateFunc(taskEvent())
	cb.handler.DeallocateFunc(taskEvent())
	p.OnSessionClose(a)
	p.OnSessionClose(b)
	if p.sessionCapture(a) != nil || p.sessionCapture(b) != nil || len(p.active) != 0 {
		t.Fatal("closed session retained")
	}
	sa, sb := sealed(t, ca), sealed(t, cb)
	if sa.Session == sb.Session || len(sa.Events) != 1 || len(sb.Events) != 1 || sa.Events[0].Kind == sb.Events[0].Kind {
		t.Fatal("session mix")
	}
	ca.handler.AllocateFunc(taskEvent())
	ca.handler.DeallocateFunc(nil)
	ca.Close()
	after, _ := ca.Snapshot()
	if !reflect.DeepEqual(sa, after) {
		t.Fatal("post-close writes")
	}
	// Returned snapshots cannot mutate capture state.
	sa.Events[0].Task = "tampered"
	again, _ := ca.Snapshot()
	if again.Events[0].Task != "T0" {
		t.Fatal("snapshot aliases storage")
	}
	t.Run("duplicate_open_fails_closed", func(t *testing.T) {
		p := newShadowPlugin(nil).(*shadowPlugin)
		s := &framework.Session{ID: "duplicate"}
		p.OnSessionOpen(s)
		c := p.sessionCapture(s)
		p.OnSessionOpen(s)
		if reflect.ValueOf(s).Elem().FieldByName("eventHandlers").Len() != 1 {
			t.Fatal("duplicate handler")
		}
		p.OnSessionClose(s)
		audit := sealed(t, c)
		if audit.Complete || audit.LifecycleErrors != 1 {
			t.Fatal(audit)
		}
	})
}

func TestCaptureG2VirtualReadOnly(t *testing.T) {
	c := newCapture("S1")
	e := taskEvent()
	before, err := json.Marshal(e.Task)
	if err != nil {
		t.Fatal(err)
	}
	c.handler.AllocateFunc(e)
	after, err := json.Marshal(e.Task)
	if err != nil || !bytes.Equal(before, after) {
		t.Fatal("source mutated", err)
	}
	// Mutating/replacing the source after callback must not alter saved records.
	e.Task.UID = "T1"
	e.Task.NodeName = "N1"
	c.handler.DeallocateFunc(e)
	e.Task.Name = "later"
	e.Task.NodeName = "later"
	e.Task = nil
	s := sealed(t, c)
	if !s.Complete || len(s.Events) != 2 || s.Events[0].Task != "T0" || s.Events[0].Node != "N0" || s.Events[1].Task != "T1" || s.Events[1].Node != "N1" {
		t.Fatal(s)
	}
	for _, name := range []string{"nil_event", "nil_task", "blank_uid", "blank_node", "long_uid", "invalid_utf8", "invalid_session"} {
		t.Run(name, func(t *testing.T) {
			c := newCapture("S1")
			e := taskEvent()
			switch name {
			case "nil_event":
				e = nil
			case "nil_task":
				e.Task = nil
			case "blank_uid":
				e.Task.UID = " "
			case "blank_node":
				e.Task.NodeName = ""
			case "long_uid":
				e.Task.UID = common_info.PodID(strings.Repeat("t", 129))
			case "invalid_utf8":
				e.Task.NodeName = string([]byte{0xff})
			case "invalid_session":
				c = newCapture(" ")
			}
			c.handler.AllocateFunc(e)
			s := sealed(t, c)
			if s.Complete || s.Invalid != 1 || len(s.Events) != 0 {
				t.Fatal(s)
			}
			if b, err := c.MarshalTrace(); err == nil || b != nil {
				t.Fatal("invalid capture exported")
			}
		})
	}
	t.Run("no_renumber_after_invalid", func(t *testing.T) {
		c := newCapture("S1")
		c.handler.AllocateFunc(taskEvent())
		c.handler.AllocateFunc(nil)
		c.handler.DeallocateFunc(taskEvent())
		s := sealed(t, c)
		if len(s.Events) != 2 || s.Events[1].Seq != 3 || s.Invalid != 1 || s.Complete {
			t.Fatal(s)
		}
	})
}

func TestCaptureG3StrictTrace(t *testing.T) {
	c := newCapture("S1")
	if _, err := c.MarshalTrace(); err == nil {
		t.Fatal("exported live capture")
	}
	c.handler.AllocateFunc(taskEvent())
	c.handler.DeallocateFunc(taskEvent())
	s := sealed(t, c)
	if !s.Complete {
		t.Fatal(s)
	}
	b, err := c.MarshalTrace()
	if err != nil {
		t.Fatal(err)
	}
	again, _ := c.MarshalTrace()
	if !bytes.Equal(b, again) {
		t.Fatal("nondeterministic trace")
	}
	var top map[string]json.RawMessage
	if err = json.Unmarshal(b, &top); err != nil {
		t.Fatal(err)
	}
	if len(top) != 6 {
		t.Fatal("extra/missing top-level key")
	}
	var tr captureTrace
	if err = json.Unmarshal(b, &tr); err != nil {
		t.Fatal(err)
	}
	if tr.SchemaVersion != "0.1" || tr.Mode != "SAVED_UNVERIFIED" || tr.Authority != "UNVERIFIED" || tr.KAIRef != captureKAIRef {
		t.Fatal(tr)
	}
	var rows []map[string]json.RawMessage
	_ = json.Unmarshal(top["events"], &rows)
	for i, row := range rows {
		if len(row) != 11 {
			t.Fatal("extra/missing event key")
		}
		for _, key := range []string{"statement_id", "checkpoint_id", "operation_index", "range_start", "range_end", "bind_request_id"} {
			if string(row[key]) != "null" {
				t.Fatal("invented transaction attribution", key)
			}
		}
		if tr.Events[i].Evidence.Authority != "UNVERIFIED" || tr.Events[i].Evidence.Source != "EVENT_HANDLER" {
			t.Fatal("authority promotion")
		}
	}
	// Independent fixed canonical vector, not computed using integrityRecord.
	canonical := `{"schema_version":"0.1","kai_ref":"374cbb796ecff07c4a9140e59286f3e3a280e84b","session_id":"S1","seq":1,"kind":"ALLOCATE_OBSERVED","task_id":"T0","node_id":"N0"}`
	h := sha256.Sum256([]byte(canonical))
	expected := hex.EncodeToString(h[:])
	if tr.Events[0].Evidence.SHA256 != expected || !strings.HasSuffix(tr.Events[0].Evidence.Locator, expected) {
		t.Fatal("false integrity hash")
	}
	// Long but valid Session ID must not make locator exceed frozen schema bounds.
	long := newCapture(strings.Repeat("s", 128))
	long.handler.AllocateFunc(taskEvent())
	long.Close()
	lb, err := long.MarshalTrace()
	if err != nil {
		t.Fatal(err)
	}
	var lt captureTrace
	_ = json.Unmarshal(lb, &lt)
	if !validIdentity(lt.Events[0].Evidence.Locator) {
		t.Fatal("invalid generated locator")
	}
}

func TestCaptureG4Overflow(t *testing.T) {
	for _, n := range []int{256, 257} {
		t.Run(map[int]string{256: "full_loss_free", 257: "tail_loss"}[n], func(t *testing.T) {
			c := newCapture("capacity")
			for i := 0; i < n; i++ {
				c.handler.AllocateFunc(taskEvent())
			}
			s := sealed(t, c)
			if s.Capacity != 256 || len(s.Events) != 256 || s.Received != uint64(n) || s.Dropped != uint64(n-256) || s.Events[255].Seq != 256 {
				t.Fatal(s)
			}
			b, err := c.MarshalTrace()
			if n == 256 && (err != nil || b == nil || !s.Complete) {
				t.Fatal(err)
			}
			if n == 257 && (err == nil || b != nil || s.Complete) {
				t.Fatal("tail loss hidden")
			}
		})
	}
}

func TestCaptureG5ConcurrencyClose(t *testing.T) {
	t.Run("nonblocking_contention", func(t *testing.T) {
		c := newCapture("contention")
		c.mu.Lock()
		done := make(chan struct{})
		go func() { c.handler.AllocateFunc(taskEvent()); close(done) }()
		select {
		case <-done:
		case <-time.After(time.Second):
			c.mu.Unlock()
			t.Fatal("callback blocked")
		}
		c.mu.Unlock()
		s := sealed(t, c)
		if s.Complete || s.Contended != 1 || len(s.Events) != 0 {
			t.Fatal(s)
		}
	})
	t.Run("concurrent_sequences", func(t *testing.T) {
		c := newCapture("parallel")
		c.handler.AllocateFunc(taskEvent())
		var wg sync.WaitGroup
		start := make(chan struct{})
		for i := 0; i < 8; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				<-start
				for j := 0; j < 16; j++ {
					c.handler.AllocateFunc(taskEvent())
				}
			}()
		}
		close(start)
		wg.Wait()
		s := sealed(t, c)
		if s.Received != 129 || len(s.Events) < 1 || len(s.Events) > 129 || s.Invalid != 0 || s.Dropped != 0 {
			t.Fatal(s)
		}
		if s.Contended > 0 && s.Complete {
			t.Fatal("contention loss hidden")
		}
	})
	t.Run("close_race_and_post_close_stability", func(t *testing.T) {
		c := newCapture("close-race")
		c.handler.AllocateFunc(taskEvent())
		var wg sync.WaitGroup
		start := make(chan struct{})
		for i := 0; i < 8; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				<-start
				for j := 0; j < 64; j++ {
					c.handler.AllocateFunc(taskEvent())
				}
			}()
		}
		wg.Add(1)
		go func() { defer wg.Done(); <-start; c.Close() }()
		close(start)
		wg.Wait()
		before := sealed(t, c)
		for i := 0; i < 100; i++ {
			c.handler.AllocateFunc(taskEvent())
			c.handler.DeallocateFunc(nil)
		}
		after, err := c.Snapshot()
		if err != nil || !reflect.DeepEqual(before, after) || !after.Closed {
			t.Fatal("post-close mutation", err)
		}
	})
}

func TestCaptureG6FrozenOfflineHandoff(t *testing.T) {
	// Build ONLY the frozen standard-library offline CLI. This executable is NOT
	// the custom KAI scheduler. No RunApp/Run/OpenSession/client/server is invoked.
	dir := t.TempDir()
	suffix := ""
	if runtime.GOOS == "windows" {
		suffix = ".exe"
	}
	goTool := filepath.Join(runtime.GOROOT(), "bin", "go"+suffix)
	cli := filepath.Join(dir, "poc1-offline-observer"+suffix)
	build := exec.Command(goTool, "build", "-o", cli, ".")
	build.Dir = "../poc1"
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("frozen CLI build: %v: %s", err, out)
	}
	fixture, err := filepath.Abs("../fixtures/01_gpu_family.json")
	if err != nil {
		t.Fatal(err)
	}
	original, err := os.ReadFile(fixture)
	if err != nil {
		t.Fatal(err)
	}
	for _, session := range []string{"S1", strings.Repeat("s", 128)} {
		t.Run("session_length_"+map[int]string{2: "2", 128: "128"}[len(session)], func(t *testing.T) {
			c := newCapture(session)
			c.handler.AllocateFunc(taskEvent())
			c.handler.DeallocateFunc(taskEvent())
			audit := sealed(t, c)
			if !audit.Complete {
				t.Fatal(audit)
			}
			tr, err := c.MarshalTrace()
			if err != nil {
				t.Fatal(err)
			}
			work := t.TempDir()
			input := filepath.Join(work, "actual-generated-trace.json")
			outdir := filepath.Join(work, "offline-result")
			if err = os.WriteFile(input, tr, 0600); err != nil {
				t.Fatal(err)
			}
			cmd := exec.Command(cli, "-trace", input, "-snapshot", fixture, "-out", outdir)
			cmd.Dir = work
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("frozen CLI: %v: %s", err, out)
			}
			copied, err := os.ReadFile(filepath.Join(outdir, "snapshot.json"))
			if err != nil || !bytes.Equal(copied, original) {
				t.Fatal("snapshot modified", err)
			}
			data, err := os.ReadFile(filepath.Join(outdir, "trace-result.json"))
			if err != nil {
				t.Fatal(err)
			}
			var result struct {
				Status       string `json:"status"`
				Mode         string `json:"mode"`
				Authority    string `json:"authority"`
				Commit       bool   `json:"real_commit_confirmed"`
				Bind         bool   `json:"real_bind_confirmed"`
				Observations []struct {
					Association string `json:"association"`
					State       string `json:"state"`
				} `json:"observations"`
				Statements map[string]any `json:"statements"`
				Bindings   map[string]any `json:"bindings"`
			}
			if err = json.Unmarshal(data, &result); err != nil {
				t.Fatal(err)
			}
			if result.Status != "TRACE_INCOMPLETE" || result.Mode != "SAVED_UNVERIFIED" || result.Authority != "UNVERIFIED" || result.Commit || result.Bind || len(result.Statements) != 0 || len(result.Bindings) != 0 || len(result.Observations) != 2 {
				t.Fatalf("unsafe offline interpretation: %s", data)
			}
			for _, o := range result.Observations {
				if o.Association != "UNKNOWN" || o.State != "VIRTUAL_OBSERVED" {
					t.Fatal(o)
				}
			}
			if root := os.Getenv("POC1B_EVIDENCE_DIR"); root != "" {
				saved := filepath.Join(root, strings.ReplaceAll(t.Name(), "/", "_"))
				if err = os.MkdirAll(saved, 0700); err != nil {
					t.Fatal(err)
				}
				for _, name := range []string{"snapshot.json", "trace-result.json", "manifest.json"} {
					b, err := os.ReadFile(filepath.Join(outdir, name))
					if err != nil {
						t.Fatal(err)
					}
					if err = os.WriteFile(filepath.Join(saved, name), b, 0600); err != nil {
						t.Fatal(err)
					}
				}
			}
		})
	}
}
