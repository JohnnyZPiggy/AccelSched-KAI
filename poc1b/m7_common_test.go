package main

// M7 shared test machinery. NOTHING here touches a frozen contract: the P1/P2/P2H receivers and
// producers are reused as they are. This file adds
//   - a separately versioned planning-cohort receiver (P2H-PC-1) for the controlled M7-B1 chain,
//   - an experimental long-lived Python worker transport (M7-B2; never wired to a plugin), and
//   - a parameterized arm runner (lambda, alpha, forecast variant, cohort source, transport).
// Controlled local test harness; simulated Fake-Session GPUs; not a production path.

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/actions/allocate"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/node_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/pod_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/pod_status"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/framework"
)

// ---- P2H-PC-1 receiver (separately versioned; the P2H receiver is untouched) -----------------------

const pcVersion = "P2H-PC-1"

var pcSources = map[string]bool{"PUBLISHED_QUEUE_LOOKAHEAD": true, "DECLARED_FORECAST": true, "OFFLINE_ORACLE_INFORMATION_NOT_NATIVE": true}

type pcGrant struct {
	p2hGrant
	cohortHash, cohortSource string
	cohortIDs                []string
}

type pcEnvelope struct {
	p2hEnvelope
	CohortHash   string   `json:"cohort_sha256"`
	CohortSource string   `json:"cohort_source"`
	CohortIDs    []string `json:"cohort_ids"`
}

func decodePC(raw []byte, g *pcGrant) (pcEnvelope, error) {
	var v pcEnvelope
	if g == nil || len(raw) == 0 || len(raw) > 131072 || !bytes.Equal(bytes.TrimSpace(raw), raw) || bridgeHash(raw) != g.envelopeHash {
		return v, fmt.Errorf("no controlled canonical handoff or byte mismatch")
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	d.UseNumber()
	if err := bridgeUnique(d); err != nil {
		return v, err
	}
	if _, err := d.Token(); err != io.EOF {
		return v, fmt.Errorf("trailing JSON")
	}
	d = json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	if err := d.Decode(&v); err != nil {
		return v, err
	}
	if err := bridgeStringFields(raw, []string{"version", "producer", "producer_sha256", "artifact_sha256", "artifact_b64", "prediction_sha256", "projection_sha256", "session", "task_uid", "cohort_sha256", "cohort_source"}, 16); err != nil {
		return v, err
	}
	var object map[string]json.RawMessage
	_ = json.Unmarshal(raw, &object)
	for _, key := range []string{"candidates", "cohort_ids"} {
		var items []json.RawMessage
		if json.Unmarshal(object[key], &items) != nil {
			return v, fmt.Errorf("%s array required", key)
		}
		for _, it := range items {
			if len(it) == 0 || it[0] != '"' {
				return v, fmt.Errorf("exact string required in %s", key)
			}
		}
	}
	for key, names := range map[string][]string{"nodes": {"name", "uid"}, "pairs": {"task_uid", "node", "score_bits"}} {
		var children []json.RawMessage
		if json.Unmarshal(object[key], &children) != nil {
			return v, fmt.Errorf("array required")
		}
		for _, child := range children {
			if err := bridgeStringFields(child, names, len(names)); err != nil {
				return v, err
			}
		}
	}
	if v.Version != pcVersion || v.Producer != "SAE_CONTROLLED_OFFLINE_TEST_HARNESS" || v.ProducerHash != g.producerHash || v.Projection != g.projectionHash ||
		v.Phase != g.phase || v.Phase < 1 || v.Phase > p2hMaxPhase || v.Session != g.owner.ID || v.Task != string(g.task.UID) {
		return v, fmt.Errorf("provenance/version/phase mismatch")
	}
	if !pcSources[v.CohortSource] || v.CohortSource != g.cohortSource || v.CohortHash != g.cohortHash || !reflect.DeepEqual(v.CohortIDs, g.cohortIDs) {
		return v, fmt.Errorf("cohort provenance (source/hash/members) differs from the declared cohort")
	}
	artifact, err := base64.StdEncoding.Strict().DecodeString(v.Artifact)
	if err != nil || base64.StdEncoding.EncodeToString(artifact) != v.Artifact || bridgeHash(artifact) != v.ArtifactHash {
		return v, fmt.Errorf("artifact byte digest mismatch")
	}
	var a struct {
		Version    string `json:"version"`
		Status     string `json:"status"`
		Prediction string `json:"prediction_sha256"`
		Pairs      []struct {
			Task  string  `json:"task_id"`
			Node  string  `json:"node_id"`
			Score float64 `json:"score"`
		} `json:"pairs"`
	}
	n := len(g.candidates)
	if json.Unmarshal(artifact, &a) != nil || a.Version != "M3A-1" || a.Status != "CERTIFIED_OFFLINE_ADVICE" || a.Prediction != v.Prediction ||
		n < 1 || n > p2hMaxNodes || len(v.Pairs) != n || len(v.Candidates) != n || len(v.Nodes) != len(g.names) {
		return v, fmt.Errorf("artifact coverage/status")
	}
	if !reflect.DeepEqual(v.Candidates, g.candidates) {
		return v, fmt.Errorf("candidate set/order differs from the native candidate list")
	}
	allowed := map[string]bool{v.Task: true}
	for _, id := range g.cohortIDs {
		allowed[id] = true
	}
	mine := 0
	for _, p := range a.Pairs {
		if !allowed[p.Task] {
			return v, fmt.Errorf("artifact scores a task outside target+declared cohort")
		}
		if p.Task == v.Task {
			if mine >= n || p.Node != g.candidates[mine] {
				return v, fmt.Errorf("target pairs differ from the native candidate set")
			}
			mine++
		}
	}
	if mine != n {
		return v, fmt.Errorf("target pair count")
	}
	for i, nd := range v.Nodes {
		actual := g.nodes[g.names[i]]
		if nd.Name != g.names[i] || actual == nil || actual.Node == nil || nd.UID != string(actual.Node.UID) {
			return v, fmt.Errorf("node identity/order")
		}
	}
	k := 0
	for _, ap := range a.Pairs {
		if ap.Task != v.Task {
			continue
		}
		p := v.Pairs[k]
		bits, e := strconv.ParseUint(p.Bits, 16, 64)
		score := math.Float64frombits(bits)
		if e != nil || len(p.Bits) != 16 || fmt.Sprintf("%016x", bits) != p.Bits || !finiteAdvice(score) || score < -1 || score > 1 ||
			p.Task != v.Task || p.Node != v.Candidates[k] || ap.Node != p.Node || math.Float64bits(ap.Score) != bits {
			return v, fmt.Errorf("pair identity/score mismatch")
		}
		k++
	}
	return v, nil
}

type pcBridge struct {
	mu                  sync.RWMutex
	owner               *framework.Session
	record              *adviceRecord
	phaseActive, closed bool
	target              *pod_info.PodInfo
	nodes               map[string]*node_info.NodeInfo
	lastPhase           int
}

func (p *pcBridge) OnSessionOpen(s *framework.Session) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if s == nil || p.owner != nil || p.closed {
		return
	}
	p.owner = s
	p.record = makeAdviceRecord(s, nil)
	s.AddNodeOrderFn(p.score)
}

func (p *pcBridge) score(t *pod_info.PodInfo, n *node_info.NodeInfo) (float64, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	if p.closed || !p.phaseActive || t == nil || n == nil || t.Pod == nil || p.target == nil || t.UID != p.target.UID || t.Job != p.target.Job || t.Pod.UID != p.target.Pod.UID || p.nodes[n.Name] == nil || n.Node != p.nodes[n.Name].Node {
		return 0, nil
	}
	return p.record.score(t, n)
}

func (p *pcBridge) begin(raw []byte, g *pcGrant, lambda float64) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.phaseActive = false
	if p.closed || g == nil || g.used || g.owner != p.owner || g.project == nil || g.phase <= p.lastPhase || g.phase > p2hMaxPhase || !finiteAdvice(lambda) || lambda < 0 || lambda > adviceLambdaLimit {
		return false
	}
	g.used = true
	current, err := g.project(p.owner)
	if err != nil || bridgeHash(current) != g.projectionHash {
		return false
	}
	v, err := decodePC(raw, g)
	if err != nil || g.owner.ClusterInfo == nil || len(g.nodes) != len(g.names) {
		return false
	}
	for name, n := range g.nodes {
		if g.owner.ClusterInfo.Nodes[name] != n {
			return false
		}
	}
	found := false
	for _, j := range g.owner.ClusterInfo.PodGroupInfos {
		for _, t := range j.GetAllPodsMap() {
			if t == g.task {
				found = true
			}
		}
	}
	if !found {
		return false
	}
	snapshot := adviceSnapshot{Version: 1, Session: v.Session, Lambda: lambda}
	for _, pair := range v.Pairs {
		bits, _ := strconv.ParseUint(pair.Bits, 16, 64)
		snapshot.Entries = append(snapshot.Entries, adviceValue{pair.Task, pair.Node, math.Float64frombits(bits)})
	}
	if p.record != nil {
		p.record.metadata.close()
	}
	p.record = makeAdviceRecord(p.owner, &snapshot)
	p.target = g.task
	p.nodes = map[string]*node_info.NodeInfo{}
	for name, n := range g.nodes {
		p.nodes[name] = n
	}
	p.phaseActive = p.record.active
	if p.phaseActive {
		p.lastPhase = g.phase
	}
	return p.phaseActive
}

func (p *pcBridge) endPhase() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.phaseActive = false
	if p.record != nil {
		p.record.metadata.close()
	}
}

func (p *pcBridge) OnSessionClose(s *framework.Session) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.owner == s {
		p.phaseActive, p.closed = false, true
		if p.record != nil {
			p.record.metadata.close()
		}
		p.target, p.nodes = nil, nil
	}
}

// ---- cohorts --------------------------------------------------------------------------------------------

type m7Cohort struct {
	Doc     map[string]any
	Hash    string
	IDs     []string
	Source  string
	Members []map[string]any
}

func m7MakeCohort(source string, arrival, phase int, demands [][]int, prefix string) m7Cohort {
	members := []map[string]any{}
	ids := []string{}
	for k, d := range demands {
		id := fmt.Sprintf("%s:%d", prefix, k)
		members = append(members, map[string]any{"id": id, "demand": d})
		ids = append(ids, id)
	}
	doc := map[string]any{"source": source, "as_of_arrival": arrival, "as_of_phase": phase, "horizon": len(members), "members": members}
	raw, _ := json.Marshal(doc)
	return m7Cohort{Doc: doc, Hash: bridgeHash(raw), IDs: ids, Source: source, Members: members}
}

// m7ForecastCohort: the immutable declared forecast read as expected pending demand - each forecast item is
// replicated by its weight (capped at 8). It depends on nothing but the scene's declared forecast.
func m7ForecastCohort(sc m6Scene, arrival, phase int) m7Cohort {
	demands := [][]int{}
	for _, f := range sc.Forecast {
		for k := 0; k < f.Weight && len(demands) < 8; k++ {
			demands = append(demands, f.Demand)
		}
	}
	return m7MakeCohort("DECLARED_FORECAST", arrival, phase, demands, "plan")
}

// m7OracleCohort uses the HELD-OUT realized future: diagnostic only, never a deployable information source.
func m7OracleCohort(sc m6Scene, arrival, phase int) m7Cohort {
	demands := [][]int{}
	for _, j := range sc.Future {
		demands = append(demands, []int{j.GPUs, j.Cores, j.GiB})
	}
	return m7MakeCohort("OFFLINE_ORACLE_INFORMATION_NOT_NATIVE", arrival, phase, demands, "oracle")
}

// ---- producer transports -----------------------------------------------------------------------------------

type m7Worker struct {
	mu       sync.Mutex
	cmd      *exec.Cmd
	in       io.WriteCloser
	out      *bufio.Reader
	Restarts int
	Boot     map[string]any
}

func (w *m7Worker) start(t *testing.T, selftest bool) {
	t.Helper()
	root, _ := filepath.Abs("..")
	python := os.Getenv("M3B_PYTHON")
	code := "import runpy,sys;sys.path.insert(0,sys.argv[1]);runpy.run_module('accelsched_kai.p2h_worker_experiment',run_name='__main__')"
	cmd := exec.Command(python, "-I", "-B", "-c", code, filepath.Join(root, "src"))
	cmd.Env = append(os.Environ(), "M7_SPAWN_EPOCH_NS="+strconv.FormatInt(time.Now().UnixNano(), 10))
	if selftest {
		cmd.Env = append(cmd.Env, "M7_WORKER_SELFTEST=1")
	}
	in, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err = cmd.Start(); err != nil {
		t.Fatal(err)
	}
	w.cmd, w.in, w.out = cmd, in, bufio.NewReaderSize(out, 1<<22)
	line, err := w.out.ReadBytes('\n')
	if err != nil {
		t.Fatal("worker did not become ready: ", err)
	}
	var ready struct {
		Op   string         `json:"op"`
		Boot map[string]any `json:"boot"`
	}
	if json.Unmarshal(line, &ready) != nil || ready.Op != "ready" {
		t.Fatal("worker ready line: ", string(line))
	}
	w.Boot = ready.Boot
}

func (w *m7Worker) stop() {
	if w.cmd == nil {
		return
	}
	_ = w.in.Close()
	done := make(chan struct{})
	go func() { _ = w.cmd.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		_ = w.cmd.Process.Kill()
		<-done
	}
	w.cmd = nil
}

// call sends one request line and waits (bounded) for one response line. A timeout or crash kills the worker;
// the caller decides whether to restart. No response is ever reused across requests.
func (w *m7Worker) call(raw []byte, timeout time.Duration) (map[string]any, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.cmd == nil {
		return nil, fmt.Errorf("worker not running")
	}
	if _, err := w.in.Write(append(raw, '\n')); err != nil {
		return nil, err
	}
	type result struct {
		line []byte
		err  error
	}
	ch := make(chan result, 1)
	go func() { l, e := w.out.ReadBytes('\n'); ch <- result{l, e} }()
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	select {
	case r := <-ch:
		if r.err != nil {
			_ = w.cmd.Process.Kill()
			_ = w.cmd.Wait()
			w.cmd = nil
			return nil, fmt.Errorf("worker died: %w", r.err)
		}
		var resp map[string]any
		if err := json.Unmarshal(r.line, &resp); err != nil {
			return nil, err
		}
		return resp, nil
	case <-ctx.Done():
		_ = w.cmd.Process.Kill()
		_ = w.cmd.Wait()
		w.cmd = nil
		return nil, fmt.Errorf("worker timeout")
	}
}

func (w *m7Worker) restart(t *testing.T, selftest bool) {
	t.Helper()
	w.stop()
	w.Restarts++
	w.start(t, selftest)
}

var m7Shared *m7Worker

func m7WorkerOrStart(t *testing.T) *m7Worker {
	t.Helper()
	if m7Shared == nil || m7Shared.cmd == nil {
		if m7Shared == nil {
			m7Shared = &m7Worker{}
		}
		m7Shared.start(t, false)
	}
	return m7Shared
}

type m7Produced struct {
	raw, artifact, controls, execution []byte
	envelope                           string
	seconds                            float64
	producerHash                       string
	stages                             map[string]any
}

func m7SourceHashes(t *testing.T, module string, deps []string) string {
	t.Helper()
	root, _ := filepath.Abs("..")
	var h string
	for _, name := range append([]string{module}, deps...) {
		b, err := os.ReadFile(filepath.Join(root, name))
		if err != nil {
			t.Fatal(err)
		}
		b = bytes.ReplaceAll(b, []byte("\r\n"), []byte("\n"))
		expected, err := exec.Command("git", "-C", root, "show", "HEAD:"+name).Output()
		if err != nil || !bytes.Equal(b, expected) {
			t.Fatal("uncommitted/mismatched controlled producer source ", name, err)
		}
		if name == module {
			h = bridgeHash(b)
		}
	}
	return h
}

var m7Deps = []string{"src/accelsched_kai/bounded_advice.py", "src/accelsched_kai/adapter.py", "src/accelsched_kai/oracle.py", "src/accelsched_kai/advisor.py", "src/accelsched_kai/schema.py", "src/accelsched_kai/p2h_producer.py"}

// m7Exec produces the envelope for projection bytes via the chosen transport: "cold" (the regular one-process
// producer), "oneshot" (the same code, instrumented, one process per task) or "worker" (long-lived, experiment only).
func m7Exec(t *testing.T, kind, transport string, praw []byte, phase int, alpha float64, forecast []m4bForecast, cohort *m7Cohort, dir string) m7Produced {
	t.Helper()
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	module := "src/accelsched_kai/p2h_producer.py"
	if kind == "p2hpc" {
		module = "src/accelsched_kai/p2hpc_producer.py"
	}
	hash := m7SourceHashes(t, module, m7Deps)
	if transport != "cold" {
		hash = m7SourceHashes(t, module, append([]string{"src/accelsched_kai/p2h_worker_experiment.py"}, m7Deps...))
	}
	request := map[string]any{"projection_b64": base64.StdEncoding.EncodeToString(praw), "phase": phase, "alpha": alpha, "BF": m4bBF, "BP": m4bBP, "forecast": forecast}
	if cohort != nil {
		request["cohort"] = cohort.Doc
	}
	names := []string{"envelope", "artifact", "snapshot", "execution", "controls"}
	var out [5][]byte
	var seconds float64
	stages := map[string]any{}
	root, _ := filepath.Abs("..")
	python := os.Getenv("M3B_PYTHON")
	switch transport {
	case "worker":
		raw, _ := json.Marshal(map[string]any{"op": "produce", "kind": kind, "request": request})
		w := m7WorkerOrStart(t)
		start := time.Now()
		resp, err := w.call(raw, 90*time.Second)
		seconds = time.Since(start).Seconds()
		if err != nil || resp["ok"] != true {
			t.Fatal("worker produce failed: ", err, resp)
		}
		for i, n := range names {
			b, derr := base64.StdEncoding.DecodeString(resp[n].(string))
			if derr != nil {
				t.Fatal(derr)
			}
			out[i] = b
		}
		stages = resp["timings"].(map[string]any)
	case "oneshot", "cold":
		reqRaw, _ := json.Marshal(request)
		input := filepath.Join(dir, "request.json")
		if err := os.WriteFile(input, reqRaw, 0600); err != nil {
			t.Fatal(err)
		}
		output := filepath.Join(dir, "producer")
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()
		var cmd *exec.Cmd
		if transport == "cold" {
			mod := "accelsched_kai.p2h_producer"
			if kind == "p2hpc" {
				mod = "accelsched_kai.p2hpc_producer"
			}
			code := "import runpy,sys;sys.path.insert(0,sys.argv[1]);sys.argv=sys.argv[2:];runpy.run_module('" + mod + "',run_name='__main__')"
			cmd = exec.CommandContext(ctx, python, "-I", "-B", "-c", code, filepath.Join(root, "src"), "producer", input, output)
		} else {
			wrap, _ := json.Marshal(map[string]any{"kind": kind, "request": request})
			input = filepath.Join(dir, "oneshot-request.json")
			_ = os.WriteFile(input, wrap, 0600)
			code := "import runpy,sys;sys.path.insert(0,sys.argv[1]);sys.argv=sys.argv[2:];runpy.run_module('accelsched_kai.p2h_worker_experiment',run_name='__main__')"
			cmd = exec.CommandContext(ctx, python, "-I", "-B", "-c", code, filepath.Join(root, "src"), "worker", "--oneshot", input, output)
			cmd.Env = append(os.Environ(), "M7_SPAWN_EPOCH_NS="+strconv.FormatInt(time.Now().UnixNano(), 10))
		}
		var stderr bytes.Buffer
		cmd.Stderr = &stderr
		start := time.Now()
		stdout, err := cmd.Output()
		seconds = time.Since(start).Seconds()
		_ = os.WriteFile(filepath.Join(dir, "producer.stderr.txt"), stderr.Bytes(), 0600)
		if err != nil {
			t.Fatal("controlled producer failed ", err, stderr.String())
		}
		for i, n := range names {
			b, rerr := os.ReadFile(filepath.Join(output, n+".json"))
			if rerr != nil {
				t.Fatal(rerr)
			}
			out[i] = b
		}
		if !bytes.Equal(out[0], stdout) {
			t.Fatal("producer handoff mismatch")
		}
		if transport == "oneshot" {
			tb, _ := os.ReadFile(filepath.Join(output, "timings.json"))
			_ = json.Unmarshal(tb, &stages)
		}
	default:
		t.Fatal("unknown transport ", transport)
	}
	for i, n := range names {
		_ = os.WriteFile(filepath.Join(dir, n+".json"), out[i], 0600)
	}
	return m7Produced{raw: out[0], artifact: out[1], controls: out[4], execution: out[3], envelope: bridgeHash(out[0]), seconds: seconds, producerHash: hash, stages: stages}
}

// ---- parameterized arm runner ----------------------------------------------------------------------------------

type m7Env struct {
	*m6Env
	pc *pcBridge
}

func m7Session(t *testing.T, id string, sc m6Scene) *m7Env {
	t.Helper()
	e := m6Session(t, id, sc.Nodes, m6Topology(sc.ID, sc.Nodes, sc.Arrivals[0], nil))
	pc := &pcBridge{}
	pc.OnSessionOpen(e.f.session) // inactive unless a P2H-PC-1 grant is begun; identical hook set in every M7 arm
	t.Cleanup(func() { pc.OnSessionClose(e.f.session) })
	return &m7Env{m6Env: e, pc: pc}
}

// m7Cfg fully declares an arm BEFORE it runs; nothing is chosen from outcomes.
type m7Cfg struct {
	Name      string
	Off       bool
	Alpha     float64
	Lambda    float64
	Kind      string // p2h | p2hpc
	Forecast  string // declared | accurate | clairvoyant
	Cohort    string // "" | forecast | oracle
	Transport string // cold | worker
	Label     string
}

func m7AccurateForecast(sc m6Scene) []m4bForecast {
	var r m6Regime
	for _, x := range m6Regimes {
		if x.Name == sc.Regime {
			r = x
		}
	}
	if r.Name == "" { // the hand-designed mechanism scene already declares a distributionally matched forecast
		return sc.Forecast
	}
	out := []m4bForecast{}
	for i := 0; i < 3; i++ {
		_, j := m6Draw(r.Realized, sc.ID, "accurate-forecast", i) // same distribution as the realized mix, independent draws
		out = append(out, m4bForecast{ID: fmt.Sprintf("F%d", i), Demand: []int{j.GPUs, j.Cores, j.GiB}, Weight: 1 + int(m6H(sc.ID, "accurate-weight", i)%3), Eligible: m6Names(4)})
	}
	return out
}

func m7ClairvoyantForecast(sc m6Scene) []m4bForecast {
	out := []m4bForecast{}
	for i, j := range sc.Future {
		out = append(out, m4bForecast{ID: fmt.Sprintf("F%d", i), Demand: []int{j.GPUs, j.Cores, j.GiB}, Weight: 1, Eligible: m6Names(4)})
	}
	return out
}

func m7Forecast(sc m6Scene, cfg m7Cfg) []m4bForecast {
	switch cfg.Forecast {
	case "accurate":
		return m7AccurateForecast(sc)
	case "clairvoyant":
		return m7ClairvoyantForecast(sc)
	}
	return sc.Forecast
}

// m7RunArm mirrors m6RunArm (same native calls, conservation and attribution assertions) with explicit controls.
func m7RunArm(t *testing.T, sc m6Scene, cfg m7Cfg, dir string) (map[string]any, map[string]any) {
	t.Helper()
	times := m6Times{}
	start := time.Now()
	e := m7Session(t, "m7-"+sc.ID+"-"+cfg.Name, sc)
	times.Fixture = time.Since(start).Seconds()
	forecast := m7Forecast(sc, cfg)
	steps, phase := []any{}, 0
	stageLog := []any{}
	for i, job := range sc.Arrivals {
		if i > 0 {
			e.arrive(t, i, job)
		}
		stepDir := filepath.Join(dir, fmt.Sprintf("arrival-%d", i))
		_ = os.MkdirAll(stepDir, 0700)
		beforeRes := m6Resources(e.f.session, e.names)
		beforeDomain := probeDomain(t, e.f)
		elig, cands, err := p2hEligibility(e.m6Env)
		if err != nil {
			t.Fatal(err)
		}
		if len(cands) == 0 {
			t.Fatalf("scene %s arm %s: arrival %d has no native candidate", sc.ID, cfg.Name, i)
		}
		eligOut := map[string]any{}
		for _, name := range e.names {
			eligOut[name] = elig[name]
		}
		s0, order0, secs := m6Scores(e.m6Env, t, cands)
		times.NativeScore += secs
		step := map[string]any{"index": i, "task": string(e.f.task.UID), "request": job, "before_resources": beforeRes, "before_domain_sha256": m4bHash(beforeDomain),
			"eligibility": eligOut, "candidates": cands, "native_scores_S0": s0, "native_order_S0": order0}
		active := false
		if !cfg.Off {
			phase++
			projStart := time.Now()
			praw, err := p2hProjection(t, e.m6Env)
			times.Projection += time.Since(projStart).Seconds()
			if err != nil {
				t.Fatal("P2H projection refused a declared-supported state: ", err)
			}
			var cohort *m7Cohort
			switch cfg.Cohort {
			case "forecast":
				c := m7ForecastCohort(sc, i, phase)
				cohort = &c
			case "oracle":
				c := m7OracleCohort(sc, i, phase)
				cohort = &c
			}
			produced := m7Exec(t, cfg.Kind, cfg.Transport, praw, phase, cfg.Alpha, forecast, cohort, filepath.Join(stepDir, "handoff"))
			times.Producer += produced.seconds
			times.LPSolve += m7LPSeconds(filepath.Join(stepDir, "handoff"))
			hooks := []uintptr{}
			for _, fn := range e.f.session.PredicateFns {
				hooks = append(hooks, reflect.ValueOf(fn).Pointer())
			}
			g := p2hGrant{owner: e.f.session, task: e.f.task, nodes: map[string]*node_info.NodeInfo{}, names: append([]string{}, e.names...), candidates: cands, phase: phase,
				envelopeHash: produced.envelope, projectionHash: bridgeHash(praw), producerHash: produced.producerHash}
			for name, n := range e.f.session.ClusterInfo.Nodes {
				g.nodes[name] = n
			}
			g.project = func(s *framework.Session) ([]byte, error) {
				if s != e.f.session {
					return nil, fmt.Errorf("foreign Session")
				}
				now := []uintptr{}
				for _, fn := range s.PredicateFns {
					now = append(now, reflect.ValueOf(fn).Pointer())
				}
				if !reflect.DeepEqual(now, hooks) {
					return nil, fmt.Errorf("predicate implementation changed")
				}
				return p2hProjection(t, e.m6Env)
			}
			begin := time.Now()
			if cfg.Kind == "p2hpc" {
				pg := &pcGrant{p2hGrant: g, cohortHash: cohort.Hash, cohortSource: cohort.Source, cohortIDs: cohort.IDs}
				active = e.pc.begin(produced.raw, pg, cfg.Lambda)
			} else {
				active = e.plugin.begin(produced.raw, &g, cfg.Lambda)
			}
			times.Begin += time.Since(begin).Seconds()
			if !active {
				t.Fatalf("fresh certified artifact did not activate (%s arrival %d)", cfg.Name, i)
			}
			step["lambda"], step["alpha"], step["BF"], step["BP"] = cfg.Lambda, cfg.Alpha, m4bBF, m4bBP
			step["projection_sha256"], step["envelope_sha256"], step["artifact_sha256"] = bridgeHash(praw), produced.envelope, m4bHash(produced.artifact)
			step["controls"] = json.RawMessage(produced.controls)
			step["artifact_pairs"] = m4bPairs(t, produced.artifact)
			stageLog = append(stageLog, produced.stages) // wall-clock: recorded only in the separate timing document
		}
		scores, order, secs := m6Scores(e.m6Env, t, cands)
		times.NativeScore += secs
		step["native_scores_S"], step["native_order_S"] = scores, order
		if active {
			var audit nodeProbeAudit
			if cfg.Kind == "p2hpc" {
				audit = e.pc.record.audit()
			} else {
				audit = e.plugin.record.audit()
			}
			step["advice_entries_returned"] = audit.Entries
			returned := map[string]float64{}
			for _, entry := range audit.Entries {
				if entry.Task == string(e.f.task.UID) {
					returned[entry.Node] = entry.Score
				}
			}
			if len(returned) != len(cands) || !audit.Complete {
				t.Fatal("native scoring did not use the bridge for every candidate ", audit)
			}
			worst := 0.0
			for _, name := range cands {
				// Exact for dyadic advice; non-dyadic certified blends (e.g. cohort prices) differ only in the last float bits.
				if d := math.Abs((scores[name] - s0[name]) - returned[name]); d > 1e-9 {
					t.Fatal("score not attributable to S0 + lambda*A ", name, scores[name], s0[name], returned[name])
				} else if d > worst {
					worst = d
				}
			}
			step["attribution_max_abs_error"] = worst
		} else if !reflect.DeepEqual(scores, s0) {
			t.Fatal("NATIVE_OFF must be the unadvised native score")
		}
		binds := len(*e.binds)
		allocStart := time.Now()
		allocate.New().Execute(e.f.session)
		times.Allocate += time.Since(allocStart).Seconds()
		if active {
			if cfg.Kind == "p2hpc" {
				e.pc.endPhase()
			} else {
				e.plugin.endPhase()
			}
		}
		p := e.f.session.ClusterInfo.PodGroupInfos[e.f.job.UID].GetAllPodsMap()[e.f.task.UID]
		if p == nil || p.Status != pod_status.Binding || len(*e.binds)-binds != 1 {
			t.Fatal("native Allocate did not bind exactly one task at arrival ", i)
		}
		afterRes := m6Resources(e.f.session, e.names)
		if err := m6Conserved(beforeRes, afterRes, e.names, p.NodeName, int64(job.GPUs), int64(job.Cores)*1000, int64(job.GiB)*nativeGiB); err != nil {
			t.Fatal("resource conservation: ", err)
		}
		step["chosen_node"], step["status_after"], step["bind"] = p.NodeName, p.Status.String(), (*e.binds)[len(*e.binds)-1]
		step["after_resources"], step["after_domain_sha256"] = afterRes, m4bHash(probeDomain(t, e.f))
		steps = append(steps, step)
	}
	futureStart := time.Now()
	future, admitted, admittedGPUs, admittedCores := []any{}, 0, 0, 0
	for k, job := range sc.Future {
		e.arrive(t, len(sc.Arrivals)+k, job)
		_, cands, err := p2hEligibility(e.m6Env)
		if err != nil {
			t.Fatal(err)
		}
		before := m6Resources(e.f.session, e.names)
		binds := len(*e.binds)
		allocate.New().Execute(e.f.session)
		p := e.f.session.ClusterInfo.PodGroupInfos[e.f.job.UID].GetAllPodsMap()[e.f.task.UID]
		row := map[string]any{"index": k, "task": string(e.f.task.UID), "request": job, "candidates": cands, "before_resources": before}
		if len(*e.binds)-binds == 1 && p.Status == pod_status.Binding {
			admitted++
			admittedGPUs += job.GPUs
			admittedCores += job.Cores
			row["outcome"], row["node"] = "ACCEPTED", p.NodeName
			if err := m6Conserved(before, m6Resources(e.f.session, e.names), e.names, p.NodeName, int64(job.GPUs), int64(job.Cores)*1000, int64(job.GiB)*nativeGiB); err != nil {
				t.Fatal("future resource conservation: ", err)
			}
		} else if len(*e.binds) != binds || p.Status != pod_status.Pending {
			t.Fatal("future task in unexpected native state")
		} else if len(cands) == 0 {
			row["outcome"], row["reason"] = "REJECTED", "NO_FEASIBLE_NODE"
		} else {
			row["outcome"], row["reason"] = "REJECTED", "FEASIBLE_BUT_NOT_BOUND"
		}
		if row["outcome"] == "REJECTED" {
			if err := e.f.job.UpdateTaskStatus(p, pod_status.Failed); err != nil {
				t.Fatal("withdrawing a rejected held-out job: ", err)
			}
			row["withdrawn_after_rejection"] = true
		}
		row["after_resources"] = m6Resources(e.f.session, e.names)
		future = append(future, row)
	}
	times.Future = time.Since(futureStart).Seconds()
	record := map[string]any{"arm": cfg.Name, "config": map[string]any{"off": cfg.Off, "alpha": cfg.Alpha, "lambda": cfg.Lambda, "kind": cfg.Kind, "forecast": cfg.Forecast,
		"cohort": cfg.Cohort, "label": cfg.Label}, "forecast_used": forecast, "scene": sc, "steps": steps, "future": future, "future_admitted": admitted,
		"future_admitted_gpus": admittedGPUs, "future_admitted_cores": admittedCores, "future_total": len(sc.Future), "final_resources": m6Resources(e.f.session, e.names),
		"bind_count": len(*e.binds), "native_future_admission": "NATIVE_ALLOCATE_NO_ADVICE_IDENTICAL_FOR_ALL_ARMS"}
	return record, map[string]any{"arm": cfg.Name, "transport": cfg.Transport, "scene": sc.ID, "regime": sc.Regime, "arrivals": len(sc.Arrivals), "seconds": times, "producer_stages": stageLog}
}

// m7LPSeconds sums the solver-attempt times of the produced M3-A execution report (any transport layout).
func m7LPSeconds(dir string) float64 {
	for _, p := range []string{filepath.Join(dir, "execution.json"), filepath.Join(dir, "producer", "execution.json")} {
		raw, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		var ex struct {
			Attempts []struct {
				Elapsed float64 `json:"elapsed_seconds"`
			} `json:"attempts"`
		}
		if json.Unmarshal(raw, &ex) != nil {
			continue
		}
		total := 0.0
		for _, a := range ex.Attempts {
			total += a.Elapsed
		}
		return total
	}
	return 0
}
