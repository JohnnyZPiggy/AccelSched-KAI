package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"

	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/actions/allocate"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/node_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/pod_info"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/api/pod_status"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/cache"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/framework"
	"github.com/kai-scheduler/KAI-scheduler/pkg/scheduler/test_utils"
	"go.uber.org/mock/gomock"
)

func fixedAdvice(f nativeFixture, lambda float64) adviceSnapshot {
	return adviceSnapshot{1, f.session.ID, lambda, []adviceValue{{string(f.task.UID), "node0", -1}, {string(f.task.UID), "node1", 1}}}
}
func adviceEvidence(t *testing.T) string {
	t.Helper()
	d := os.Getenv("POC1B_ADVICE_EVIDENCE_DIR")
	if d == "" {
		return t.TempDir()
	}
	if err := os.MkdirAll(d, 0700); err != nil {
		t.Fatal(err)
	}
	return d
}
func adviceCountDelta(a, b nodeProbeAudit) map[string]uint64 {
	x, out := probeCounts(a), probeCounts(b)
	for k := range out {
		out[k] -= x[k]
	}
	return out
}

// Uses unchanged B4/M1 projections; only the expected score/order is generalized.
func adviceScoring(t *testing.T, f nativeFixture, want map[string]float64, order []string, observations ...func(string)) probeScoringReceipt {
	t.Helper()
	s := f.session
	r := probeScoringReceipt{Before: probeDomain(t, f), Eligible: []string{}, PredicateErrors: map[string]*string{}, Scores: map[string]float64{}, NativeErrors: map[string]*string{}}
	if err := s.PrePredicateFn(f.task, f.job); err != nil {
		t.Fatal(err)
	}
	nodes := []*node_info.NodeInfo{}
	for _, name := range []string{"node1", "node0"} {
		n := s.ClusterInfo.Nodes[name]
		err := s.PredicateFn(f.task, f.job, n)
		r.PredicateErrors[name] = nil
		if err != nil || !s.FittingNode(f.task, n, false) {
			t.Fatal("native candidate not eligible", name, err)
		}
		nodes = append(nodes, n)
	}
	r.Eligible = []string{"node0", "node1"} // Detached set; native order below is never sorted.
	r.Ordered = probeNames(s.OrderedNodesByTask(nodes, f.task))
	if !reflect.DeepEqual(r.Ordered, order) {
		t.Fatal("native order", r.Ordered, "want", order)
	}
	for _, observe := range observations {
		observe("controlled")
	}
	for _, n := range nodes {
		score, err := s.NodeOrderFn(f.task, n)
		r.NativeErrors[n.Name] = nil
		if err != nil || !finiteAdvice(score) || score != want[n.Name] {
			t.Fatal("native score", n.Name, score, err, want)
		}
		r.Scores[n.Name] = score
	}
	for _, observe := range observations {
		observe("aggregate-queries")
	}
	repeated := []*node_info.NodeInfo{}
	for i := 0; i < 8; i++ {
		repeated = append(repeated, nodes...)
	}
	r.StressOrder = probeNames(s.OrderedNodesByTask(repeated, f.task))
	expected := []string{}
	for _, name := range order {
		for i := 0; i < 8; i++ {
			expected = append(expected, name)
		}
	}
	if !reflect.DeepEqual(r.StressOrder, expected) {
		t.Fatal("native pool order", r.StressOrder)
	}
	for _, observe := range observations {
		observe("pool")
	}
	r.After = probeDomain(t, f)
	if !bytes.Equal(r.Before, r.After) {
		t.Fatal("scoring mutated domain")
	}
	return r
}

func adviceAllocate(t *testing.T, f nativeFixture, mock *cache.MockCache, want string) probeActionReceipt {
	t.Helper()
	r := probeActionReceipt{Before: probeDomain(t, f), Config: nativeJSON(t, f.session.Config), Bind: []json.RawMessage{}, Pipeline: []json.RawMessage{}, Evict: []json.RawMessage{}, Result: "native Execute void; exact in-process fake Bind nil; no real Kubernetes transaction"}
	mock.EXPECT().Bind(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).DoAndReturn(func(p *pod_info.PodInfo, n string, a map[string]string, numa cache.NUMAPrediction) error {
		r.Bind = append(r.Bind, nativeJSON(t, map[string]any{"task": nativeTaskState(p), "node": n, "annotations": a, "numa": numa, "returned_error": nil}))
		return nil
	}).Times(1)
	mock.EXPECT().TaskPipelined(gomock.Any(), gomock.Any()).Times(0)
	mock.EXPECT().Evict(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).Times(0)
	allocate.New().Execute(f.session)
	f = probeRefresh(t, f)
	r.After, r.UID, r.Status, r.Node = probeDomain(t, f), string(f.task.UID), f.task.Status.String(), f.task.NodeName
	if f.task.Status != pod_status.Binding || r.Node != want || len(r.Bind) != 1 {
		t.Fatal("native Allocate", r.Status, r.Node, "want", want)
	}
	return r
}

func TestFixedAdviceNativeScoring(t *testing.T) {
	if runtime.Version() != "go1.26.8" {
		t.Fatal("exact Go 1.26.8 required")
	}
	test_utils.InitTestingInfrastructure()
	dir := adviceEvidence(t)
	old := map[string]uintptr{}
	for _, name := range []string{"predicates", "nodeplacement", shadowName, nodeProbeName} {
		if b, ok := framework.GetPluginBuilder(name); ok {
			old[name] = reflect.ValueOf(b).Pointer()
		}
	}
	if _, ok := framework.GetPluginBuilder(adviceProbeName); ok {
		t.Fatal("unique test-only registry occupied")
	}
	framework.RegisterPluginBuilder(adviceProbeName, newAdviceScoringProbe)
	builder, ok := framework.GetPluginBuilder(adviceProbeName)
	if !ok {
		t.Fatal("missing registry builder")
	}
	for name, ptr := range old {
		b, ok := framework.GetPluginBuilder(name)
		if !ok || reflect.ValueOf(b).Pointer() != ptr {
			t.Fatal("existing builder changed", name)
		}
	}
	args := framework.PluginArguments{"test-only": "fixed-advice"}
	if builder(args).Name() != adviceProbeName || !reflect.DeepEqual(args, framework.PluginArguments{"test-only": "fixed-advice"}) {
		t.Fatal("builder identity/args")
	}
	nativeSave(t, dir, "registry.json", map[string]any{"name": adviceProbeName, "old_builders_unchanged": true, "registration": "test only, process-scoped; no init/main/Helm"})
	cases := []struct {
		name   string
		lambda float64
		mode   string
	}{
		{"off", 0, "off"}, {"m1", 0, "m1"}, {"zero", 0, "advice"},
		{"below", 4, "advice"}, {"tie", 4.5, "advice"}, {"above", 5, "advice"}, {"invalid", 5, "invalid"},
	}
	type normalized struct {
		Scoring      probeScoringReceipt
		Action       probeActionReceipt
		Counts       map[string]uint64
		ActionCounts map[string]uint64
		Entries      []nodeProbeEntry
	}
	first := map[string]normalized{}
	var identities [][]any // Keep objects alive while proving fresh independent arms.
	var seals []*adviceRecord
	var audits []nodeProbeAudit
	for repeat := 0; repeat < 3; repeat++ {
		for _, tc := range cases {
			t.Run(fmt.Sprintf("repeat_%d_%s", repeat, tc.name), func(t *testing.T) {
				d := filepath.Join(dir, fmt.Sprintf("repeat-%d", repeat), tc.name)
				if err := os.MkdirAll(d, 0700); err != nil {
					t.Fatal(err)
				}
				f, mock, ctrl := actionFixture(t, "B")
				objects := []any{f.session, f.session.ClusterInfo, mock, ctrl, f.session.Cache.KubeClient(), f.task, f.job, f.task.Pod, f.task.VectorMap, &f.task.ResReqVector[0], &f.job.AllocatedVector[0]}
				for _, name := range []string{"node0", "node1"} {
					n := f.session.ClusterInfo.Nodes[name]
					objects = append(objects, n, &n.IdleVector[0])
				}
				objects = append(objects, f.session.ClusterInfo.Queues["queue0"])
				for _, previous := range identities {
					for i, object := range objects {
						if reflect.ValueOf(object).Pointer() == reflect.ValueOf(previous[i]).Pointer() {
							t.Fatal("shared mutable arm object", i)
						}
					}
				}
				identities = append(identities, objects)
				// Unique per-Session metadata, excluded only from repeated metadata comparisons.
				f.session.ID = fmt.Sprintf("m2-%d-%s", repeat, tc.name)
				baseline := probeScoring(t, f, nil, d, "baseline")
				if baseline.Scores["node0"] != 1019 || baseline.Scores["node1"] != 1010 {
					t.Fatal("frozen score scale changed")
				}
				threshold := (baseline.Scores["node0"] - baseline.Scores["node1"]) / 2
				if threshold != 4.5 {
					t.Fatal("crossing threshold", threshold)
				}
				hooks, state := probeHooks(t, f.session), probeDomain(t, f)
				config := nativeJSON(t, f.session.Config)
				var plugin framework.Plugin
				var record *adviceRecord
				var neutral *nodeProbeRecord
				if tc.mode == "m1" {
					p := newNodeScoringProbe(nil).(*nodeScoringProbe)
					plugin = p
					p.OnSessionOpen(f.session)
					neutral = p.record(f.session)
				} else if tc.mode != "off" {
					p := builder(nil).(*adviceScoringProbe)
					plugin = p
					snapshot := fixedAdvice(f, tc.lambda)
					if tc.mode == "invalid" {
						snapshot.Entries[1].Advice = math.NaN()
					}
					if !p.prepare(snapshot) {
						t.Fatal("prepare failed")
					}
					// Mutating caller-owned data cannot alter the prepared immutable clone.
					snapshot.Entries[0].Advice = 99
					snapshot.Lambda = 999
					if p.prepare(fixedAdvice(f, 0)) {
						t.Fatal("snapshot may only be prepared once")
					}
					p.OnSessionOpen(f.session)
					record = p.record(f.session)
					if record == nil || record.active != (tc.mode != "invalid") {
						t.Fatal("snapshot activation")
					}
					before := len(f.session.NodeOrderFns)
					p.OnSessionOpen(f.session)
					if len(f.session.NodeOrderFns) != before {
						t.Fatal("duplicate open registered twice")
					}
				}
				if plugin != nil {
					t.Cleanup(func() { plugin.OnSessionClose(f.session) })
				}
				expectedHooks := map[string]int{}
				for k, v := range hooks {
					expectedHooks[k] = v
				}
				if plugin != nil {
					expectedHooks["NodeOrderFns"]++
				}
				if !reflect.DeepEqual(expectedHooks, probeHooks(t, f.session)) || !bytes.Equal(state, probeDomain(t, f)) || !bytes.Equal(config, nativeJSON(t, f.session.Config)) {
					t.Fatal("open side effect")
				}
				nativeSave(t, d, "hooks.json", map[string]any{"before": hooks, "after": probeHooks(t, f.session)})
				lambda := tc.lambda
				if tc.mode != "advice" {
					lambda = 0
				}
				want := map[string]float64{"node0": 1019 - lambda, "node1": 1010 + lambda}
				order := []string{"node0", "node1"}
				if lambda > threshold {
					order = []string{"node1", "node0"}
				}
				observe := func(phase string) {
					if record == nil {
						return
					}
					a := record.audit()
					perKey := map[string]uint64{"controlled": 1, "aggregate-queries": 2, "pool": 10}[phase]
					if a.Total != 2*perKey || !reflect.DeepEqual(probeCounts(a), map[string]uint64{"native-job-0/node0": perKey, "native-job-0/node1": perKey}) {
						t.Fatal("measured phase counts", phase, a)
					}
					nativeSave(t, d, phase+"-probe.json", a)
				}
				scoring := adviceScoring(t, f, want, order, observe)
				var start, finish nodeProbeAudit
				if record != nil {
					start = record.audit()
				} else if neutral != nil {
					start = neutral.audit()
				}
				if plugin != nil && (start.Total != 20 || !reflect.DeepEqual(probeCounts(start), map[string]uint64{"native-job-0/node0": 10, "native-job-0/node1": 10})) {
					t.Fatal("controlled/aggregate/pool counts", start)
				}
				action := adviceAllocate(t, f, mock, order[0])
				if record != nil {
					finish = record.audit()
				} else if neutral != nil {
					finish = neutral.audit()
				}
				counts := map[string]uint64{}
				if plugin != nil {
					counts = adviceCountDelta(start, finish)
					if !reflect.DeepEqual(counts, map[string]uint64{"native-job-0/node0": 1, "native-job-0/node1": 1}) || finish.Total != 22 {
						t.Fatal("actual Allocate callback counts", counts)
					}
					for _, e := range finish.Entries {
						value := -lambda
						if e.Node == "node1" {
							value = lambda
						}
						if e.Score != value || e.Error != nil {
							t.Fatal("callback increment/error", e)
						}
					}
				}
				norm := normalized{scoring, action, probeCounts(finish), counts, finish.Entries}
				if original, ok := first[tc.name]; ok {
					if !reflect.DeepEqual(original, norm) {
						t.Fatal("three-repeat evidence differs")
					}
				} else {
					first[tc.name] = norm
				}
				if tc.name == "m1" || tc.name == "zero" || tc.name == "invalid" {
					if !reflect.DeepEqual(first["off"].Scoring, scoring) || !reflect.DeepEqual(first["off"].Action, action) {
						t.Fatal("neutral/invalid OFF parity")
					}
				}
				nativeSave(t, d, "normalized.json", norm)
				nativeSave(t, d, "raw-counts.json", map[string]any{"after_scoring": start, "after_allocate": finish, "actual_action_counts": counts})
				nativeSave(t, d, "threshold.json", map[string]any{"original": baseline.Scores, "advice": map[string]float64{"node0": -1, "node1": 1}, "threshold": threshold, "lambda_effective": lambda, "native_tie_rule": "descending score then ascending node name, pinned sortNodesByScore; observed in native output", "arithmetic": "all values exactly representable binary64; exact equality assertions"})
				if plugin != nil {
					plugin.OnSessionClose(f.session)
					sealed := finish
					if record != nil {
						sealed = record.audit()
						if plugin.(*adviceScoringProbe).record(f.session) != nil {
							t.Fatal("closed session retained")
						}
					} else {
						sealed = neutral.audit()
					}
					if !sealed.Closed {
						t.Fatal("not sealed")
					}
					f = probeRefresh(t, f)
					nodes := []*node_info.NodeInfo{f.session.ClusterInfo.Nodes["node1"], f.session.ClusterInfo.Nodes["node0"]}
					before := probeDomain(t, f)
					post := probeNames(f.session.OrderedNodesByTask(nodes, f.task))
					if len(post) != 2 || !bytes.Equal(before, probeDomain(t, f)) {
						t.Fatal("post-close native scoring mutation/drop")
					}
					for _, n := range nodes {
						base := float64(0)
						for _, fn := range f.session.NodeOrderFns[:len(f.session.NodeOrderFns)-1] {
							v, e := fn(f.task, n)
							if e != nil {
								t.Fatal(e)
							}
							base += v
						}
						v, e := f.session.NodeOrderFn(f.task, n)
						if e != nil || v != base {
							t.Fatal("closed callback not neutral", v, base, e)
						}
					}
					if record != nil {
						if !reflect.DeepEqual(record.audit(), sealed) {
							t.Fatal("post-close write")
						}
						seals = append(seals, record)
						audits = append(audits, sealed)
					} else if !reflect.DeepEqual(neutral.audit(), sealed) {
						t.Fatal("M1 post-close write")
					}
					for i, r := range seals {
						if !reflect.DeepEqual(r.audit(), audits[i]) {
							t.Fatal("cross-session contamination")
						}
					}
					nativeSave(t, d, "sealed.json", sealed)
					nativeSave(t, d, "post-close.json", map[string]any{"native_order": post, "native_scores_equal_original_callbacks": true, "sealed_unchanged": true})
				}
				t.Logf("lambda=%g effective=%g native=%v order=%v Bind=1/%s counts=%v peak=%d", tc.lambda, lambda, scoring.Scores, scoring.Ordered, action.Node, counts, finish.Peak)
			})
		}
	}
	nativeSave(t, dir, "scope.json", map[string]any{"claim": "BOUNDED_FIXED_ADVICE_NATIVE_SCORING_AND_ALLOCATE_DIFFERENTIAL (tested fixture only)", "failed": t.Failed(), "sessions": 21, "repetitions": 3, "neutral_parity": "OFF=M1=lambda0=invalid snapshot", "nonzero": "lambda4 no flip; lambda4.5 exact tie node0; lambda5 flips scoring and native Allocate to node1", "limits": "test-only CPU; no production registration, LP/PGFP, K8s, GPU, scheduler process, performance, universal safety or race-detector claim"})
}

func TestAdviceFailSafeAndSessionIsolation(t *testing.T) {
	test_utils.InitTestingInfrastructure()
	dir := filepath.Join(adviceEvidence(t), "diagnostics")
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name string
		edit func(*adviceSnapshot)
	}{
		{"version", func(s *adviceSnapshot) { s.Version = 2 }},
		{"foreign-session", func(s *adviceSnapshot) { s.Session = "other-session" }},
		{"negative-weight", func(s *adviceSnapshot) { s.Lambda = -1 }},
		{"nan-weight", func(s *adviceSnapshot) { s.Lambda = math.NaN() }},
		{"infinite-weight", func(s *adviceSnapshot) { s.Lambda = math.Inf(1) }},
		{"overflow-risk-weight", func(s *adviceSnapshot) { s.Lambda = math.MaxFloat64 }},
		{"nan-advice", func(s *adviceSnapshot) { s.Entries[1].Advice = math.NaN() }},
		{"infinite-advice", func(s *adviceSnapshot) { s.Entries[1].Advice = math.Inf(-1) }},
		{"range", func(s *adviceSnapshot) { s.Entries[1].Advice = 1.01 }},
		{"unknown-task", func(s *adviceSnapshot) { s.Entries[1].Task = "unknown-task" }},
		{"unknown-node", func(s *adviceSnapshot) { s.Entries[1].Node = "unknown-node" }},
		{"duplicate", func(s *adviceSnapshot) { s.Entries[1] = s.Entries[0] }},
		{"empty", func(s *adviceSnapshot) { s.Entries = nil }},
		{"oversize", func(s *adviceSnapshot) { s.Entries = make([]adviceValue, nodeProbeCapacity+1) }},
		{"malformed-id", func(s *adviceSnapshot) { s.Entries[1].Node = "" }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f, _, _ := actionFixture(t, "B")
			base := probeScoring(t, f, nil, dir, "off")
			p := newAdviceScoringProbe(nil).(*adviceScoringProbe)
			snapshot := fixedAdvice(f, 5)
			tc.edit(&snapshot)
			p.prepare(snapshot)
			p.OnSessionOpen(f.session)
			t.Cleanup(func() { p.OnSessionClose(f.session) })
			r := p.record(f.session)
			if r.active || len(r.values) != 0 {
				t.Fatal("malformed snapshot partially activated")
			}
			on := adviceScoring(t, f, base.Scores, base.Ordered)
			if !reflect.DeepEqual(base, on) {
				t.Fatal("diagnostic changed native eligibility/score/order/state")
			}
			p.OnSessionClose(f.session)
			nativeSave(t, dir, tc.name+".json", map[string]any{"neutral_native_parity": true, "receipt": on, "audit": r.audit()})
		})
	}
	t.Run("missing-capacity-and-reuse", func(t *testing.T) {
		f, _, _ := actionFixture(t, "B")
		base := probeScoring(t, f, nil, dir, "off")
		p := newAdviceScoringProbe(nil).(*adviceScoringProbe)
		snapshot := fixedAdvice(f, 0)
		snapshot.Entries = snapshot.Entries[:1]
		if !p.prepare(snapshot) {
			t.Fatal("prepare")
		}
		p.OnSessionOpen(f.session)
		t.Cleanup(func() { p.OnSessionClose(f.session) })
		r := p.record(f.session)
		on := adviceScoring(t, f, base.Scores, base.Ordered)
		if !reflect.DeepEqual(base, on) {
			t.Fatal("missing advice zero fallback changed native eligibility")
		}
		// Public native aggregator with only the actual probe, for inputs that the
		// original native plugins do not support. No synthetic failing plugin.
		shell := &framework.Session{}
		shell.AddNodeOrderFn(r.score)
		if v, e := shell.NodeOrderFn(nil, nil); v != 0 || e != nil {
			t.Fatal("nil fallback")
		}
		for i := 0; i < nodeProbeCapacity; i++ {
			n := &node_info.NodeInfo{Name: fmt.Sprintf("unknown-%d", i)}
			if v, e := shell.NodeOrderFn(f.task, n); v != 0 || e != nil {
				t.Fatal("unknown/capacity fallback")
			}
		}
		a := r.audit()
		if len(a.Entries) != nodeProbeCapacity || a.Diagnostics == 0 || a.Complete {
			t.Fatal("unbounded diagnostics", a)
		}
		p.OnSessionClose(f.session)
		seal := r.audit()
		if v, e := shell.NodeOrderFn(f.task, f.session.ClusterInfo.Nodes["node0"]); v != 0 || e != nil || !reflect.DeepEqual(r.audit(), seal) {
			t.Fatal("closed write")
		}
		// Even the same Session label cannot consume an already used snapshot.
		other, _, _ := actionFixture(t, "B")
		p.OnSessionOpen(other.session)
		t.Cleanup(func() { p.OnSessionClose(other.session) })
		if p.record(other.session).active {
			t.Fatal("snapshot reused across Sessions")
		}
		if !reflect.DeepEqual(probeScoring(t, other, nil, dir, "other-off"), adviceScoring(t, other, base.Scores, base.Ordered)) {
			t.Fatal("foreign-session fallback")
		}
		if !reflect.DeepEqual(r.audit(), seal) {
			t.Fatal("old session contaminated")
		}
		nativeSave(t, dir, "missing-capacity-and-reuse.json", map[string]any{"missing_native_parity": true, "seal": seal, "same_label_new_session_active": false, "old_seal_unchanged": true})
	})
	// The cap also prevents a new overflow when appended to any finite aggregate;
	// this checks arithmetic only, without inventing a failing native plugin.
	for _, base := range []float64{math.MaxFloat64, -math.MaxFloat64, 1019, 1010} {
		for _, delta := range []float64{-adviceLambdaLimit, adviceLambdaLimit} {
			if !finiteAdvice(base + delta) {
				t.Fatal("unsafe aggregate cap")
			}
		}
	}
}
