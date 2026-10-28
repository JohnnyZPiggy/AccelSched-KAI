# AccelSched-KAI

[English](README.md) | [简体中文](README.zh-CN.md)

**Certified optimization advice for native KAI scheduling experiments.**

AccelSched-KAI connects the [AccelSched](https://github.com/JohnnyZPiggy/AccelSched) optimization research engine with the [KAI Scheduler](https://github.com/kai-scheduler/KAI-scheduler) through a versioned, fail-closed test bridge. It combines original Shape-LP certificates, native KAI Fake Session allocation, whole-GPU resource checks, and reproducible policy comparisons.

![License](https://img.shields.io/badge/license-MIT-0f766e) ![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab) ![KAI](https://img.shields.io/badge/KAI-native%20Fake%20Session-334155) ![Status](https://img.shields.io/badge/status-research%20software-64748b)

**Explore:** [Highlights](#highlights) · [Architecture](#architecture) · [Quick start](#installation-and-quick-start) · [Research findings](#research-findings) · [Repository layout](#repository-layout) · [Limitations](#scope-and-limitations)

> **Research integration, not a deployment.** The native experiments call pinned KAI Predicate, node scoring, Allocate and Fake Cache **inside simulated Fake Sessions**. No live Kubernetes cluster or physical GPU is used. The repository is not an official NVIDIA/KAI plugin or a production scheduler.

## Highlights

- **Native scheduling experiments:** use original KAI candidate feasibility, node ordering, Allocate and Bind rather than labeling an offline heuristic as native.
- **Certified optimization:** reuse the frozen AccelSched Shape-LP/PGFP model, independent primal-dual verification and resource audits.
- **Trusted advice:** versioned, single-use, current-state scoring grants; stale, unsupported and inconsistent advice fails closed.
- **Whole-GPU semantics:** exact GPU/CPU/RAM projections and heterogeneous 2–4-node Fake Session studies, with unsupported semantics rejected.
- **Reproducible research:** Native OFF vs PGFP/LP comparisons, forecast shifts, negative results, and independent replay.
- **Transport experiments:** cold Python subprocess versus a warm experimental worker; no production integration.

## Architecture

```text
  Native KAI Session + declared forecast
                  │
                  ▼
      Validated resource snapshot
                  │
                  ▼
        AccelSched optimization
    PGFP marginal + original Shape-LP
                  │
           Certified artifact
                  │
                  ▼
      Versioned trusted bridge
     (identity, state, candidate set)
                  │
                  ▼
    KAI Predicate / NodeOrderFn
                  │
                  ▼
      Native Allocate + Fake Bind
                  │
                  ▼
    Independent audit + held-out
       Native future admission
```

The bridge influences **scoring only**; KAI retains native feasibility and allocation authority. P1, P2, P2H and P2H-PC are separate bounded **test contracts**, not production APIs. Predicted future demand is not a realized future trace. An offline Oracle is used only for independently labeled comparisons.

## Installation and quick start

Python 3.10+; NumPy and SciPy are required. The repository depends on an **exact source-pinned AccelSched revision**, not merely the `accelsched==1.0.0` package version. Runtime source hashes are checked.

```bash
git clone https://github.com/JohnnyZPiggy/AccelSched.git
git -C AccelSched checkout --detach 9dc0c263d2aefeb151d565166b6b950e2ac9f6a5
python -m pip install ./AccelSched

git clone https://github.com/JohnnyZPiggy/AccelSched-KAI.git
cd AccelSched-KAI
python -m pip install .
python -m accelsched_kai.advisor fixtures/01_gpu_family.json
python -m pytest -q
```

For the official KAI native Fake Session test suite, use the pinned module in `poc1b/go.mod` (KAI commit `374cbb796ecff07c4a9140e59286f3e3a280e84b`; tested development toolchain Go 1.26.8):

```bash
cd poc1b
go test -p 2 ./...
```

Go's frozen dependency graph has a known read-only `yaml.v2` module metadata issue in some environments. Historical testing used an **external diagnostic modfile** rather than silently editing `go.mod`/`go.sum`. A clean-checkout green build is not claimed here. Running `python tests/test_poc.py` may overwrite bounded historical evidence files.

## Research findings

| Study | Recorded result | Evidence |
| --- | --- | --- |
| **Native heterogeneous GPU placement (M6)** | 31 controlled scenes / 94 held-out future jobs: Native OFF **89**, PGFP **85**, PGFP+LP **86** admitted | [M6 PR #34](https://github.com/JohnnyZPiggy/AccelSched-KAI/pull/34) |
| **PGFP failure analysis (M7)** | Larger advice weights did not improve admission on the tested scenes; negative and neutral cases retained | [M7 PR #36](https://github.com/JohnnyZPiggy/AccelSched-KAI/pull/36) |
| **LP price information (M7)** | Single-target LP: **0/140** nonzero-price cases; declared planning cohort: **50/140**; **no observed admission gain** | [M7 Issue #35](https://github.com/JohnnyZPiggy/AccelSched-KAI/issues/35) |
| **Producer latency (M7)** | Engineer-reported cold **0.52–0.56 s/request** vs warmed experimental worker **24–26 ms/request** | [M7 PR #36](https://github.com/JohnnyZPiggy/AccelSched-KAI/pull/36) |

The one beneficial PGFP case in M6 was an explicitly disclosed hand-designed mechanism scene; the 30 seeded scenes provided no beneficial PGFP case. The LP cohort probe is test-only, and certified prices alone did not change the final Native allocation outcome. These are **bounded simulated results**, not evidence of generalized admission or GPU acceleration gains. Engineer-run raw M4–M7 evidence lives in an uncommitted local workspace; only historical `evidence/` is bundled here.

## Repository layout

```text
src/accelsched_kai/   Optimization adapter, certified artifacts,
                      trusted producers, audits, latency experiment
poc1/                 Early KAI interface probes
poc1b/                Pinned native KAI Fake Session experiments
tests/                Python regression and fault tests
fixtures/             Bounded example snapshots
evidence/             Early offline acceptance records
```

## Scope and limitations

- **Not production-ready:** no live Kubernetes deployment, real GPU execution, official KAI plugin or service-level performance guarantee.
- **Narrow trusted domain:** integer whole GPUs, CPU cores and RAM GiB in controlled Fake Sessions; fractional/shared GPUs, MIG, DRA, NUMA/topology and releasing/preemption are not supported.
- **No proven policy superiority:** M6/M7 preserve negative results; original KAI policies remain the comparison baseline.
- **No GPU solver speedup claim:** this repository does not benchmark PyTorch, cuPDLP or GPU-resident scheduling.
- **Pinned interfaces:** frozen AccelSched/KAI SHAs, private research APIs and original LP certificates are part of the experimental reproducibility boundary.
- **Partial raw archive:** later Mac-local raw logs and race-detector evidence are not part of the committed repository.

## Related projects

- [AccelSched](https://github.com/JohnnyZPiggy/AccelSched) — heterogeneous placement and optimization engine.
- [KAI Scheduler](https://github.com/kai-scheduler/KAI-scheduler) — upstream scheduler, independently maintained.
- [SAE Framework](https://github.com/JohnnyZPiggy/SAE-Framework) — Scientist–Architect–Engineer research workflow.

## License

[MIT License](LICENSE). This independent project is not affiliated with or endorsed by NVIDIA/KAI.
