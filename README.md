# AccelSched-KAI

**Research-grade, reproducible experiments connecting AccelSched optimization advice to the NVIDIA KAI Scheduler's native scheduling path.**

[中文介绍](#中文介绍) · [Architecture](#architecture) · [Evidence](#what-the-results-show) · [Getting started](#getting-started) · [Limitations](#scope-and-limitations)

AccelSched-KAI studies a practical question: **can a separately certified optimization signal improve KAI's own node-selection decisions, without bypassing native feasibility checks?**

It combines an offline optimization oracle, a versioned and fail-closed advice bridge, native KAI Fake Session tests, and independent replay/auditing. It is an **experimental integration and evaluation repository**, not a production KAI plugin, Kubernetes operator, or GPU-accelerated solver.

> **Research status.** M3–M7 have been integrated into the repository. Experiments include real calls to pinned KAI native scheduling functions *in simulated Fake Sessions*—not a deployed cluster or physical GPUs. Negative and null results are first-class findings.

## At a glance

| Area | What is implemented |
| --- | --- |
| Offline optimization | Frozen AccelSched Shape-LP/PGFP reference, independent resource and future-admission oracle, certified LP witnesses |
| Trust boundary | Versioned, test-only advice producers and receivers; exact source, state, task/node and candidate verification; neutral fallback |
| Native KAI experiments | Original Predicate/FittingNode, NodeOrderFn/OrderedNodesByTask, Allocate and Fake Cache/Bind in official pinned Fake Sessions |
| GPU resource semantics | Whole-integer GPU, CPU cores and RAM GiB projections with unsupported KAI semantics rejected |
| Reproducible comparison | Native OFF vs dynamic PGFP vs Shape-LP–blended advice, forecasting shifts, held-out native future admission, negative cases |
| Performance research | Experimental cold-process vs persistent Python-worker transport; **not** connected to production |

## What the results show

The M6 controlled heterogeneous-GPU study used **31 scenes** (30 seeded and one disclosed, hand-designed mechanism case), **four simulated GPU nodes per scene**, and **94 held-out future jobs** across scenes:

| Policy | Native future jobs admitted |
| --- | ---: |
| KAI Native OFF | **89 / 94** |
| Dynamic PGFP advice | **85 / 94** |
| PGFP + Shape-LP blend | **86 / 94** |

These results **do not establish a scheduling improvement**. PGFP helped in one hand-designed mechanism scene but harmed some seeded scenes. M7 reproduced the M6 baseline and examined native-score margins, forecast changes and advice strength: stronger advice did not improve these particular workloads.

M7 also found that the original certified Shape-LP returned **zero prices for all 140 single-target cases**, while a separately declared multi-task planning cohort produced **nonzero certified prices in 50/140 decisions**. Those prices **did not improve native future admission** in the tested cases. The planning-cohort path is a separate test-only, source-declared experimental contract—not an authenticated production forecast feed.

For the transport experiment, Engineer-reported matched requests took approximately **0.52–0.56 s** with a separate Python process and **24–26 ms** using an experimental warmed worker (about 21–22× at the request level). This is a *producer transport* finding, not GPU acceleration or production KAI throughput. Detailed raw experiment evidence, repeated runs and exceptions were recorded during SAE review; the uncommitted Mac-local evidence directories are not part of this repository.

**Read results as scoped, falsifiable evidence, not benchmark marketing.** The project has not demonstrated a general throughput, admission-rate or GPU-compute advantage over native KAI.

## Architecture

```text
Typed snapshot / declared forecast (no realized future leakage)
                     |
             AccelSched Shape-LP / PGFP
              certified primal-dual witness
                     |
           versioned test-only advice producer
                     |
       trusted, single-use, current-state bridge
                     |
  pinned KAI Predicate -> NodeOrderFn -> Native Allocate
                     |
       Fake Cache/Bind + native resource receipts
                     |
    independent audit / held-out future Native Allocate
```

The original KAI scheduler owns feasibility, candidate membership and allocation. Advice modifies only the authorized scoring component for supported sessions; invalid, stale or unsupported advice is rejected. P1 (CPU-only), P2 (two-node whole-GPU), P2H (2–4-node whole-GPU) and P2H-PC (planning-cohort probe) are separate bounded **research/test contracts**, not interchangeable public production APIs.

The certified original Shape-LP is **eligibility-relaxed**; its dual prices are not a proof of discrete node feasibility. The independent oracle is separate from genuine native KAI tests. `F` represents weighted future task/node feasibility opportunities, **not** realized job admissions.

## Getting started

### Python offline reference

**Prerequisites:** Python ≥3.10, NumPy, SciPy, pytest, and the exact frozen AccelSched research sources at commit [`9dc0c263`](https://github.com/JohnnyZPiggy/AccelSched/tree/9dc0c263d2aefeb151d565166b6b950e2ac9f6a5). The Python package metadata specifies `accelsched==1.0.0`, **but the version number alone is insufficient**: source hashes are verified and a mismatched installation fails closed. Dependencies must be available before using the offline `--no-index` installation example.

```bash
git clone https://github.com/JohnnyZPiggy/AccelSched-KAI.git
git clone https://github.com/JohnnyZPiggy/AccelSched.git AccelSched-reference
git -C AccelSched-reference checkout --detach 9dc0c263d2aefeb151d565166b6b950e2ac9f6a5

python -m pip install ./AccelSched-reference
python -m pip install ./AccelSched-KAI
cd AccelSched-KAI
python -m accelsched_kai.advisor fixtures/01_gpu_family.json
python -m pytest -q
```

The offline CLI produces structured JSON; `python tests/test_poc.py` can **overwrite bounded evidence files**, so use a disposable checkout if regenerating them.

### Native Fake Session probes

Native Go experiments are under `poc1/` and `poc1b/`. They depend on a pinned official [KAI Scheduler](https://github.com/kai-scheduler/KAI-scheduler) module revision `374cbb796ecff07c4a9140e59286f3e3a280e84b` (see `poc1b/go.mod`) and were developed with Go **1.26.8**. They are **tests** of original KAI scheduling functions using Fake Sessions; they do not launch a real scheduler.

```bash
cd poc1b
go test -p 2 ./...
```

The frozen Go dependency graph may require a separate local diagnostic modfile because a read-only build reports a missing explicit `yaml.v2` requirement. Do **not** silently change `go.mod`/`go.sum` to conceal that limitation; consult the reproducibility notes in issue handoffs before claiming a green environment. Go and Python tests require their respective dependencies to be installed.

## Repository guide

| Path | Purpose |
| --- | --- |
| `src/accelsched_kai/` | Offline solver adapter, certified artifact, versioned producers, independent analysis and experimental worker |
| `poc1/` | Early KAI interface exploration |
| `poc1b/` | Pinned native KAI Fake Session integration tests and test-only scoring/bridge experiments |
| `tests/` | Python regressions and failure/validation tests |
| `fixtures/` | Small reproducible offline scenarios |
| `evidence/` | Early offline acceptance records (not a complete archive of later M4–M7 Mac-local runs) |

## Scope and limitations

- **Not production-ready:** no Kubernetes cluster deployment, actual GPU execution, supported production KAI plugin, release-grade performance or operational security review.
- **Supported research scope is narrow:** integer whole GPUs, CPU and RAM in bounded controlled Fake Sessions. Fractional/shared GPUs, MIG, DRA, NUMA/topology, releasing/preemption and other unmodeled KAI semantics fail closed.
- **No proven algorithmic advantage:** M6/M7 deliberately preserve harmed and neutral cases; offline findings must not be conflated with native KAI allocations.
- **No GPU solver benchmark:** neither PyTorch acceleration nor cuPDLP/cuPDLPx GPU speedups are claimed in this repository.
- **Source/identity constraints:** pinned upstream SHA and hash checks are essential; alternative Git tags or unverified package versions are not equivalent.
- **Reproducibility limits:** some raw M4–M7 run logs live only in an uncommitted Mac-synced workspace; race detector and full environment compatibility were not established.

## Related projects and license

- [AccelSched](https://github.com/JohnnyZPiggy/AccelSched) — independent optimization engine, separately developed and released.
- [NVIDIA KAI Scheduler](https://github.com/kai-scheduler/KAI-scheduler) — upstream scheduler; this repository is an independent research experiment, not an official NVIDIA/KAI distribution.
- [SAE Framework](https://github.com/JohnnyZPiggy/SAE-Framework) — research engineering workflow used for scoped independent review.

Licensed under [MIT](LICENSE). See the license file for its existing copyright notice.

---

## 中文介绍

**AccelSched-KAI** 是一个面向研究与可复现实验的开源项目，探索如何将 AccelSched 的优化建议通过可信、版本化的测试桥接机制接入 **KAI Scheduler 原生节点评分与分配流程**。

它不是生产级 Kubernetes 插件，也没有部署到真实 GPU 集群。项目中的原生调度实验使用官方固定版本 KAI 的 **Fake Session**，真实调用原生 Predicate、节点评分、Allocate 和 Fake Cache，而不是将离线模拟器的结果冒充原生调度结果。

**目前的成果：** 完成了原始 LP 证书、严格资源投影、可信 Advice、GPU 整数资源语义、异构负载差分、失败归因和 Python Worker 延迟实验。M6 的 31 个受控场景中，Native OFF 接纳 89/94 个未来任务，PGFP 接纳 85/94，PGFP+LP 接纳 86/94；这说明**尚未证明 PGFP 优于 KAI 原生策略**。M7 验证多任务规划可产生非零认证 LP 价格，但本轮没有带来额外接纳收益。

**项目定位：** 可复现的 KAI 原生调度优化实验平台，强调真实对照、数学证书、拒绝不支持的资源语义，以及保留负面结果。后续不以新增算法或 GPU 求解器作为仓库成熟化的前提。

代码入口见上方 [Getting started](#getting-started)；限制见 [Scope and limitations](#scope-and-limitations)。欢迎通过 GitHub Issues 交流复现问题、接口经验和可审计的实验结果。
