# AccelSched-KAI

[English](README.md) | [简体中文](README.zh-CN.md)

**面向 KAI 原生调度实验的可信优化建议与可复现研究。**

AccelSched-KAI 将 [AccelSched](https://github.com/JohnnyZPiggy/AccelSched) 的数学优化研究与 [KAI Scheduler](https://github.com/kai-scheduler/KAI-scheduler) 的原生调度路径连接起来，使用版本化、失败即拒绝的测试桥接机制，验证优化建议是否能够影响真实的原生节点排序与分配。

![MIT](https://img.shields.io/badge/license-MIT-0f766e) ![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab) ![KAI](https://img.shields.io/badge/KAI-native%20Fake%20Session-334155) ![研究项目](https://img.shields.io/badge/status-research%20software-64748b)

**导航：** [项目特点](#项目特点) · [系统架构](#系统架构) · [安装与快速开始](#安装与快速开始) · [研究结果](#研究结果) · [目录结构](#目录结构) · [适用范围](#适用范围与限制)

> **研究项目，并非生产部署。** 项目使用固定版本的官方 KAI **Fake Session**，真实调用原生 Predicate、节点排序、Allocate 与 Fake Cache，但不部署 Kubernetes，也不使用物理 GPU。它不是 NVIDIA/KAI 官方插件。

## 项目特点

- **原生 KAI 验证：** 直接调用原生可行性检查、节点排序、Allocate 和 Fake Bind，而不把离线模拟结果冒充原生行为。
- **优化证书：** 复用冻结的 AccelSched Shape-LP / PGFP，独立核查原始与对偶解、资源状态和物理可行性。
- **可信 Advice：** 版本化、单次使用、绑定当前 Session 状态与任务身份；错误、过期或不支持的建议被拒绝。
- **GPU 资源语义：** 严格整数 GPU、CPU 核心与 RAM GiB 投影，以及受控的 2–4 节点异构 GPU 实验。
- **保留负面结果：** 比较 Native OFF、PGFP 和 LP 混合 Advice，分析预测偏移和未来任务接纳。
- **执行开销研究：** 比较 Python 冷启动与实验性常驻 Worker；没有接入生产调度器。

## 系统架构

```text
KAI 原生 Session + 已声明预测
                │
                ▼
        可信资源快照
                │
                ▼
       AccelSched 优化
     PGFP / 原始 Shape-LP
                │
                ▼
         认证 Advice
                │
                ▼
       版本化可信 Bridge
     （身份、状态、候选集合）
                │
                ▼
  KAI Predicate / NodeOrderFn
                │
                ▼
     Native Allocate / Bind
                │
                ▼
      独立审计与未来任务
       Native 接纳差分
```

优化器只影响受控范围内的**节点评分**，不能覆盖 KAI 的原生可行性与资源分配判断。P1、P2、P2H、P2H-PC 是相互隔离的研究测试契约，并非稳定生产接口。离线 Oracle 与 Native KAI 的结果严格区分。

## 安装与快速开始

要求 Python 3.10+、NumPy 和 SciPy。必须使用冻结的 AccelSched 源码提交；仅安装同名 `accelsched==1.0.0` 并不足以满足源码哈希校验。

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

KAI 原生 Fake Session 测试位于 `poc1b/`，固定上游 KAI 提交 `374cbb796ecff07c4a9140e59286f3e3a280e84b`，研发所用 Go 版本为 1.26.8：

```bash
cd poc1b
go test -p 2 ./...
```

冻结的 Go 依赖元数据在部分只读构建中存在 `yaml.v2` 显式依赖缺失问题；历史实验使用工作区外置诊断 modfile，**不修改**仓库中的 `go.mod` 或 `go.sum`。以上命令不等于保证所有全新环境都可直接通过。`python tests/test_poc.py` 还可能覆盖早期 `evidence/` 文件，建议在临时工作副本中运行。

## 研究结果

| 研究 | 已记录结果 | 证据 |
| --- | --- | --- |
| **M6 异构 GPU Native 差分** | 31 个受控场景、94 个未来任务：Native OFF **89**、PGFP **85**、PGFP+LP **86** 个接纳 | [PR #34](https://github.com/JohnnyZPiggy/AccelSched-KAI/pull/34) |
| **M7 PGFP 失败分析** | 较大 Advice 权重没有提高这些场景的接纳，保留受损和无收益实例 | [PR #36](https://github.com/JohnnyZPiggy/AccelSched-KAI/pull/36) |
| **M7 LP 价格信息** | 单任务 LP 的非零价格为 **0/140**；已声明规划任务组为 **50/140**，但未增加实际接纳 | [Issue #35](https://github.com/JohnnyZPiggy/AccelSched-KAI/issues/35) |
| **M7 Python Worker 延迟** | Engineer 报告冷启动 **0.52–0.56 秒/请求**，实验性 Warm Worker **24–26 毫秒/请求** | [PR #36](https://github.com/JohnnyZPiggy/AccelSched-KAI/pull/36) |

M6 唯一改善场景是单独披露的人工机制案例；30 个不同种子/配置的场景中未观察到 PGFP 改善。这些证据来自模拟 GPU 的 Fake Session，**并未证明算法能够普遍提高接纳率或 GPU 加速性能**。M4–M7 的部分原始实验记录仍保存在未提交的 Mac 工作区，仓库中的 `evidence/` 主要为早期离线验收记录。

## 目录结构

```text
src/accelsched_kai/   优化适配、证书、Advice、审计与 Worker 实验
poc1/                 早期 KAI 接口探索
poc1b/                官方固定版本 KAI Fake Session 原生测试
tests/                Python 回归与故障测试
fixtures/             小规模可复现输入
evidence/             早期离线验收证据
```

## 适用范围与限制

- **不是生产级调度器：** 没有真实 Kubernetes 部署、实体 GPU 执行或官方 KAI 插件发布。
- **严格资源边界：** 当前只支持受控整数 whole-GPU、CPU 和 RAM；不支持共享 GPU、MIG、DRA、NUMA/拓扑和抢占等未建模语义。
- **没有宣称算法优势：** 原生 KAI 是重要基线，M6/M7 的负面结果完整保留。
- **没有 GPU 求解加速结论：** 本仓库未验证 PyTorch、cuPDLP 或 GPU 常驻调度带来的性能提升。
- **依赖固定源码：** 上游提交、内部研究接口和原始 LP 证书属于复现条件；本地原始证据不一定全部随仓库发布。

## 相关项目

- [AccelSched](https://github.com/JohnnyZPiggy/AccelSched)：异构计算资源优化与放置引擎。
- [KAI Scheduler](https://github.com/kai-scheduler/KAI-scheduler)：独立维护的上游调度器。
- [SAE Framework](https://github.com/JohnnyZPiggy/SAE-Framework)：Scientist–Architect–Engineer 科研工作流。

## 许可证

[MIT License](LICENSE)。本项目为独立研究成果，不代表 NVIDIA/KAI 官方发布或背书。
