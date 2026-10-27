# AccelSched-KAI PoC-0

CPU-only **offline KAI-like scheduling advisor**, schema v0.1. This is not native
KAI integration. It neither submits workloads nor implements Kubernetes semantics.
All results are `PROBE_ONLY`, not algorithm performance measurements.

Four production modules: `schema.py` validates inputs/statuses; `adapter.py` maps
stable input indices to frozen `m0_oracle.Instance`; `advisor.py` certifies original
Shape-LP prices and calls the pinned research `_replay`; `oracle.py` independently
audits physical placement and future admissions. No scheduler `validate` or
placement decision is reused by the independent Oracle. The one test file also
records evidence; no new framework or service is needed.

## Reuse and installation

Required upstream: [AccelSched at
9dc0c263d2aefeb151d565166b6b950e2ac9f6a5](https://github.com/JohnnyZPiggy/AccelSched/tree/9dc0c263d2aefeb151d565166b6b950e2ac9f6a5).
Read references: [m0_oracle.py](https://github.com/JohnnyZPiggy/AccelSched/blob/9dc0c263d2aefeb151d565166b6b950e2ac9f6a5/m0_oracle.py),
[certified_pgfp.py](https://github.com/JohnnyZPiggy/AccelSched/blob/9dc0c263d2aefeb151d565166b6b950e2ac9f6a5/certified_pgfp.py),
[bench_ultra_pgfp.py](https://github.com/JohnnyZPiggy/AccelSched/blob/9dc0c263d2aefeb151d565166b6b950e2ac9f6a5/bench_ultra_pgfp.py).
The pinned commit is development main, not necessarily the v1.0.0 release tag.
The dependency metadata says `accelsched==1.0.0`; **that version alone is
insufficient**. Runtime checks all five algorithm/LP module SHA-256 hashes against
the pinned Git blobs (only CRLF to LF normalization). A mismatch fails closed as
`INTERNAL_ERROR`, with no advice. No code is vendored or patched upstream.

Use an isolated environment with existing NumPy, SciPy, pytest and setuptools>=77.
Install the frozen upstream from a separate local clone/export first, then this
repository, entirely offline once those files/dependencies are present:

```text
git clone https://github.com/JohnnyZPiggy/AccelSched.git ../AccelSched-KAI-reference
git -C ../AccelSched-KAI-reference checkout --detach 9dc0c263d2aefeb151d565166b6b950e2ac9f6a5
python -m pip install --no-index --no-build-isolation --no-deps ../AccelSched-KAI-reference
python -m pip install --no-index --no-build-isolation --no-deps .
python -m accelsched_kai.advisor fixtures/01_gpu_family.json
python -m pytest -q -ra
python tests/test_poc.py
```

`python tests/test_poc.py` requires this package installed (or `PYTHONPATH=src`)
and overwrites the bounded acceptance JSON evidence, not source/fixtures.
`accelsched-kai <snapshot.json>` is the equivalent installed console entrypoint.
CLI stdout is strict JSON; exit 0 means `PHYSICAL_VALID`, exit 1 means failure.
File/JSON failures are `INVALID_INPUT`; no recommendations are emitted on failure.
Solver settings, all attempts, native statuses, primal/dual vectors and independent
recertification are retained. A native time limit is not a hard process watchdog.

## Strict schema

The five files under `fixtures/` define runnable examples. F01–F04 are snapshots;
F05 contains six individually rejected snapshots in `cases`. A snapshot has
**exactly** `schema_version`, `units`, `nodes`, `tasks`, `future`,
`realized_future`, `constraints`. Unknown fields at every object level, duplicate
JSON keys, NaN/Infinity, duplicate IDs, booleans masquerading as numbers and unknown
node references are rejected. Missing values never silently receive defaults.

- Version is `"0.1"`; units are exactly
  `["whole GPUs", "integer CPU cores", "integer RAM GiB"]`.
- Node: `id`, `gpu_family`, `shape`, `capacity` (three integers).
- Current/realized job: `id`, `gpu_family` (string or null), `eligible` (node IDs),
  `demand` (three integers), `value` (finite nonnegative scalar).
- Predicted future type: same job fields, with `weight` replacing `value`.
- Eligibility means the explicit node list **and** the family predicate; an empty
  list or an absent matching family lawfully rejects the task. Input order fixes
  node/task/type indices; Shape labels use first occurrence order. Different GPU
  families do not create different Shapes unless the input explicitly does so.
- Supported `constraints` is `{}`. Any declared `gang`, `required_topology`,
  `fractional_gpu`, `mig`, `dra`, `preemption` semantics fails
  `UNSUPPORTED_CONSTRAINT`; an unknown constraint fails `INVALID_INPUT`.
  Fractional GPU coordinates also fail `UNSUPPORTED_CONSTRAINT` directly.
  There is no Gang atomicity, Rack topology, fractional GPU, MIG/ResourceClaim,
  preemption or other unexpressed KAI semantics.
- PoC limits: 1–8 nodes, 1–16 current jobs, 1–8 predicted types, 0–8 realized jobs,
  nonzero demands, resource coordinates/value/weight <=1,000,000, and
  `(nodes+1)^realized_jobs <=100000`. Oversized input fails `INVALID_INPUT`.
  These are safety bounds for an exact tiny offline Oracle, not benchmark limits.

## Mathematical meaning

The **unchanged original Shape-LP** aggregates capacities by Shape and relaxes
eligibility. Its optimum is not a discrete physical/family feasibility proof.
HiGHS is requested explicitly (`fallback=none`, maximum ten-second native limit).
The original canonical model hash, primal feasibility, dual signs/stationarity,
gap, complementary slackness and raw objective fields are checked again before
price extraction. Full original model coefficients and raw certificates are output.

`true_pgfp` calls frozen `certified_pgfp._replay` directly: utility
`value - prices @ demand`, descending maximum utility then task index, descending
per-task Shape utility then Shape index, first Shape with eligible feasible nodes,
maximum `F(after)-F(before)` then smallest node index. **One fixed certified price
matrix** is used by the replay and LP-guided best-fit. The ledger verifier checks
every task/Shape order, visited Shape, candidate list, marginal, choice, residual
step and final placement against normalized original inputs. `_replay` is a pinned
private research reference, not a public upstream API. No indexed GPU benchmark
code is executed. `first_fit` uses input task/node order; `lp_best_fit` uses the
same LP task/Shape ordering, then minimum residual resource sum and node index
(the frozen reference's best-fit criterion).

`F` is a **weighted node/type suitability surrogate**, not a number of future
jobs: a type can contribute on multiple nodes. Actual future jobs are evaluated
only after current placement is frozen. Sequential future first-fit uses realized
input order and reports count/value/residual; exhaustive memoized offline future
admission reports a separate optimal value/placement/count on that **same fixed
current residual**. It may reject jobs, but may not move current jobs. This is an
offline counterfactual for future admissions, not hindsight optimization of the
current policy. Predictions do not read `realized_future`.

| Fixture | Acceptance claim |
|---|---|
| F01 | Same Shape, two families: PGFP N1 (marginals -1,0); tied best-fit N0 |
| F02 | Initial F=4; placement on N0 gives F=1, on N1 F=3; PGFP N1 |
| F03 | Same predicted replay as F01; actual B-only future value PGFP=0, best-fit=1 |
| F04 | Both (mu,lambda_GPU)=(1,0),(0,1) certify max x with task/GPU bounds x<=1; proves dual nonuniqueness only |
| F05 | Six unsupported scenarios fail explicitly, with no physical-valid advice |

F04 analytical vectors are independently certified on the exact canonical model,
but are labeled analytical test witnesses, not claimed native solver outputs.
Separate alternative-candidate replay tests use manual fixed prices explicitly
labeled **not certified**; arbitrary prices never enter the public advisor API.

Statuses: `INVALID_INPUT`, `UNSUPPORTED_CONSTRAINT`, `LP_UNCERTIFIED`,
`SOLVER_TIMEOUT`, `REPLAY_MISMATCH`, `PHYSICAL_INVALID`, `PHYSICAL_VALID`,
`INTERNAL_ERROR`. Lawful rejection, even rejection of all jobs, can be physically
valid. Every policy is independently audited before advice is exposed. Failure
reports retain diagnostic traces, but never an `advice` field.

## Evidence and limits

See [evidence manifest](evidence/manifest.json), [fixture numerical reports](evidence/fixture-runs.json),
[two dual witnesses](evidence/dual-witnesses.json), [failure cases](evidence/failures.json),
[Oracle fault detection](evidence/oracle-faults.json), [manual replay witnesses](evidence/manual-replay.json),
[test collection](evidence/collection.txt), [test results](evidence/pytest.txt),
and [installation/environment receipt](evidence/acceptance.json).
Manifest records exact dependency versions, fixture/output file SHA-256 and
status denominators, separating native fixture runs from injected failures.
Each successful output includes normalized input SHA-256 and an output SHA-256
over the whole report **excluding that hash field**. Timings are diagnostic and
make full output hashes vary between runs; decision traces are compared directly.
Raw fixture file hashes and canonical JSON hashes are deliberately distinguished.

No GPU, Colab, Kubernetes, Docker, paid service, upstream edit, CI workflow,
large benchmark or performance certification is included. Validation is on one
Windows/Python/NumPy/SciPy environment; the fresh task venv shares preinstalled
numerical dependencies. This is not a Python-version or native KAI compatibility
matrix. Installation requires the frozen source; wrong/missing versions fail closed.

Engineer delivers a Draft PR for SAE Chief Architect independent T2; no self-merge.
