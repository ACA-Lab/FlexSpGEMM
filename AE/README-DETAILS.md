# FlexSpGEMM PPoPP 2027 artifact

> **Reviewer entry point (22 October 2026):** [AE_GUIDE.md](AE_GUIDE.md) / [PDF](AE_GUIDE.pdf) provide the PPoPP-style getting-started and step-by-step guide. The agreed core scope is A100 Figures 7, 8, 9, 11, 13 and 15; run `scripts/run_key_results.sh --fresh`. Figures 10/12 below remain qualified additional experiments. [Validation evidence](VALIDATION.md) records newly executed checks and known failures.


This directory supplies experiment and plotting drivers for paper Figures
**7, 8, 9, 10, 11, 12, 13 and 15**. The main FP64 comparison uses 142 SuiteSparse
matrices and computes `C=A²`.
It contains a fresh public checkout, an independent experiment driver, and the
MOSparse source explicitly supplied by the author. Figures 7-13 use fresh GPU
measurements. Figure 15 evaluates saved models against the author's frozen
exhaustive-grid evaluation dataset. Raw logs, command lines, return codes,
validation outcomes and SHA-256 hashes are retained.

## Source and scope

- FlexSpGEMM: https://github.com/for-the-juan/FlexSpGEMM , pinned at
  `1cf3e46b1266a4d6ad9a45d18bf62044d9811e66`.
- TileSpGEMM and spECK: the baseline source trees shipped at that same commit.
- MOSparse: `baselines/MOSparse-local`, copied from the author's specified local
  baseline directory. Build outputs were excluded; see `configs/provenance.json`.
- `configs/matrices.csv`: 142 unique matrices, deduplicated from the public
  `data/mat-142.csv` (which contains 143 records).
- This public commit contains optimizations made after the paper measurements;
  timing differences are reported, rather than adjusting code or configurations
  to match the paper numbers.

The AB generator and multi-scale ML checkpoints were recovered from the
author's older `FlexSpGEMM` experiment directory; copies and provenance are
included, without modifying that directory. `sources/probes` contains the
complete CPU feature-extraction source recovered from the author's earlier
data-preparation tree. The artifact checks its freshly compiled output against the released executables
for all 142 matrices; see `configs/rebuilt-feature-validation.csv`. tSparse is pinned at
`15bf5c415ae187a0ea99d060697606551923f70b` from https://github.com/oresths/tSparse.
H200 scripts are supplied; actual H200 execution requires that GPU.

## Requirements

Linux x86-64, Python 3.12, GNU C++/OpenMP, Git, NVIDIA driver, and CUDA Toolkit.
Use CUDA 11.8 / sm_80 on A100; CUDA 12.4 / sm_90 on H200, matching the paper's
toolkits. The test machine has A100-SXM4-80GB and driver 580.178.04; this driver
differs from the original paper. Input files occupy about 49 GiB uncompressed.
Allow additional GPU memory for each algorithm's workspace. Baseline allocation
failures are recorded and excluded individually.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-ml.txt
python scripts/bootstrap.py
```

All feature extractors, GPU multiplication and baselines can now be rebuilt
from the included sources. `bootstrap.py` obtains the public sources and the
README-linked Qwen checkpoint at pinned revisions. Public Git source bundles
are included; bootstrap clones them locally before considering a network clone. Multi-scale adapters,
classical models and frozen training/evaluation data are included separately.

For Figure 12, the current full-feature extractor is CPU based. The GPU timer
released upstream exposes only a subset of model features. Until the author's
complete GPU feature implementation is identified, its CPU-based preprocessing
measurement is explicitly labelled; it must not be presented as verification
of the paper's GPU feature-extraction overhead.

## Run

Run from this directory. Both launchers default to **physical GPU 1**. The
driver exposes only that GPU through `CUDA_VISIBLE_DEVICES`; programs then use
logical device 0. A device-name check prevents accidental H200 runs on A100.

```bash
# Full A100 workflow for all eight requested figures:
MATRIX_DIR=/path/to/mat-142 GPU_ID=1 bash scripts/run_paper_a100.sh --fresh

# Reuse the author's original input matrices, checking dimensions and SHA-256.
MATRIX_DIR=/path/to/mat-142 GPU_ID=1 bash scripts/run_a100.sh

# Or omit MATRIX_DIR to download the exact group/name pairs from SuiteSparse.
GPU_ID=1 bash scripts/run_a100.sh

# On an H200 host with CUDA 12.4 installed:
CUDA_HOME=/usr/local/cuda-12.4 GPU_ID=1 bash scripts/run_paper_h200.sh
```

The first invocation builds nine FlexSpGEMM tile shapes and all baselines.
In the 142-matrix main campaign, each matrix uses a fresh LightGBM prediction
over features measured from that input. The 81-class label mapping and five device features follow the published
predictor. For AA, both input-feature groups represent A, as in upstream step2.
Neither historical timings nor per-matrix search results select configurations.

```bash
# Small check after setup/build:
python3 scripts/ae.py run --matrices m_t1,pdb1HYS --repeats 1

# Full campaign (three independent process repetitions), resumable:
python3 scripts/ae.py run --gpu 1 --repeats 3
python3 scripts/ae.py plot

# Rebuild/rerun selected jobs explicitly:
python3 scripts/ae.py run --matrices m_t1 --repeats 3 --force
```

Completed attempts, including failures and timeouts, are preserved on resume.
Use `--force` to retry them. The default per-process timeout includes input
loading, preprocessing, multiplication and reference comparison. Change it with
`--timeout SECONDS` on `ae.py run`. Do not run multiple GPU campaigns concurrently.

## Timing and correctness

The reported GPU SpGEMM time is the implementation's symbolic+numeric interval,
including allocations inside it. Input loading, feature extraction, prediction,
format conversion and reference comparison are outside this interval.
FlexSpGEMM internally repeats five times; spECK uses its published one warmup and
ten repetitions; MOSparse performs one warmup and one timed execution;
TileSpGEMM uses its original one execution. The driver takes the median across
three independent invocations of each program. These are not end-to-end times.

The isolated artifact applies `patches/validation-integrity.patch` to the pinned
public source. It materializes the untimed validation CSR only after the final
internal repetition (the original overwrote and leaked four previous copies),
checks validation/output allocation and cuSPARSE API failures, and frees the
cuSPARSE workspaces. The arithmetic kernels and their timed intervals are
unchanged. The upstream validator only compared structure despite printing
“numerically correct”; the patch adds a real value check: FP64 absolute tolerance
1e-8 plus relative tolerance 1e-10; FP16 absolute and relative tolerance 0.005.
Nonfinite values fail. Mismatch samples are printed in the raw logs. These are
explicit AE validation changes and can alter memory pressure and cuSPARSE cleanup
time; they are not claimed to be the original paper's unmodified executable.
All pre-repair source snapshots, binaries, job metadata and result CSVs remain in
`diagnostics/unfixed-validation-materialization`; their raw logs remain in `logs`.
For example, `TSOPF_FS_b300_c3` now reports a FlexSpGEMM timing instead of failing
while constructing repeated validation outputs. The same full-input
Compute Sanitizer memcheck reports 21,922 errors before the repair and zero
after it (`diagnostics/validation-memory-repair-evidence.json`). Before the separate Tensor Core repair below, the tested all-Tensor Core
configuration on `m_t1` had numerical mismatches, while CUDA-only passed;
all those timings and raw outputs are retained.

The primary kernel also applies the separately disclosed three-line patch
`patches/optional-tensor-bitmask-correctness.patch`. Tile masks encode columns
from the most significant bit, while the published sparse-to-dense helper
interpreted them from the least significant bit and used the opposite rank.
The patch makes the helper follow the actual mask layout and CSR value order.
This changes kernel code and is **not** presented as the unmodified paper
executable. Independent 8×8 and 16×32 all-Tensor Core checks on `m_t1` pass after
the correction. New primary measurements are rerun using the corrected kernels;
the available earlier published-kernel measurements, binaries and snapshots are
retained in `diagnostics/unfixed-tensor-bitmask`, with raw logs in `logs`.
`configs/execution-policy.json` records the choice. Setting
`primary_tensor_bitmask_repair` to false and rerunning bootstrap/build selects
the published arithmetic kernel; source signatures invalidate affected cached
measurements and preserve the previous metadata and binaries.

FlexSpGEMM is built with `CHECK_RESULT=1` and compares nnz, row pointers, column
indices and values against cuSPARSE. The cuSPARSE baseline timing is obtained
from that call. Failed reference computation is distinguished from a passed
validation. MOSparse's upstream comparison checks structure, not all values;
spECK's published config disables its comparison. These statuses are reported
explicitly. An exit code of zero alone never counts as successful measurement.
As requested by the author, available timings are retained even when validation
fails. Validation status and process status remain explicit. Only the cuSPARSE
pairing excludes matrices whose cuSPARSE comparison failed or was unavailable.
An apparent cuSPARSE throughput accompanied by zero output nnz is not a usable
cuSPARSE measurement. Other reported method timings remain available.

Mean speedups use the matrices where both methods return usable results.
Arithmetic and geometric means and the common matrix count are all reported.
Cases with no reported timing have blank timings, never synthetic zero times.
Figure plots distinguish missing data from actual reported measurements.
The memory plot is input-A representation size in MiB as printed by the code;
the legacy `TileSpGEMM Memory Cost` label inside FlexSpGEMM describes the sparse
part of its current tiling, not a separately measured fixed-16×16 baseline.
Figure 11 uses the actual baseline program's footprint and leaves a point
missing when that footprint was not reported; it does not substitute the
legacy-labelled FlexSpGEMM value.

## Outputs

- `results/a100/environment.json`: device, software location, invocation.
- `configs/input-audit.csv`: exact input identity and checksums.
- `results/a100/features/`: fresh predictor input and selected configuration.
- `results/a100/jobs/`: per-attempt process metadata and parsed metrics.
- `results/a100/repeats.csv`: every attempt, including failures.
- `results/a100/medians.csv`, `speedups.csv`, `REPORT.md`: aggregate results.
- `results/a100/figures/`: standalone PDF plots.
- `logs/a100/`: complete raw probe and benchmark logs.

H200 uses corresponding `h200` paths. It is not marked validated by an A100 run.

## Individual figures

| Figure | Run / collect | Plot input |
|---|---|---|
| 7, 8 | `python scripts/ae.py run` then `collect` | `results/a100/medians.csv` |
| 9 | `python scripts/supplement.py generate-ab` then `ab` | `results/a100/ab.csv` |
| 10 | `python scripts/supplement.py fp16-manifest`, `fp16-build`, `fp16` | `results/a100/fp16.csv` |
| 11 | Same main campaign | `results/a100/medians.csv` |
| 12 | `python scripts/supplement.py preprocessing` | `results/a100/preprocessing.csv` |
| 13 | `python scripts/supplement.py ablation` | `results/a100/ablation.csv` |
| 15 | `python scripts/ml_eval.py --gpu 1` then `python scripts/ml_representatives.py --gpu 1` | `ml_scaling.csv`, `ml_representatives.csv`, `ml_summary.csv` |

```bash
python scripts/plot_paper.py
```

The plotting script writes standalone PDF and 300-dpi PNG files under
`results/paper-figures`, with the paper's layout, methods, ordering, colours,
hatches, labels and axes. `manifest.json` identifies each input by SHA-256.
It does not fill missing measurements using the paper's original bar values.

Figure 9 follows the author's deterministic Graph500/R-MAT generator with
vertex permutation and induced-graph trimming; TS selects the same nested,
sorted 128 columns using the author's hash-derived seed. P is the first-level
PyAMG 5.3.0 smoothed-aggregation prolongator on the four SPD matrices. Generation
parameters and selected-column identities are recorded in sidecars.

For Figure 10, both algorithms use the same structure and positive exactly
FP16-representable input value `1/16`. This avoids input-range rejection in
tSparse and gives both implementations the same numerical workload. It is an
explicit input-value normalization, not a claim that the unreleased original
FP16 value policy has been recovered. FlexSpGEMM configurations currently come
from the released FP64-trained LightGBM predictor; a separate original FP16
checkpoint/configuration policy has not been identified. Original-value trial logs are retained in
`diagnostics/fp16-original-input-values` and the raw log directory.

Figure 13's Base is fixed 16×16 CUDA-only. AT+C disables WMMA while retaining
the adaptive layout. AT+T disables the tiny/small/large dispatch buckets and
routes every nonempty C tile to the Tensor Core dense/full kernels. Sparse
kernel template buffer sizes stay positive. The full variant uses the ordinary
predicted density threshold. All generated-source changes are in `patches`.

The pinned ML environment uses Transformers 5.3.0 / PEFT 0.19.1, matching the
recovered experiment. Its maximum-scale fresh accuracies are 86.72%, 80.51% and
76.93% for LightGBM, LLM and SVM, respectively, matching the paper. The earlier
Transformers 4.51.3 evaluation remains under `ml-alternatives/transformers-4.51.3`.

Figure 15(a) uses the author's four-domain refit checkpoints with
200/400/600/800/**999 base records** (995 distinct matrix identities in the full source pool). The paper rounds the last size to
1000. LightGBM and LLM use their recovered scale-study checkpoints. The paper's
SVM curve comes from the separate weak sigmoid SVM study (`C=0.09`, `gamma=scale`,
`coef0=0.5`). That study's metadata explicitly describes test-informed model
selection; this artifact preserves that disclosure. The ordinary RBF SVM
results are retained under `results/a100/ml-alternatives/rbf`.

The recovered paper aggregation uses only runnable configurations for the
LightGBM and LLM means, while SVM includes all 142 matrices, including failed
predicted configurations as zero achieved throughput. `ml_scaling.csv` includes
both the all-matrix and runnable-only means and their counts so this inconsistent
denominator policy remains visible. Figure 15(b) instead uses the **released**
LightGBM/SVM and README-linked LLM checkpoints, evaluated by
`python scripts/ml_representatives.py --gpu 1`. These are different checkpoints
from the scale-study refits. The default LLM path reproduces the original experiment's loader call with
`subfolder="checkpoint"`. With the pinned Transformers version, adapter
auto-detection searches the repository root and does **not** enable the adapter
stored in that subfolder. The base-only prediction matches the recovered paper
panel. This behavior is disclosed, not represented as a successful evaluation
of the enabled LoRA adapter. An additional
`--loader explicit-adapter` evaluation enables the adapter through PEFT and
writes separate results under `ml-alternatives/explicit-adapter`. Raw generated
text, full prompts and loader state are retained for both protocols.

All predictions and inference times are freshly measured. The throughput
oracle is the frozen author exhaustive-grid dataset, used only for model
accuracy, never to select a configuration for Figures 7-13. Training scripts,
exact splits and checkpoint metadata are included. Table training durations
come from saved refit metadata, not a fresh training run. The original released
checkpoint timing evidence records **2994.097 ms as LLM first-call inference**,
not training seconds; it also states that released-checkpoint training time was
not recorded. Consequently the artifact cannot substantiate the paper's
2994.1-second training-time entry and does not relabel that inference latency
as training time. See `configs/evidence/legacy_checkpoint_timing_gpu0_20260713.json`.

All raw logs are immutable: a rerun creates a timestamped file. Previous job
metadata is archived on `--force`; failed builds, timeouts and interrupted
experiments remain in `logs` / `diagnostics`. GPU workflows use a shared lease on local `/tmp` storage,
so multiple artifact drivers do not run GPU measurements simultaneously.

## Integrity checks and portable package

```bash
python scripts/test_artifact.py
python scripts/audit.py --platform a100
# Strict completion check: fails on missing attempts, damaged logs or the
# unresolved FP16 value policy (Figure 10) and GPU feature-extraction protocol (Figure 12).
python scripts/audit.py --platform a100 --final
python scripts/package_artifact.py
```

The timestamped tarball contains auxiliary source, models, frozen ML data,
configuration manifests, the generated AB right-hand-side matrices, plots,
results, and all raw experiment logs. It omits
rebuildable binaries, Python environment, downloaded matrices, and the expanded public checkouts. `bootstrap.py` expands the included Git
source bundles and fetches the pinned LLM base weights. The archive has a SHA-256 sidecar.
Unpack, create the environment, bootstrap, and run the relevant launcher.
Packaging during a running campaign is a snapshot; final coverage must be
assessed from `AUDIT.json`, not inferred from a tarball's existence.

The H200 full launcher runs the main and supplemental GPU workloads and writes
separate H200 plots. Its model features use the published H200 device constants.
Figure 15 follows the A100 model evaluation protocol. H200 compilation and
execution remain unverified until performed on actual H200 hardware.

An additional upstream GPU microbenchmark is available separately:

```bash
python scripts/gpu_probe_timing.py --gpu 1
python scripts/gpu_probe_timing.py --collect-only
python scripts/plot_paper.py --figures 12 --preprocess-input preprocessing-gpu-subset.csv --out results/gpu-probe-subset-figures
```

This uses the published benchmark's five repetitions and default 100,000-nnz
probe limit. It reports the A probe plus C build/feature interval; transfer
latency is retained separately. It produces only a subset of the model's
features and therefore does not resolve the full GPU preprocessing protocol.
Its output cannot pass the strict Figure 12 protocol audit.

Fresh optional training uses the shipped splits without changing recovered
checkpoints. Use the original training dependency versions listed in
`sources/ml-paper-svm/requirements-lock.txt` (Transformers 5.3.0 / PEFT 0.19.1),
then run, for example:

```bash
python scripts/retrain_ml.py --method svm-paper --sizes 200
bash scripts/build_lightgbm_cuda.sh
python scripts/retrain_ml.py --method lightgbm --gpu 1 --sizes 200 400 600 800 999
python scripts/retrain_ml.py --method llm --gpu 1 --sizes 200 400 600 800 999
```

Each run writes a new `training-runs` directory and immutable command logs;
training does not replace the recovered checkpoints. All five SVM sizes were
freshly trained and reproduced every recovered-checkpoint prediction (142/142
for each size); see `configs/evidence/svm-retraining-prediction-parity.json`. The published input-pool
CSV files fetched by bootstrap are used only for source-overlap annotations.
The frozen data audit records 42 of the 142 test identities in the source pool.
The original group checks prevent train/validation overlap; this experiment
must not be described as evaluation on 142 entirely unseen matrix identities.

The package includes the 115 MiB generated AB inputs, so AMG interpolation
values do not depend on a reviewer's BLAS or sparse-library rounding. Generator
code, seeds, sidecars and input hashes are also included; fresh regeneration
is checked against `configs/ab-reference-sha256.csv`.

LightGBM inference uses the ordinary 4.6.0 package. The recovered training
protocol uses its CUDA tree learner, which ordinary CPU wheels do not provide.
`build_lightgbm_cuda.sh` builds 4.6.0 with `USE_CUDA=ON` into the artifact only;
the training wrapper places it on PYTHONPATH and acquires the same GPU lease
as all other measurements.

Figure 9's recovered final paper bars use **grid-selected configurations**,
not the direct model predictions. The original procedure searches 81 choices,
then chooses from the fastest five the one closest to 90% of the best measured
throughput. `configs/ab-paper-configurations.csv` preserves those configuration
identities only; `scripts/ab_paper.py` remeasures their timings and correctness.
The original selection tables/scripts are included as provenance. Fresh direct
prediction results remain in `ab-predicted.csv` and their original logs.

```bash
python scripts/ab_paper.py --gpu 1
# Optional search from scratch; 12 workloads x 81 configurations:
python scripts/ab_grid.py build
python scripts/ab_grid.py run --gpu 1
python scripts/ab_grid.py select
python scripts/ab_paper.py --gpu 1 --config results/a100/ab-fresh-grid-configurations.csv
```

The grid disables reference checking for selection, as the original tuner did;
selected configurations are checked again by the production executable.
Historical configuration identities may enter this Figure 9 protocol; historical
runtime values never replace new measurements. Figures 7/8/11/12 still use fresh
LightGBM inference, and do not import per-matrix grid choices.

The original AMG generator records a seed but does not seed PyAMG's internal
random state. Exact regenerated P values are therefore not guaranteed. The
measured P matrices are shipped, checksummed and reused by default; this makes
benchmark input identity reproducible without pretending the original
unseeded numerical generation is byte deterministic.

A fresh CUDA LightGBM N=200 refit was verified using the archived selected
42-round count: it matched 141/142 recovered predictions. This is a build/refit
check, not a rerun of the 1000-round selection search. Use the frozen checkpoint
for exact Figure 15 prediction reproduction. Evidence is stored under
`configs/evidence/lightgbm-cuda-fresh-refit-n200.json`.
On this Conda host, the CUDA-built library requires the system C++ runtime;
the full training wrapper sets LD_PRELOAD for that process only. The short
refit verification command is:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python scripts/refit_lightgbm.py --gpu 1 --sizes 200
```

With the original dependency versions, all 36 fresh representative-model
predictions (12 matrices x 3 models) match the recovered original prediction
records. The parity evidence and original configuration identities are included
under `configs/evidence`; historical runtime columns are not used for this check.

Use `python scripts/package_artifact.py --include-llm-base` to also include
the pinned full LLM checkpoint, avoiding its download on the review machine.
Both archive modes include all logs and the public Git source bundles.

The supplied original checkout also contains an earlier **full-input** GPU
probe under `tools/gpu_probe_cuda118_src`. Its original median driver and the
CPU32 inference timer have now been recovered under `sources/gpu-probe-author`.
They are replayed independently of the newer 100K-nnz diagnostic:

```bash
python scripts/preprocessing_inference.py --gpu 1
python scripts/gpu_probe_author.py run --gpu 1
python scripts/gpu_probe_author.py collect
python scripts/plot_paper.py --figures 12 --preprocess-input preprocessing-author-gpu-proxy.csv
```

The protocol is five independent processes per matrix, median A-probe time plus
median C-build/feature time, `merge8`, `hash-avg`, no nnz truncation, and no
separate csr2tile microbenchmark. CPU inference uses two warmups and twenty
measurements with 32 threads, as in the recovered source. All repeats are logged.
The plotting script automatically prefers this recovered protocol when its CSV
exists. A footer identifies it as a **GPU probe proxy**: it still does not
produce the complete model feature vector. Replaying the original timing
measurement must be distinguished from verifying the full GPU preprocessing
claim. The complete CPU feature measurements remain in `preprocessing.csv`.

The distributed package includes producer results for inspection. For **new
measurements**, pass `--fresh` to the full launcher: it propagates `--force` to
all measurement/model stages and preserves previous logs and job metadata.
Without it the workflow resumes cached attempts. Individual experiment drivers
also accept `--force`. A different GPU/driver cannot silently reuse the main
campaign cache. Optional training remains separate from checkpoint inference.

After a campaign, `python scripts/finalize.py` refreshes all summaries, plots
and integrity audits without launching GPU measurements. Its stage logs and
exit codes are retained in `results/a100/FINALIZATION.json`. Strict plot
validation reports unresolved paper protocols explicitly.

The main campaign now uses the recovered MOSparse watchdog: 15 seconds after
its explicit `PPoPP27_LOAD_COMPLETE` marker, alongside the overall 300-second
process limit. This avoids repeated GPU stalls after parsing. Set
`--mos-post-load-timeout 0` to disable it. Earlier generous 300-second timeout
trials remain in the logs; available kernel timings from any interrupted
post-validation run are still retained. The original baseline driver is
archived under `sources/baseline_timeouts`.

The packager writes `configs/archived-log-inventory.csv` with the byte count and
SHA-256 of every preserved log immediately before archiving. This includes
failed builds, timeouts, interrupted attempts, diagnostics and old runs. Raw
logs are never replaced by parsed summaries. The final console messages about
the archive itself remain in the workspace's packaging/supervisor log after the
archive closes.

Figure 11 representation sizes and format-conversion observations remain in
the summary even if a later kernel/reference computation fails. They are input
metadata, and do not establish a successful SpGEMM run. `memory-summary.csv`
reports arithmetic/geometric memory ratios and the aggregate byte ratio against
the actual fixed-16×16 baseline, with explicit common-matrix counts.

H200 build validation was completed on this host without executing H200 GPU
code: all 52 experiment binaries compiled with CUDA 12.4.1 / NVCC 12.4.131,
and CPU-side `cuobjdump --list-elf` inspection confirmed `sm_90` for every binary.
See `configs/h200-build-verification.json` and the retained compile/inspection
logs. The H200 launcher repeats this check before benchmarking. The tSparse
H200 build uses an isolated copy with member `tuple.get<N>()` changed to
`thrust::get<N>(tuple)` for the CUDA 12.4 Thrust API; the A100 tSparse source
and measured binary are unaffected. The exact compatibility diff is in
`patches/tSparse-CUDA124-thrust-tuple.patch`. H200 execution and performance
remain unverified until the scripts run on actual H200 hardware.

The optional `scripts/fetch_cuda124.py` assembles the privately used toolkit
from NVIDIA's redistribution manifest at
https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.4.1.json
(with the official `.cn` mirror as a download fallback). Every component archive
is checked against the frozen NVIDIA SHA-256 and size in `configs`; toolkit
binaries are excluded from the AE archive and can be fetched again.

After packaging, `python scripts/verify_package.py --latest` checks the archive's
own SHA-256 and streams every archived file to verify the log inventory,
source/model/data fingerprints and (when included) all 17 base-checkpoint files.
Its JSON verification report stays next to the archive. This checks the actual
archived bytes, not only the files currently present in the workspace.

Independent sampled-row validation is available with
`python scripts/validate_sampled_rows.py --gpu 1`. A separate diagnostic build
computes selected rows on the CPU directly from the input CSR arrays and checks
both GPU structure and values. The current seven-workload run checked 112 rows
with zero mismatches, including cases where cuSPARSE cannot allocate its
reference workspace. This is partial validation, never a full-matrix pass, and
it does not change primary runtimes, reference-failure flags or comparison
eligibility. Every sampled row and skipped row is recorded in its raw log.

`python scripts/compare_committed_revision.py --gpu 1` separately benchmarks the
supplied directory's committed revision e32a90c. This is a diagnostic comparison,
with the same bitmask correction, CPU format conversion and CHECK_RESULT=0 to
avoid its legacy FP32 reference helper. It uses the older revision's minimum of
five internal timings; the primary newer source uses the median. Results stay
under diagnostics and never replace primary measurements. Confirm the intended
paper commit before treating either revision as the exact paper executable.

When a new timing exceeds the paper's fixed axes, the plotter also emits
`fig07-full-range` and `fig08-full-range` in PDF and PNG. The paper-axis versions
explicitly state which values are clipped; the full-range versions retain all
measurements. No value is rescaled to fit the original paper's axes.

Primary runtime collection additionally checks the program's own
`work/<platform>/result/results_tile.csv`, which records six decimal places.
Printed runtimes have only two decimal places. A higher-precision value is used
only when every CSV record matching the reported work, output nnz, runtime and
throughput agrees on exactly the same metric pair. Ambiguous cases retain the
printed values. Exact matched CSV text is copied beside the raw log, with its
SHA-256 and the original printed metrics preserved in `repeats.csv`. No timing
is guessed from an ambiguous record or filled from historical paper results.

For Figure 11, the native `all_dense_bytes` print expression overflows 32-bit
integer multiplication on cage15 (reported as 17,592,186,030,608.19 MiB).
The collector keeps that number as `reported_Dense_MB` and computes the intended
logical tile-value storage cost from the verified input tile count using Python
integers: 6,621,208 × 32 × 32 × 8 bytes = 51,728.1875 MiB. The raw logs and native
CSV remain unchanged; no kernel timing is altered. `Dense_MB`, byte counts,
source labels and an overflow flag make this correction explicit. Figure 11
also has a `fig11-full-range` version to show values beyond the paper axes.

The cuSPARSE series follows the published in-process reference call after
FlexSpGEMM. At that point the FlexSpGEMM input tile buffers and GPU golden CSR
are still resident, so the reference has less free device workspace than a
standalone cuSPARSE process. This context is part of the measured protocol;
reference failures are retained explicitly and excluded from cuSPARSE speedup
pairings. These results are not labelled as an independent standalone cuSPARSE
memory-capacity benchmark.
