# FlexSpGEMM artifact evaluation

This artifact contains the scripts and data needed to evaluate Figures **7, 8, 10, 11, 12, 13 and 15** of *FlexSpGEMM: Adaptive Tiling and Hybrid Execution for SpGEMM on Modern GPUs*. The short, Overleaf-ready artifact description is [overleaf/ae.tex](ae.tex). Detailed protocols and evidence are in [README-DETAILS.md](README-DETAILS.md) and [VALIDATION.md](VALIDATION.md).

## 1. Requirements

Use Linux x86-64, an NVIDIA A100 80 GB, CUDA Toolkit 11.8 (including `nvcc` and cuSPARSE), a compatible NVIDIA driver, GCC/G++ with OpenMP, make, CMake, Git, and Python 3.12. Allow at least 100 GiB disk space, plus substantial host RAM (the prior machine had 377 GiB). The 142 main matrices occupy 48.16 GiB uncompressed. Download and full evaluation can take a working day or longer. The paper used an A100 PCIe; previous artifact tests used an A100 SXM4, so performance may differ.

## 2. Get the artifact and source

The public GitHub repository contains the implementation; **the separate AE archive is also required** for the experiment drivers, patches, MOSparse source, models and frozen evaluation data. The archive must be supplied alongside these instructions; it is not currently available from a confirmed public URL. From a directory containing the delivered archive and its `.sha256` file:

```bash
export ARCHIVE=/path/to/FlexSpGEMM-AE-RELEASE.tar.gz
cd "$(dirname "$ARCHIVE")"
sha256sum -c "$(basename "$ARCHIVE").sha256"
tar -xzf "$ARCHIVE"
cd FlexSpGEMM-AE
git clone https://github.com/for-the-juan/FlexSpGEMM.git upstream
git -C upstream checkout --detach 1cf3e46b1266a4d6ad9a45d18bf62044d9811e66
```

The archive created by `scripts/package_artifact.py` does not contain the expanded `upstream` checkout. If it is already present, check its revision instead of cloning into it. Bootstrap applies the documented AE patches, checks the revision, and fetches other pinned dependencies and LLM weights as needed. A GitHub clone alone does not reproduce the seven figures.

## 3. Install and check

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-ml.txt
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GPU_ID=0                 # physical GPU index from nvidia-smi
nvidia-smi
nvcc --version
python scripts/bootstrap.py
python scripts/quickstart.py --gpu "$GPU_ID" --cuda "$CUDA_HOME"
```

The quick check downloads `m_t1` and `mult_dcop_03`, builds the required executable, and checks their outputs against cuSPARSE. Both checks must pass. To reuse local matrices, append `--matrix-dir /path/to/mat-142`; the directory must directly contain the `.mtx` files. The quick check is not a full evaluation.

## 4. Run the experiments

```bash
GPU_ID="$GPU_ID" CUDA_HOME="$CUDA_HOME" bash scripts/run_requested_figures.sh --fresh
```

This obtains the matrix inputs, builds the programs, measures the requested experiments, evaluates saved models, and writes plots to `results/requested-figures/`. To reuse the main and FP16 inputs, prepend `MATRIX_DIR=/path/to/matrices` to the command; the flat directory must contain all 142 main and 16 FP16 `.mtx` files. Missing files are downloaded. The SuiteSparse collection reported an outage in October 2026, so provide local inputs if its server remains offline. `--fresh` reruns measurements; without it completed attempts, including failures, can be reused. Run on an otherwise idle GPU. Raw logs are in `logs/`, detailed results in `results/a100/`, and plot input/coverage and protocol status in `results/requested-figures/manifest.json`. Inspect the manifest, failed attempts and validation status before interpreting the figures; producing a PDF does not prove that an experiment passed or reproduced the paper.

| Figures | Experiment and what to compare |
|---|---|
| 7, 8 | FP64 `C=A²` on 142 matrices and 12 representatives; compare throughput and paired speedups with cuSPARSE, MOSparse, spECK and TileSpGEMM. |
| 10 | FP16 `C=A²` on 16 matrices; compare FlexSpGEMM with tSparse. Both use input values `1/16`; the paper's original value and configuration policies are unverified. |
| 11 | Compare input-A format storage for CSR, TileSpGEMM, FlexSpGEMM and hypothetical dense tiles; this is not peak GPU memory. |
| 12 | Compare preprocessing and end-to-end latency. The recovered GPU probe measures only a subset of features; the complete GPU feature-extraction protocol is unavailable. Its plot is a proxy and cannot verify the paper's GPU overhead claim. |
| 13 | Compare fixed CUDA-only, adaptive CUDA-only, adaptive Tensor-only and hybrid execution on three matrices. |
| 15 | Evaluate saved LightGBM, LLM and SVM models on a frozen throughput grid; compare achieved/optimal throughput, not classification accuracy. No full model retraining or new grid search is performed. |

Figure 7 also includes H200 in the paper; this A100 workflow does not validate that panel. See [README-DETAILS.md](README-DETAILS.md) for the H200 launcher and its unverified execution status.

## 5. Interpret and check the results

The paper reports average A100 speedups of 2.71×, 1.37×, 1.30× and 1.82× over cuSPARSE, spECK, MOSparse and TileSpGEMM, respectively, on completed pairs. Compare these with `results/a100/speedups.csv`, including the common-matrix counts and both mean definitions. Previous artifact results differ because the public source revision, patches, hardware and completed pairs differ. Numerical and reference failures are retained: a timing is not proof of correctness. The patched source still has a documented `pdb1HYS` numerical failure on the earlier test host; do not claim all matrices pass. The baseline programs also do not all use identical numerical input values; see the detailed README.

For Figure 15, the recovered study uses 999 records for the largest size (rounded to 1000 in the paper), includes overlap between the source pool and test identities, uses model-specific averaging, and has test-informed SVM model selection. The representative LLM's original loader does not enable its adapter. The paper's LLM training-time entry is not substantiated by the recovered evidence. These qualifications, and the Figure 10/12 limitations, prevent claiming exact reproduction of every original conclusion.

The source commit includes post-paper changes and applies a validation repair and a Tensor Core bitmask correction. Changes and validation evidence are documented in [README-DETAILS.md](README-DETAILS.md). The strict `python scripts/audit.py --platform a100 --final` is expected to report the unresolved Figure 10/12 protocols even if all measurements exist.
