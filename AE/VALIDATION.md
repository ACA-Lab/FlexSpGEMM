# AE documentation validation, 22 October 2026

This record distinguishes newly executed checks from inspection of producer results.
The paper hash is recorded in `AE_GUIDE.md`.

| Check | Evidence | Outcome |
|---|---|---|
| Relocate producer archive into a new `/tmp` tree | `documentation-validation/relocation-extract.json`, `relocation-stages.json` | Archive hash verified; source bootstrap and input hashes passed |
| Existing artifact parser/integrity tests | `documentation-validation/relocation-tests.log` | 13 tests passed |
| Public quick-start command, freshly built | `documentation-validation/quickstart-results.json`, `quickstart-validated.log` | `m_t1`, `mult_dcop_03`: both cuSPARSE structure/value checks passed; process exit 0 |
| Clean minimal Python environment | `documentation-validation/fresh-environment-summary.json`, `fresh-environment-quickstart-results.json` | New isolated venv; dependency installation 35.68 s; pip check passed; fresh GPU quick check 61.07 s, both matrices passed |
| Known numerical failure | `documentation-validation/quickstart-known-failure-results.json`, native `pdb1HYS` logs | 544 mismatched values; quick-start exit 2 |
| Unavailable reference | `documentation-validation/quickstart-case39-results.json`, `quickstart-case39.native.log` | cuSPARSE unavailable; quick-start exit 2 |
| Key-result collector, integrity audit and plots | `documentation-validation/key-validation-stages.json`, `key-audit.json` | 1704 main and 216 supplemental attempt slots; 2156 evidence files checked |
| New coverage checker | `documentation-validation/key-tests.log` | 4 tests passed, including duplicate/missing identity and damaged-log checks |
| Fresh dependency resolution | `documentation-validation/dependency-resolution.json` | Full ML dependency set resolves to 70 packages; no fresh installation performed |

The initial relocated tests used the existing Python 3.12.9 dependency environment.
A subsequent test used a new isolated venv with `requirements.txt`, no system/user
site packages and no inherited PYTHONPATH. Its dependency installation and
cuSPARSE-validated GPU quick check passed; exact packages are captured in
`documentation-validation/fresh-environment-requirements.lock.txt`. The full ML/Torch
environment was not freshly installed.
They compiled source in the new directory and ran physical GPU 1,
NVIDIA A100-SXM4-80GB, CUDA 11.8, driver 580.178.04.
The default public quick-start took approximately 62 seconds after dependency
installation, with local input files. This duration excludes a cold download.

The old archive used as the relocation baseline was
`FlexSpGEMM-AE-20261019T210725Z.tar.gz`, SHA-256
`193228480d5f8da9843257e9598c55578da2020879a047cfef2dd5bd356ba4d5`.
The new quick-start helper was subsequently copied to that extracted tree and
tested through its public command line. Its tested SHA-256 is
`8872cbe312ca86a15a7495c7a4104f899944bd743de1cfb913b55c7468aad818`.
The new release includes the helper, key-result launcher/checker, guide and
validation evidence. Archive verification is a separate packaging check.

No new full 142-matrix GPU campaign, full baseline rebuild, H200 execution,
fresh machine provisioning, or complete from-scratch model training was
performed for this documentation task. Core coverage and plot checks examine
the existing producer measurements and retain their validation failures.
The guide does not claim all 142 outputs are numerically correct.

`safetensors` was 0.5.3 in the reused execution environment; the declared full
ML requirements request 0.8.0. Dependency resolution is not execution parity
testing of those two versions. The documented model/library protocol and
reference outputs must be checked when establishing a new environment.
