# FLOATBench raw time-series audit subset

End-to-end audit of the FLOATBench label-generation pipeline (raw
OpenFAST signal -> stress -> rainflow -> S-N -> Miner -> released damage
label). The audit package is distributed separately from the tabular
dataset (about 1.4 GB, CC-BY-4.0); this folder holds its runner.

## Package contents

- `signals/<tower>/sim_<id>.out`: raw OpenFAST output time series,
  byte-identical to the campaign outputs and only renamed for a flat
  layout (147 files: 49 simulations x 3 towers, 9.3 MB each). The
  OpenFAST version is recorded in each file header (v3.5.2, double
  precision).
- `labels/<tower>_audit_labels.csv`: the corresponding released tabular
  labels (per-section damage), extracted verbatim from the public
  dataset for convenience; the canonical source remains the released
  dataset files.
- `geometry/tower_<tower>.json`: the public tower geometries used for the
  moment-to-stress conversion.
The released post-processing pipeline itself is in this repository,
[`floatbench/openfast/`](../../../floatbench/openfast); the package holds
the data only. Download it (folder `audit_package/`) from the anonymized
data link
<https://osf.io/te9na/?view_only=224887b50912448b871620b3ef96cefc>.

## Subset composition (fixed, deterministic)

`sim_id` 1 to 49 on every tower: the complete 7x7 conditional wave grid
at the cut-in wind level (itself a held-out extrapolation level),
turbulence seed S1, same operating conditions on the three towers. These
simulations span the full wave grid, including its held-out levels, and
include near-zero damage labels, where numerical tolerance handling is
most demanding.

## Pipeline parameters (as used for the released labels)

- Post-processed window: 400 s to 1000 s (600 s).
- Fore-aft bending moment, interpolated to 30 sections.
- S-N: DNV-RP-C203 type E, bilinear, log a = [12.010, 15.350], slopes
  m = [3, 5], knee at 1e7 cycles.
- Thickness correction: t_ref = 25 mm, exponent k = 0.2.
- Palmgren-Miner summation.

## How to run

```bash
pip install numpy pandas scipy matplotlib rainflow ruamel.yaml tqdm
python scripts/analyses/audit/run.py \
    --flagfile=scripts/analyses/audit/config.cfg \
    --package_dir=<audit_package>
```

Reference environment: Python 3.11.6, numpy 2.1.3, pandas 2.3.3,
scipy 1.16.3, matplotlib 3.10.8, rainflow 3.2.0, ruamel.yaml 0.19.1. The
same per-tower deviations were also obtained, digit for digit, on a
fresh Python 3.12.9 environment with numpy 2.5.1 and pandas 3.0.5.

## Expected result

Per-label relative deviation, defined as
`|reproduced - released| / max(|released|, 1e-12)`, across all
49 x 3 x 30 = 4,410 labels:

- On the reference environment: maximum 4.6e-16 (machine precision).
- Acceptance criterion: <= 1e-6 (floating-point rainflow arithmetic
  varies across platforms; bit-for-bit equality is not claimed).

## Chain of custody

The OpenFAST version is recorded in every output file and the tower
input decks are part of the released geometry artifacts. The subset
audits the chain time series -> labels end to end; re-running a
simulation from scratch also needs its turbulence and wave seeds, which
the subset does not include.
