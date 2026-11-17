# Pairwise MT certification: development validation and hindsight cost bound

This page documents the original WMT25 Czech to German development study. The [external WMT22 comparison](../adaptive_validation/RESULTS_WMT22.md) has its own matched policy results and [reproduction instructions](../adaptive_validation/README.md).

This directory records the shared-prefix pairwise racing development study from November 13, 2026. [PR #7](https://github.com/Apple32452/future-proof-mt-evaluation/pull/7), merged that day as `db3b20314b5882470d025f83f3d50784cbc7bcc1`, added the executable implementation, tests, protocols, summaries, and audit reports to `main`. No ZIP import is required to run the tracked code. The input CSV and regenerable JSONL trial records are not bundled in this directory; see Reproduction below.

### Historical archive provenance

The original code-and-results package was `MT_Pairwise_Certification_Study_2026-11-13.zip` (SHA-256: `114d4c9a96349791187258022583703a3fd96e56cc706b8af27d9675a6f92afd`). This identifies the historical archive, not a checksum of the current Git tree. `import_study.py` remains a historical staging utility; it is not a prerequisite for reproduction and should not be rerun over the merged implementation.

## Research question and estimand

For a fixed finite population of N evaluation segments and M systems with bounded human scores Y[i,s] in [0,1], certify signs of all differences between fixed-test-set means:

`Delta(a,b) = N^{-1} sum_i (Y[i,a] - Y[i,b])`.

A certificate is a statistically justified ordering. It is not a guarantee about new documents, annotator noise, or future systems. [METHOD.md](METHOD.md) gives the simultaneous, anytime familywise error-control argument at alpha=0.05 for a fixed complete bounded score matrix, an independently drawn uniform shared item permutation, and pairwise prefix-only inference. Adaptive allocation preserves those constraints; deterministic completion and exact-census ties are handled separately. This mathematical guarantee does not imply that an acquisition heuristic saves labels. The development study tracks 90% pair coverage, full pair coverage, and the leading-pair decision separately.

## Frozen study sequence

- **v1:** shared random segment order; adaptively select system scores; reuse acquired scores across eligible comparisons; sequentially valid pairwise inference.
- **v2:** change the *acquisition rule only* to target comparisons approaching certification; keep the inferential test unchanged.
- **Hindsight allocation diagnostic:** expose all development scores to an offline optimizer and bound the minimal final-prefix costs under the same certification machinery. This is an oracle diagnostic, not a deployable method or a statistical confidence interval.

The v1/v2 paired development replay used Czech-to-German human-score entries from an already examined WMT25 panel. Each published score cell is the mean of two historical ratings; this distinction must be retained when discussing annotation costs. The oracle used five prespecified exposed development orders. The primary target was certification of at least 171 of 190 system pairs.

## Main costs (mean score cells)

| Policy | v2 study: 90% coverage, 20 orders | Five-order oracle diagnostic |
|:--|--:|--:|
| Revised v2 racing | 4,336.05 | 4,323.0 |
| Random-pair allocation | 4,438.80 | 4,424.0 |
| Common-active random sampling | 4,550.4 | 4,551.8 |
| Offline hindsight optimum | Not evaluated | 4,114.2 |
| Numerical lower bound | Not evaluated | 4,113.2 |

The v2 development gains at 90% coverage are 2.31% relative to random-pair allocation and 4.71% relative to common-active. **The prespecified development gate of at least 10% savings against both baselines was not met.** This was an internal project threshold fixed before v2 results, not independent preregistration or a publication standard. The 20 orders measure randomization variability on one already-exposed matrix, not generalization across languages or datasets. See [RESULTS_V2.md](RESULTS_V2.md). The numerical oracle result permits at most about 4.85% further mean savings compared with v2 *in those five runs under those exact constraints*. It is not a general impossibility statement. These are floating-point MILP bounds under fixed inference, shared orders, four-row warmup, and per-system prefix acquisitions; they are neither statistical confidence intervals nor exact-rational certificates. The hindsight optimizer sees all scores and is not a deployable acquisition policy. See [RESULTS_ORACLE.md](RESULTS_ORACLE.md).

The final archived suite reported 63 passing tests, including nine oracle tests; the v2-stage report recorded 54 before those additions. The separate oracle-output auditor reported 22,122 checks. These historical checks support implementation consistency and arithmetic, not a new theorem or future generalization. V2 produced an incorrect certificate in 2/100 all-tied synthetic runs and 0/20 real replays; neither result empirically establishes a 5% upper error bound. See [RESULTS_V2.md](RESULTS_V2.md) for uncertainty intervals.

Full-ranking cost is a separate outcome: v2 averaged 4,592.25 of 4,620 cells, compared with 4,590.70 for random pair. Partial-coverage savings do not establish cheaper complete rankings. No new human annotations or annotation-time/currency measurements were collected.

## Interpretation

The hindsight bound quantifies the room for improving allocation on the sampled development orders while keeping the certificate rule fixed. Its constraints do not cover a new inference procedure or a different item order.

Pair coverage and winner certification measure different objectives. A winner requires a strict certificate against every rival.

## Reproduction

Tracked entry points are `pairwise_racing.py`, `certificate_racing_v2.py`,
`validate.py`, `validate_v2.py`, and `run_oracle_diagnostic.py`, with
`test_*.py` and the three output auditors below. The repository tracks frozen
summaries/manifests and audit reports, but its root `.gitignore` excludes
`*.jsonl`. Thus a fresh checkout lacks `validation_v1/trials.jsonl`,
`validation_v2/{new_trials,comparison_trials}.jsonl`, and
`oracle_v1/runs.jsonl`. Running the v2 or oracle driver immediately would fail;
regenerate v1, then v2, then the oracle.

Run the following in Bash from the repository root. Use Python 3.12 (the
historical manifests record 3.12.14) and the pinned NumPy/SciPy dependencies.
A temporary copy keeps the checkout and its frozen artifacts unchanged:

```bash
set -euo pipefail
study_work=$(mktemp -d)
cp -R experiments/pairwise_certification "$study_work/study"
cd "$study_work/study"
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Fetch only the exact, already-public development input pinned in
[inputs/provenance.json](inputs/provenance.json); retain it locally:

```bash
python3 - <<'PY'
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

p = json.loads(Path("inputs/provenance.json").read_text())
url = p["retrieved_from"].replace(
    "https://github.com/", "https://raw.githubusercontent.com/"
).replace("/blob/", "/", 1)
with urlopen(url) as response:
    raw = response.read()
if len(raw) != p["size_bytes"] or hashlib.sha256(raw).hexdigest() != p["sha256"]:
    raise SystemExit("Pinned input size/checksum mismatch")
Path("inputs", p["local_file"]).write_bytes(raw)
PY
python3 -m unittest discover -v
```

The input is `inputs/cs-de_DE_human_scores.csv`, SHA-256
`f6de1e8da36fb67c7d99ec354e8807a016045c2d5b420f3ab8aeb72816bc279d`.
It contains 231 items and 20 systems, normalized from the known 0–100 scale.
Without this optional input, the historical-matrix unit test skips; full
development replays and output audits require it.

In this temporary copy only, retain the original result directories under
separate names and generate fresh outputs at the paths expected by the oracle
driver and auditor (which hard-code `validation_v1/` and `validation_v2/`):

```bash
mv validation_v1 archived_validation_v1
mv validation_v2 archived_validation_v2
mv oracle_v1 archived_oracle_v1

python3 validate.py --synthetic-reps 100 --real-reps 20 \
  --real-csv inputs/cs-de_DE_human_scores.csv --out validation_v1
python3 audit_outputs.py --results validation_v1 > reproduced_audit_v1.json

python3 validate_v2.py --baseline-dir validation_v1 \
  --real-csv inputs/cs-de_DE_human_scores.csv --out validation_v2
python3 audit_v2_outputs.py --baseline validation_v1 \
  --results validation_v2 > reproduced_audit_v2.json

python3 run_oracle_diagnostic.py --out reproduced_oracle
python3 audit_oracle_outputs.py --run-dir reproduced_oracle \
  --out reproduced_oracle_audit.json
```

These are full development runs: 2,480 v1 baseline records, 620 new v2 records
(3,100 combined, including the reused baselines), and five oracle optimizations.
All driver output directories must be new. Keep the pinned sources, protocols,
seeds, and repetition counts unchanged; do not bypass checksum failures.
The v2 auditor checks the regenerated v1 trial hash against the frozen reference.
New manifests record the reproduction environment and paths, so they are not
expected to be byte-identical to historical manifests.

Compare the new summaries and audit outcomes with `archived_validation_v1/`,
`archived_validation_v2/`, `archived_oracle_v1/`, and the tracked audit reports.
The archived directories in a Git checkout still lack JSONL traces; auditing
those historical directories directly requires the matching archived traces.
The commands above instead audit freshly regenerated records and leave the
historical records intact. Reproduction does not turn this exposed development
study into independent validation.
