# Adaptive MT validation: reproducible comparison runner

This folder contains the adapted published allocation comparators and WMT22 external-validation workflow. It reuses the **frozen** certificate inference in `../pairwise_certification/`. No source-derived claim of novelty or independent statistical validation is implied.

## Source status

The public-data experiment was run on a Mac from the 2026-11-15 study package before these files were uploaded. The original run source was recorded by SHA-256 in each `results/*/manifest.json`. This repository copy ports the import path to the existing sibling inference engine, reformats the runner, and preserves the original policy formulas, randomization, and **external run seed behavior**. It is not byte-identical to all of the user-local source files; those are checked by `replay_check.py` with `--runner-dir` pointing to the Mac's unmodified original folder.

Published policy code adaptations draw on Zouhar et al., Findings of EMNLP 2026, [paper](https://aclanthology.org/2026.findings-emnlp.360/) / [code](https://github.com/zouharvi/evaluation-bandit/blob/main/evaluation_bandit/algorithms.py). Their original objective is ranking quality, **not** 90%-pair certification. Therefore our cost90 numbers are a comparison on our *adapted* downstream objective, not a faithful reproduction of their paper's evaluation.

## Initial verification

From the repository root:

```bash
python -m pip install -r experiments/adaptive_validation/requirements.txt
python -m unittest discover -s experiments/adaptive_validation -p 'test_*.py' -v
```

Tests cover the five policies and paired bootstrap checks. No external human ratings are needed for unit tests.

## Real-data workflow

The raw released SQM annotations are publicly available via Hugging Face. Our loader selects eight non-reference MT systems by judged-item coverage; it does not inspect model quality. Dataset here is WMT22 **en-de** and **en-zh** (not WMT23 or zh-en). We have not claimed document-disjoint sampling.

```bash
cd experiments/adaptive_validation
python prepare_sqm_hf.py --year 2022 --lp en-de --out inputs/wmt22_sqm_ende.csv
python prepare_sqm_hf.py --year 2022 --lp en-zh --out inputs/wmt22_sqm_enzh.csv

python run_comparison.py --matrix inputs/wmt22_sqm_ende.csv --reps 20 --out results/new_ende
python audit_comparison.py results/new_ende
python run_comparison.py --matrix inputs/wmt22_sqm_enzh.csv --reps 20 --out results/new_enzh
python audit_comparison.py results/new_enzh
```

The runner never overwrites existing results, records SHA-256 digests of input and code, and uses the original version's frozen seeds. Results should reproduce the user-reported summaries **if the prepared CSVs are byte-identical**. Compare matrix SHA-256 against `RESULTS_WMT22.md`; an upstream dataset change must be reported, not silently treated as the original cohort.

## Next validation step on original Mac runs

For existing user-local results, use the standalone tools in this folder. These commands **do not** rerun the whole 20×5 experiment:

```bash
cd ~/Downloads/mt_adaptive_validation
python /path/to/future-proof-mt-evaluation/experiments/adaptive_validation/paired_audit.py \
 results/wmt22_ende --matrix inputs/wmt22_sqm_ende.csv --code-dir . \
 --out results/wmt22_ende/paired_audit.json
python /path/to/future-proof-mt-evaluation/experiments/adaptive_validation/paired_audit.py \
 results/wmt22_enzh --matrix inputs/wmt22_sqm_enzh.csv --code-dir . \
 --out results/wmt22_enzh/paired_audit.json

python /path/to/future-proof-mt-evaluation/experiments/adaptive_validation/replay_check.py \
 results/wmt22_ende --matrix inputs/wmt22_sqm_ende.csv --runner-dir . \
 --replicates 0 1 --out results/wmt22_ende/replay_check.json
python /path/to/future-proof-mt-evaluation/experiments/adaptive_validation/replay_check.py \
 results/wmt22_enzh --matrix inputs/wmt22_sqm_enzh.csv --runner-dir . \
 --replicates 0 1 --out results/wmt22_enzh/replay_check.json
```

The paired bootstrap is **within-dataset random-order uncertainty**: it does not treat the 20 permutations as independent language datasets. Replay verifies deterministic rerun using the *same* policy implementation, not correctness of the martingale argument. To establish broader transfer, add distinct language directions and stronger newly released MT systems, and compare the published methods on their original top-ranking objective under matched labeling constraints.

**Do not commit human-score CSVs or full raw trial logs automatically**. Preserve them locally; only share them when the collaborators agree on provenance and redistribution.
