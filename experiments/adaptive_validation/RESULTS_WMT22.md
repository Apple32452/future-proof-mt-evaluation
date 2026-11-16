# Independent WMT22 SQM external-panel follow-up (2026-11-15)

**Status:** externally sourced, *retrospective* public human-score panels; two language directions, each replayed under five policies and 20 randomization seeds. **Preliminary:** summary audit passed on the user's Mac; underlying `trials.jsonl` and original normalized matrices are held locally, not in this GitHub commit. The following numerical outputs are transcribed from the user-supplied terminal logs, not independently replayed by this repository.

## Scientific question

Does the WMT25 Czech–German-developed `certificate_racing_v2` transfer to different publicly released human-scored MT populations? Compare it at matched cost-accounting and fixed simultaneous testing level (familywise alpha = 0.05) with random pair, common-active random, and **adaptations** of inverse-rank-squared and confusion-minimization acquisition from Zouhar et al., *Dynamically Allocating Evaluation Effort for Model Ranking*, Findings of EMNLP 2026 ([paper](https://aclanthology.org/2026.findings-emnlp.360/), [code](https://github.com/zouharvi/evaluation-bandit/blob/main/evaluation_bandit/algorithms.py)).

These published acquisition policies were fitted to **our** sequential pairwise-certification decision rule, not the paper's original ranking objective. A comparison of cost-to-90%-certified-pairs is therefore **not** a faithful reproduction of their method's published benchmark. We also report exploratory fixed-half-budget ranking metrics as objective-aligned checks.

## External input panels, quality-blind system selection

All observed score cells are averages of pre-existing WMT22 SQM human ratings (normalized from 0–100 to [0,1]), and **not** new paid ratings. System identities were chosen by coverage, not performance, and complete common item overlap was required. The loader excludes `refA`, `refB`, etc. The dataset's original `doc_id` is unavailable (literal 'unknown'), so **no document-disjoint validation claim** is justified. The Hugging Face dataset revision was not pinned; exact local normalized CSV hashes are recorded.

| Panel | Items | Systems | Possible mean-score cells | Matrix SHA-256 |
|---|---:|---:|---:|---|
| WMT22 English→German (`en-de`) | 1,161 | 8 | 9,288 | `e5665eec15a72390c891d37cc29f3d6d44ff964b08b9f122b450978846085f52` |
| WMT22 English→Chinese (`en-zh`) | 1,124 | 8 | 8,992 | `510d850760d6ac449e7fe86be9e5279f7253a8232cbc4b0ed20d852b73918459` |

The German panel selected JDExploreAcademy, Lan-Bridge, Online-A, Online-B, Online-G, Online-W, Online-Y, OpenNMT. The Chinese panel selected AISP-SJTU, DLUT, HuaweiTSC, JDExploreAcademy, Lan-Bridge, LanguageX, Manifold, Online-A.

For 8 systems, the 90% target is **26 of 28** signed pairwise decisions. Every run reported reaching that threshold. Each cohort uses 20 seed repetitions × 5 policies = 100 policy trials. Replications are permutations/replays on **the same underlying fixed human panel**, not 20 independent MT datasets.

## Main result: score cells to certify ≥90% of pairs (mean; lower is better)

| Policy | en-de | en-zh |
|---|---:|---:|
| Common-active random | 8,871.80 | 8,756.85 |
| Random pair | 8,444.75 | 8,641.15 |
| **Certificate racing v2** | **8,327.60** | **8,475.50** |
| Published inverse-rank², adapted | 8,350.95 | 8,618.00 |
| Published confusion-minimization, adapted | 8,383.50 | 8,525.25 |

V2's ratio-of-means savings relative to common-active are **6.13%** (en-de) and **3.21%** (en-zh); relative to random pair **1.39%** and **1.92%**. No empirical variation intervals are asserted here: obtain them from actual matching trial rows using `paired_audit.py` below.

Full pairwise resolution costs are nearly census-level for v2: **8,880 / 9,288** (95.6%) en-de and **8,762.6 / 8,992** (97.4%) en-zh. Broad partial coverage is **not** equivalent to early certification of the best system. Report all-pairs results and this high cost, not just cost90.

## Secondary diagnostic: post-hoc top-1 from sample means at half budget

| Policy | en-de | en-zh |
|---|---:|---:|
| Common-active random | 30% | 75% |
| Random pair | 35% | 80% |
| **Certificate racing v2** | **35%** | **75%** |
| Published inverse-rank², adapted | **95%** | **95%** |
| Published confusion-minimization, adapted | 40% | **95%** |

These are **post-hoc, non-certified rankings** obtained from half-budget sample means, evaluated against the same matrix's full mean. They are **not** statistically certified top-1 decisions. Rank-weighted tau at half budget was also better for inverse-rank² than v2 in both directions (en-de 0.928 vs 0.383; en-zh 0.927 vs 0.780). Thus published top-focused allocation rules are not uniformly inferior; objectives matter.

At half budget, v2 had resolved 40.00% of the 28 pairs for en-de and 16.07% for en-zh, compared with 6.43%/4.82% for inverse-rank². Again, this measures certification coverage and is distinct from top-1 ranking fidelity.

## Checks and interpretation

The user's terminal outputs report:
- Both input preparations successful with 8 systems and >1,100 complete shared items each.
- Every policy reached the 90% threshold in all 20 runs for each panel.
- `audit_comparison.py` reported `PASS` twice: 100 trial rows and 30 summary checks **per panel**, with zero observed incorrect issued certificates.
- The audit checks trial-log integrity, summary arithmetic, and high-level constraints; it **does not independently rerun acquisition policies or validate their published-source fidelity**. An absence of observed errors does not prove a 5% familywise error rate.

The panel choice follows failure of an initial WMT23 loader: the public `RicardoRei/wmt-sqm-human-evaluation` dataset holds **2022** annotations, and the available Chinese direction here is **English→Chinese**, not Chinese→English. No WMT23 validation is claimed.

**Evidence boundary:** Two genuinely external directions relative to the WMT25 Czech–German development matrix provide promising limited transfer evidence for **partial pair certification cost**, but neither confirms the method on newer MT generations nor yields independent-language confidence intervals. The adapted published methods' strong top-1 performance underscores the necessity of an objective-matched comparison. Do not retroactively label these as an untouched preregistration or infer per-language significance from 20 correlated random-seed trials.

## Next step: audit paired trials on the user's Mac

The trial-level files were not included in this commit. This keeps private local paths and large logs out of the main repository. Use the **provided new utility**:

```bash
cd ~/Downloads/mt_adaptive_validation
python /path/to/future-proof-mt-evaluation/experiments/adaptive_validation/paired_audit.py \
  results/wmt22_ende --matrix inputs/wmt22_sqm_ende.csv --code-dir . \
  --out results/wmt22_ende/paired_audit.json

python /path/to/future-proof-mt-evaluation/experiments/adaptive_validation/paired_audit.py \
  results/wmt22_enzh --matrix inputs/wmt22_sqm_enzh.csv --code-dir . \
  --out results/wmt22_enzh/paired_audit.json
```

This rehashes the original matrix and recorded source files, checks 20 matched policy comparisons, and reports 10,000 paired bootstrap intervals for randomization **conditional on each fixed dataset**. It does not constitute independent policy replay. Move to independent replay / original-ranking-objective benchmarking before asserting novelty or updating an ACL submission's central claim.

### Primary sources

- WMT SQM release: https://huggingface.co/datasets/RicardoRei/wmt-sqm-human-evaluation
- Zouhar et al. (2026), Findings of EMNLP: https://aclanthology.org/2026.findings-emnlp.360/
- Archived development algorithm in this repository: `experiments/pairwise_certification/`
