# Future-Proof Machine Translation Evaluation

**Researchers:** Taewoon Choi and Min-Hua Luo

## Research Question

Can a small, carefully selected subset of machine translation test
sentences preserve the full-test-set scores and rankings of unseen
MT systems?

## Core Idea

We compare:

1. Random sampling
2. Difficulty-based selection
3. System-disagreement selection
4. Diversity-aware selection
5. Difficulty + disagreement + diversity

## Evaluation

We will measure:

- Mean Absolute Error (MAE)
- Spearman rank correlation
- Kendall rank correlation
- Pairwise ranking accuracy

## Data

We use publicly available WMT25 General Machine Translation data.

Raw WMT data are kept outside this repository.
