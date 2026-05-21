# NovaScore — case study

> A credible, end-to-end ML credit scoring pipeline on the Home Credit Default
> Risk public dataset. Documents the architecture, training methodology,
> ensemble construction, calibration, fairness analysis, and the journey from
> synthetic gig-worker data to a real-world benchmark.

## 1. Problem statement

Conventional credit scoring (FICO, CIBIL) penalizes "credit-invisible" workers
who lack documented borrowing history. NovaScore was built to demonstrate an
**ML scoring engine that surfaces signal from alternative behavioral data** —
trip activity, financial transactions, and bureau-balance history — and
calibrates the model output into an actionable 300–950 score with four
decision bands (Platinum / Gold / Silver / Bronze). A built-in fairness audit
quantifies group-level disparities (gender, age, car ownership, family status)
and applies per-group threshold optimization to reduce equal-opportunity gaps.

## 2. Dataset journey

### 2.1 Synthetic gig-worker data (Phase 2 → 4.5)

The original GrabHack 2025 brief asked for a model on Grab's
ride/transaction/merchant data. We could not redistribute Grab's actual data
publicly, so we built a faithful synthetic generator
(`ml/novascore/data/synthesize.py`):

- **10K synthetic users** with realistic distributions: log-normal trip
  distances and fares, Poisson trip counts, beta-distributed latent default
  PD (~10% population rate), normally distributed ratings and safety scores
  with means tied to PD.
- **Bias injection** (so the fairness module had something real to mitigate):
  young + female + bicycle users carry +0.15 base default PD, producing a
  19pp gap in observed user-level positive rate (12% overall, 31% in biased
  subgroup).
- **Schema match**: trips.parquet + txns.parquet exactly mirror the legacy
  Colab pipeline's schema so the existing preprocessing code drops in.

We ran a 15-trial random hyperparameter sweep on this synthetic data (Phase
4.5; 11 trials completed before a forced pivot). **Result: AUROC plateaued
at 0.60–0.62 across all trials, regardless of d_tab, dropout, lr, or graph
inclusion.** The bottleneck was not the architecture — it was the
signal-to-noise ratio of "any bad txn in 90 days" labeling combined with the
synthesizer's distributional assumptions. We could have raised the apparent
numbers by tuning the synthesizer to give the model more help, but that would
have been dishonest.

The partial Phase 4.5 sweep log is archived at
`ml/results/_archive_phase4_5_synth/` (local-only; not committed). The
trials that finished:

| Trial | d_tab | dropout | lr | val | test |
|------:|------:|--------:|---:|----:|-----:|
| 1 | 384 | 0.3 | 3e-4 | 0.6149 | 0.6065 |
| 2 | 384 | 0.1 | 1e-4 | 0.6118 | 0.6062 |
| 3 | 128 | 0.3 | 3e-4 | 0.6138 | 0.6146 |
| 4 | 256 | 0.1 | 3e-4 | 0.6056 | 0.6077 |
| 5 | 256 | 0.3 | 1e-4 | 0.6120 | 0.6118 |
| 6 | 128 | 0.1 | 1e-3 | 0.6162 | 0.6100 |
| 7 | 256 | 0.1 | 1e-4 | 0.6137 | 0.6108 |
| 8 | 256 | 0.2 | 3e-4 | 0.6147 | 0.6107 |
| 9 | 128 | 0.2 | 3e-4 | 0.6193 | 0.6155 |
| 10 | 384 | 0.3 | 3e-4 | 0.6138 | 0.6060 |
| 11 | 384 | 0.2 | 1e-4 | 0.6021 | 0.6046 |

Best val AUROC: **0.6193** — a flat ceiling.

### 2.2 Pivot to Home Credit Default Risk (Phase 4.6)

To get a credible headline number we pivoted to a public dataset where the
ground truth is real:

- **Home Credit Default Risk** (Kaggle competition, 2018; 307,511 anonymized
  loan applications with binary TARGET label).
- Five linked tables: bureau, bureau_balance, previous_application,
  installments_payments, application_train.
- Same architecture; new data wiring.
- Same calibration anchors and decision bands.

We used a stratified **60,000-applicant sample** for compute budget reasons
(see §6 Limitations). The 70/15/15 split (42K train / 9K val / 9K test) is
stratified on TARGET.

## 3. Methodology

### 3.1 Feature engineering (`ml/novascore/data/home_credit.py`)

**Tabular features** (final count: 80 after SelectKBest filtering):

- **application_train** columns + derived ratios:
  - `age_years = -DAYS_BIRTH/365`
  - `employed_years = -DAYS_EMPLOYED/365` (sentinel 365243 → NaN)
  - `credit_to_income`, `annuity_to_income`, `credit_to_goods`
- **bureau aggregations** per applicant: count, mean, max, min of DAYS_CREDIT,
  CREDIT_DAY_OVERDUE, AMT_CREDIT_SUM(_DEBT/_LIMIT/_OVERDUE),
  AMT_CREDIT_MAX_OVERDUE, CNT_CREDIT_PROLONG, plus `bureau_n_active`.
- **previous_application aggregations**: count, mean, max of AMT_ANNUITY,
  AMT_APPLICATION, AMT_CREDIT, DAYS_DECISION, CNT_PAYMENT; plus
  `prev_n_approved`, `prev_n_refused`, `prev_refusal_rate`.
- **installments_payments aggregations**: sum/mean of AMT_INSTALMENT,
  AMT_PAYMENT, `payment_diff`, `payment_ratio`, `days_late`, `was_late`.
- **Categorical handling**: one-hot encode all columns with ≤30 unique values;
  drop higher-cardinality strings.
- **Missing-value handling**: drop columns with >80% missingness, median-impute
  the rest.
- **Feature selection**: SelectKBest with f_classif against TARGET keeps the
  top 80 features so the FT-Transformer's O(n²) attention stays tractable.

**Sequence features** (bureau monthly STATUS, shape `(60, 8)`):

For each applicant, we join bureau ↔ bureau_balance via `SK_ID_BUREAU` and
collect monthly status records from the past 60 months. STATUS values (`0–5`,
`C`, `X`) are one-hot encoded into 8 channels, summed across the applicant's
multiple bureau loans per (applicant, month) cell, and per-channel
z-score-normalized. Months without data remain zeroed (the standardized
"unknown" state).

### 3.2 Architecture (preserved across the synth → real pivot)

```
Hybrid model:
  ├─ FT-Transformer  (d_tab=256, 2 transformer layers, dropout=0.3)
  │     Tokenize each scalar feature → d_tab-dim → CLS-pooled
  ├─ TCN encoder    (d_seq=128, 3 dilated conv blocks, kernel=3)
  │     Channels [trips, dist, dur, cancels, rating, earnings, spend, txns,
  │     merchants] (synth) OR bureau STATUS one-hot 8 channels (Home Credit)
  ├─ Fusion MLP    (d_tab + d_seq → 128 → 1)
  └─ Output:        raw logit (paired with BCEWithLogitsLoss)
```

Node2Vec was dropped from the production model: in Phase 4.5 the graph tower
added ~0.001 AUROC at 4× the compute, and the Home Credit schema has no
natural user–merchant bipartite graph. The module remains in
`ml/novascore/models/node2vec_embed.py` for ablation.

Final parameter count: **1,922,177**.

### 3.3 Training setup

- **Optimizer**: AdamW(lr=3e-4, weight_decay=1e-4)
- **Schedule**: CosineAnnealingLR over 10 epochs
- **Loss**: BCEWithLogitsLoss with `pos_weight` from class imbalance (~11.5×
  for Home Credit's 8% positive rate)
- **Early stopping**: patience=4 on val AUROC
- **Batch size**: 128 (reduced from 512 after macOS MPS OOM at 60K sample)
- **Hardware**: Apple M-series MPS backend, fp32. (CUDA AMP path preserved.)
- **Memory guardrails** added after the OOM: `gc.collect()` and
  `torch.mps.empty_cache()` between trials and between epochs.

### 3.4 Hyperparameter sweep

4 random-search trials sampled from:
- `d_tab ∈ {128, 256, 384}`
- `dropout ∈ {0.1, 0.2, 0.3}`
- `lr ∈ {1e-4, 3e-4, 1e-3}`

All four trials completed without crashes. Per-trial sweep history
(`ml/results/hp_sweep.json`):

| Trial | d_tab | dropout | lr | val AUC | test AUC | params | wall (s) |
|------:|------:|--------:|---:|--------:|---------:|-------:|---------:|
| 1 | 256 | 0.3 | 3e-4 | **0.7364** | **0.7253** | 1.92M | 666 |
| 2 | 256 | 0.2 | 1e-4 | 0.7286 | 0.7197 | 1.92M | 789 |
| 3 | 256 | 0.1 | 1e-4 | 0.7324 | 0.7222 | 1.92M | 771 |
| 4 | 384 | 0.3 | 1e-3 | 0.6790 | 0.6696 | 3.93M | 1300 |

Best by val AUROC: **Trial 1** (`d_tab=256, dropout=0.3, lr=3e-4`). The
larger d_tab=384 with the higher learning rate (Trial 4) underperformed —
the model couldn't converge in 10 epochs and the higher LR with high dropout
appears to have destabilized training.

### 3.5 LightGBM baseline

Single-config training with strong defaults:
- learning_rate=0.05, num_leaves=64, min_child_samples=50
- bagging_fraction=0.8, feature_fraction=0.9
- scale_pos_weight from class imbalance
- 3000 boost rounds with early_stopping(200)

Result: **best_iter=110, val AUROC=0.7593, test AUROC=0.7450** in ~5
seconds. LightGBM is a remarkably strong tabular baseline; the hybrid
narrowly trailed it on this dataset (0.7253 vs 0.7450 test).

### 3.6 Ensemble

We compared two ensembling strategies on the held-out validation set:
1. **Simple 50/50 average** of hybrid + LightGBM probabilities
2. **Val-AUROC-weighted average**: weights ∝ `max(val_auc - 0.5, ε)`

The val-weighted strategy won: hybrid weight 0.477, LightGBM weight 0.523.
**Ensemble val AUROC = 0.7585, test AUROC = 0.7456** — slightly beating
LightGBM alone (+0.0006), comfortably beating the best hybrid (+0.0203).

### 3.7 Calibration

PD → score: `score = A − B · logit(PD)` with anchors PD=(0.01, 0.20) →
score=(900, 650) → A=540.7, B=78.0. Empirical refinement rescales the model's
PD distribution so its 20th/80th percentiles align with the PDs corresponding
to score 600/800; final params committed to `ml/results/calibration.json`.

**Decision-band distribution on full 60K sample**:
- Platinum (≥800): 20.0%
- Gold (700–799): 28.8%
- Silver (600–699): 31.2%
- Bronze (<600): 20.0%

A well-spread distribution; no pathological collapse into one band.

**Calibration-direction bug fix** (recalibrated post-Phase 4.6): the original
empirical-refinement function in `ml/novascore/calibration.py` had its targets
swapped (`q_low` mapped to `logit(p600)` instead of `logit(p800)`), inverting
the PD → score direction. Symptom: high-PD applicants were getting score 950
and low-PD applicants score ~430. The model itself was unaffected; only the
post-hoc PD → score map was wrong. Fixed in commit following the Phase 4.6
landing; the saved `all_probs.npy` was re-calibrated in place via
`ml/scripts/recalibrate.py` (no retraining needed). Sanity check: lowest-PD
test applicant → 902, highest-PD → 432.

## 4. Fairness analysis

We measured four protected attributes from the application table:
- `gender` (M / F; XNA dropped)
- `age_bucket` (18-25 / 26-40 / 41-55 / 56+, derived from DAYS_BIRTH)
- `own_car` (FLAG_OWN_CAR Y/N)
- `family_status` (Married / Single / Civil marriage / Separated / Widow)

Metrics per attribute at the global threshold that achieves overall TPR=0.80
(threshold = 0.469):

| Attribute | Demographic Parity Ratio | ΔTPR | ΔFPR | EOD |
|-----------|--------:|--------:|--------:|--------:|
| gender | 0.000 | 0.046 | 0.293 | 0.293 |
| age_bucket | **0.237** | **0.323** | 0.457 | 0.457 |
| own_car | 0.931 | 0.006 | 0.018 | 0.018 |
| family_status | 0.000 | 0.187 | 0.315 | 0.315 |

The DPR=0.000 readings on gender and family_status indicate at least one
subgroup has zero positive predictions at the chosen threshold (likely the
small subgroup `XNA` for gender or `Widow` for family_status). Age bucket
has the largest meaningful equal-opportunity gap (ΔTPR=0.32), so we mitigated
on that attribute.

### Per-group threshold optimization on age_bucket

Target TPR = 0.80, grid-search per group, 2pp tolerance:

| age_bucket | TPR before | Threshold | TPR after | Score adjustment |
|-----------:|-----------:|----------:|----------:|-----------------:|
| 18-25 | 0.940 | 0.615 | 0.801 | **+46.1** |
| 26-40 | 0.874 | 0.535 | 0.797 | +20.5 |
| 41-55 | 0.748 | 0.430 | 0.799 | −12.3 |
| 56+ | **0.617** | 0.340 | 0.801 | **−42.1** |

**Equal-opportunity intervention summary**: 18-25 had a very high TPR (0.94)
— the model was over-eager to flag young applicants who actually default.
Raising their threshold to 0.615 brings their TPR down to 0.80 and gives
them a +46 score boost at the decision boundary. 56+ had the opposite
problem: TPR of only 0.62 means the model was missing the actual defaulters
in that group. Lowering their threshold to 0.34 raises their TPR to 0.80,
which in score-space terms is a −42 adjustment — older applicants are
treated more strictly. **ΔTPR drops from 0.323 to 0.0044** — a 73× reduction.

### Honest caveat on equal-opportunity mitigation

Equal-opportunity fairness equalizes true-positive rates, which is the
academically defensible interpretation of "treating similar default risks the
same regardless of group." But the *direction* of the score adjustment can be
counterintuitive: in this run, older applicants get LOWER adjusted scores
because the model under-detected their defaulters before mitigation.
Real-world deployment should pair this metric with demographic-parity and
predictive-parity checks and a human-judgment review — see `fairness.py` for
all five metric implementations.

## 5. Wall-clock budget

Phase 4.6 end-to-end on a 60K-applicant sample (Apple M-series MPS, batch
size 128, 10 epochs max, patience 4):

| Stage | Wall time |
|-------|----------:|
| Data prep + bureau merges + feature selection | ~30s |
| LightGBM baseline (3000 boost rounds, early-stop @110) | ~4s |
| Hybrid HP sweep (4 trials × ~12 min avg) | 58 min |
| Ensemble + calibration + fairness + plots + artifacts | ~30s |
| **Total** | **~60 min** |

The compute spent on the hybrid sweep represents the bulk of the time, and
for this dataset the marginal AUROC contribution of the hybrid over
LightGBM-alone is small (+0.0006). LightGBM-only deployment is a defensible
alternative; the hybrid's value here is as a deep-learning complement that
the ensemble weights at 47.7%.

## 6. Limitations and future work

1. **60K-of-307K sample** — full-dataset training would push AUROC ~0.78
   based on community Kaggle results. We sampled for compute budget on
   commodity Apple Silicon. The pipeline runs on the full dataset with
   `--sample-n 0` (untested in this iteration).
2. **POS_CASH_balance and credit_card_balance untouched** — these two tables
   would add millions of behavioral signals. Skipped for compute reasons.
3. **Single-attribute mitigation** — only `age_bucket` was mitigated;
   `family_status` (ΔTPR=0.19) is left for future work. Intersectional
   mitigation across (gender × age × family_status) requires careful subgroup
   sizing.
4. **Ensemble pool of 2** — adding XGBoost or CatBoost to the ensemble is
   a cheap incremental improvement we did not run.
5. **macOS MPS OOM on 60K with batch 512** — root cause: MPS buffer growth
   across HP trials. Mitigated by `gc.collect()` + `torch.mps.empty_cache()`
   and reducing batch size to 128. On Linux + CUDA the original batch size
   should work; we did not benchmark there.
6. **The Phase 4.5 synthetic ceiling is real** — readers reproducing the
   synth path will see val AUROC plateau at ~0.62, not because of code bugs
   but because the synthesizer's label generation is the bottleneck.
   `ml/novascore/data/synthesize.py` is preserved as-is so this is a
   reproducible finding, not a moving target.

## 7. References

- Gorishniy et al., *Revisiting Deep Learning Models for Tabular Data*, NeurIPS 2021 (FT-Transformer)
- Bai, Kolter, Koltun, *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*, 2018 (TCN)
- Ke et al., *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*, NeurIPS 2017
- Hardt, Price, Srebro, *Equality of Opportunity in Supervised Learning*, NIPS 2016
- Home Credit Default Risk Kaggle competition: https://www.kaggle.com/c/home-credit-default-risk

## 8. Credits

**Anupam Kumar**, **Anish Kishore** (primary contributor — completed the
broken ML pipeline, built the FastAPI backend, built the Next.js frontend,
ran the dataset pivot and the headline benchmarking), and **Swaraj Thakur**.
All three at BIT Mesra.

Originally a Grab AI National Hackathon 2025 (semi-finalists) project;
rebuilt to a public-data benchmark for honest performance evaluation.
