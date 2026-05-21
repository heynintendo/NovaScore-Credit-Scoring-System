# NovaScore

**An equitable credit scoring engine demonstrated on the Home Credit Default Risk public dataset (307K real anonymized loan applications).**

NovaScore is a hybrid ML credit scoring system that combines an FT-Transformer
(tabular features), a Temporal Convolutional Network (monthly bureau status
sequences), and a LightGBM baseline into an ensemble that produces a calibrated
NovaScore in the 300–950 range, mapped to Platinum / Gold / Silver / Bronze
decision bands. A built-in fairness audit measures disparities across four
protected attributes (gender, age bucket, car ownership, family status) and
applies per-group threshold optimization to equalize true-positive rates.

> *"Respecting those who carry trust, with credit that carries them."*

Originally a GrabHack national semi-finalist project for gig-economy partners,
the work was later rebuilt on the public Home Credit Default Risk dataset for
credible benchmarking. See the [Dataset Journey](#dataset-journey) section
below — we are open about what worked and what didn't.

---

## Headline Results (Phase 4.6, Home Credit, seed=42, 60K-applicant sample)

| Model | Val AUROC | Test AUROC |
|------|----------:|-----------:|
| LightGBM (tuned baseline) | 0.7593 | **0.7450** |
| Hybrid FT-Transformer + TCN (best of 4 trials) | 0.7364 | 0.7253 |
| **Ensemble (val-AUROC-weighted)** | **0.7585** | **0.7456** |

The ensemble beat both component models (narrowly over LightGBM at +0.0006,
clearly over the hybrid at +0.0203). Ensemble weights: hybrid 0.477, LightGBM
0.523. All numbers are from the **same training run** committed in
`ml/results/metrics.json` — no cherry-picking, no fishing for lucky seeds.

### Fairness (mitigation attribute: `age_bucket`, target TPR = 0.80)

| Attribute | ΔTPR before | ΔTPR after | Note |
|-----------|------------:|-----------:|------|
| age_bucket | 0.3229 | **0.0044** | Mitigated (per-group thresholds equalize to ~0.80 TPR) |
| family_status | 0.1874 | 0.1874 | Not mitigated in this run; future work |
| gender | 0.0462 | 0.0462 | Below mitigation threshold |
| own_car | 0.0060 | 0.0060 | Negligible disparity |

Per-group threshold map applied to age_bucket: `18-25→0.615`, `26-40→0.535`,
`41-55→0.430`, `56+→0.340`. These translate to score adjustments of
`+46.1, +20.5, −12.3, −42.1` points respectively — the equal-opportunity
intervention boosts young applicants (whose TPR was already very high at 0.94)
back down to the target and pulls older applicants up (their TPR rose from
0.62 to 0.80).

### Score distribution across decision bands

| Band | Range | Share | Policy |
|------|-------|------:|--------|
| Platinum | 800–950 | 20.0% | Auto-approve, large limit |
| Gold | 700–799 | 31.2% | Standard approve, medium limit |
| Silver | 600–699 | 28.8% | Manual review, repayment coaching |
| Bronze | <600 | 20.0% | Decline, saving plans |

Plots: `ml/results/roc_curve.png`, `calibration_plot.png`,
`fairness_before_after.png`, `feature_importance.png`,
`score_distribution.png`.

---

## Dataset Journey

NovaScore was originally designed around synthetic gig-worker behavioral data
(trips + transactions + user–merchant network) for the GrabHack 2025 brief.
After the modular ML pipeline was completed and a 15-trial hyperparameter
sweep ran on a 10K-user synthetic dataset, **AUROC plateaued at ~0.61** —
not a code bug, but a signal-to-noise ceiling intrinsic to the synthesizer's
label generation (`any bad txn in 90 days`) combined with the limited number
of trips per user.

Rather than fabricate stronger numbers, we **pivoted to the Home Credit
Default Risk public dataset** (307K real anonymized loan applications, Kaggle)
for credible benchmarking. The same architecture (FT-Transformer + TCN +
LightGBM ensemble + calibration + threshold-based fairness mitigation) carried
over; only the data wiring and the protected attributes changed. The
synthetic generator is preserved at `ml/novascore/data/synthesize.py` for
reproducibility and ablation comparison; the partial Phase 4.5 sweep results
are archived at `ml/results/_archive_phase4_5_synth/` (local only).

Node2Vec graph embeddings were dropped from the production model — they
added compute cost without improving AUROC at this scale, and the Home Credit
schema does not contain a natural user–merchant bipartite graph anyway. The
graph tower remains in `ml/novascore/models/node2vec_embed.py` for ablation.

---

## Architecture

```
   ┌──────────────┐      ┌──────────┐      ┌────────────┐
   │  application │      │  bureau  │      │ bureau_bal │
   │  + linked    │      │  + prev  │      │  monthly   │
   │  table aggs  │      │  + inst  │      │  STATUS    │
   └──────┬───────┘      └─────┬────┘      └─────┬──────┘
          │ 80 tabular features │                 │ 60 months × 8 channels
          ▼                     ▼                 ▼
     ┌────────────────────────────────┐    ┌──────────────┐
     │      FT-Transformer (d=256)    │    │ TCN encoder  │
     │  per-feature tokenization →    │    │ (3 dilated   │
     │  2-layer TransformerEncoder    │    │  conv blocks)│
     └────────────────┬───────────────┘    └──────┬───────┘
                      │ 256-d                     │ 128-d
                      └─────────┬─────────────────┘
                                ▼
                  ┌─────────────────────────────┐
                  │ fusion MLP (256+128 → 128 → 1)
                  └──────────────┬──────────────┘
                                 ▼ raw logit
                  ┌─────────────────────────────┐
                  │ PD = sigmoid(logit)         │
                  │ score = A − B · logit(PD)   │
                  │ clip [300, 950]             │
                  │ band: Plat/Gold/Silv/Bronze │
                  └─────────────────────────────┘

           Parallel: LightGBM on same 80 tabular features
           Ensemble: val-AUROC-weighted average of hybrid + LGB probs
```

Architecture invariants preserved throughout the rebuild:
- 13-week → 60-month sequence (synth → Home Credit), single TCN encoder
- 9 → 8 sequence channels (synth weekly aggregates → bureau STATUS one-hot)
- Calibration anchors fixed at PD=(0.01, 0.20) → score=(900, 650)
- Decision bands: Platinum (≥800), Gold (700–799), Silver (600–699), Bronze (<600)
- Score clip [300, 950]

Final hybrid parameter count: **1,922,177** (`sum(p.numel())` of best
checkpoint). The original deck claimed ~8M; the actual leaner implementation
gets the same fusion behavior with fewer parameters and is deployable on
commodity hardware.

---

## Repository layout

```
.
├── ml/
│   ├── novascore/
│   │   ├── data/
│   │   │   ├── home_credit.py       — Home Credit ingestion + feature engineering
│   │   │   ├── synthesize.py        — synthetic gig-worker generator (preserved)
│   │   │   ├── preprocessing.py     — synth-data preprocessing utilities
│   │   │   └── sequences.py         — weekly-sequence builder for synth path
│   │   ├── models/
│   │   │   ├── hybrid.py            — HybridModel fusion architecture
│   │   │   ├── ft_transformer.py    — FT-Transformer + FeatureTokenizer
│   │   │   ├── tcn.py               — Chomp1d + TemporalBlock + TCNEncoder
│   │   │   ├── lightgbm_baseline.py — gradient-boosted baseline
│   │   │   └── node2vec_embed.py    — Node2Vec module (kept for ablation)
│   │   ├── home_credit_pipeline.py  — Phase 4.6 end-to-end training pipeline
│   │   ├── train.py                 — synth-data training + shared utilities
│   │   ├── sweep.py                 — synth-data HP sweep (Phase 4.5 reference)
│   │   ├── calibration.py           — PD → score map + Platinum/Gold/Silver/Bronze
│   │   ├── fairness.py              — DPR / EOD / ΔTPR + per-group threshold opt.
│   │   ├── evaluate.py              — plots + predictions DataFrame
│   │   ├── io.py                    — artifact save/load
│   │   └── cli.py                   — `novascore train|evaluate|score|sweep`
│   ├── tests/                       — pytest test suite
│   ├── data/{synthetic,home_credit}/— data dirs (not committed)
│   ├── results/                     — committed metrics, plots, checkpoint
│   ├── notebooks/                   — legacy Colab notebook
│   ├── docs/REPORT.md               — long-form case study
│   ├── pyproject.toml
│   └── requirements.txt
├── api/                             — FastAPI inference backend (HF Spaces)
├── frontend/                        — Next.js + shadcn/ui demo
├── assets/                          — architecture diagrams
├── .github/workflows/               — CI
└── README.md
```

---

## Reproducing the headline number

```bash
# 1. Clone + install
git clone <repo>
cd NovaScore-Credit-Scoring-System
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ml/

# 2. Set up Kaggle API credentials (one-time)
#    https://www.kaggle.com/docs/api  →  ~/.kaggle/kaggle.json
#    Then accept the Home Credit competition rules on kaggle.com.

# 3. Train end-to-end on Home Credit (downloads + unzips ~700MB if missing)
OMP_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
  novascore train --dataset home_credit \
                  --sample-n 60000 \
                  --n-trials 4 \
                  --epochs 10

# 4. Rebuild plots from saved artifacts (no retraining)
novascore evaluate --results-dir ml/results

# 5. Score one applicant from a JSON file
novascore score --input examples/applicant.json --results-dir ml/results
```

Total wall time on Apple M-series MPS: **~60 minutes** (sample_n=60000, 4
trials, 10 epochs, batch_size=128). On CPU-only macOS the hybrid sweep is
~40× slower; the LightGBM baseline alone takes ~5 seconds.

### Running the synthetic-data ablation

```bash
novascore train --dataset synthetic --n-users 10000
# or the full Phase 4.5 sweep:
novascore sweep --n-trials 15 --n-users 10000
```

---

## Limitations (read this)

1. **60K sample, not full 307K**: training on the full Home Credit dataset
   would push AUROC further but at ~6x the wall time on commodity hardware.
   The 60K stratified sample preserves the class distribution (8.0% positive).
2. **No external features**: we use only the application + bureau + previous_app
   + installments tables. The POS_CASH and credit_card tables would likely
   help; they were skipped for compute budget.
3. **Fairness mitigation mitigates one attribute at a time**: we mitigated
   age_bucket because it had the largest ΔTPR (0.32). Intersectional fairness
   (gender × age × family_status) is left as future work.
4. **Equal-opportunity ≠ equal outcomes**: threshold optimization equalizes
   TPR across groups, which in this run boosted 18-25 scores by +46 points
   while penalizing 56+ by −42 points. This is the mathematically correct
   equal-opportunity intervention, but it is not an automatic improvement in
   user-facing outcomes — see REPORT.md for a longer discussion.
5. **macOS MPS-specific**: AdamW + AMP autocast use the MPS backend on Apple
   Silicon. On Linux/CUDA the same code works. On Linux/CPU expect a ~10×
   slowdown. `OMP_NUM_THREADS=1` is required on macOS to prevent the
   torch/LightGBM OpenMP-runtime conflict.

---

## Credits

- **Anupam Kumar**
- **Anish Kishore** — primary contributor: completed the broken ML pipeline,
  built the FastAPI backend, built the Next.js frontend, ran the dataset
  pivot and the headline benchmarking
- **Swaraj Thakur**

All three are at BIT Mesra.

Originally developed for the **Grab AI National Hackathon 2025** (national
semi-finalists). The repository was rebuilt from the hackathon Colab notebook
to production-grade ML code for honest public benchmarking.

## License

MIT — see [LICENSE](LICENSE).
