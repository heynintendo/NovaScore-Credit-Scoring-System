---
title: NovaScore API
emoji: 📊
sdk: docker
app_port: 7860
---

# NovaScore Inference API

FastAPI backend that serves a calibrated NovaScore (300–950) for credit
applications. Loads the committed LightGBM booster (`ml/results/lightgbm.txt`,
test AUROC 0.7450 on Home Credit Default Risk) and applies the empirical
calibration + per-group threshold-based fairness mitigation that the offline
pipeline produced.

The deep-learning hybrid (FT-Transformer + TCN) is preserved in the repo
under `ml/novascore/models/` as documented architectural exploration; it
underperformed LightGBM on this dataset (0.7253 vs 0.7450 test) and is not
served at inference for simplicity and latency.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/score` | Score a single applicant, return PD + score + decision band + fairness-adjusted score |
| GET | `/api/health` | Liveness probe |
| GET | `/` | HTML stub pointing to `/docs` |
| GET | `/docs` | Auto-generated OpenAPI / Swagger UI |

## Local run

```bash
cd api
pip install -r requirements.txt
# Models are loaded relative to the working directory.
cp -R ../ml/results models
uvicorn app.main:app --reload --port 7860
```

Visit http://localhost:7860/docs for the interactive Swagger UI.

## Docker

```bash
docker build -t novascore-api .
docker run -p 7860:7860 novascore-api
```

## Deploy to Hugging Face Spaces

The metadata block at the top of this file is what HF Spaces reads to
provision a Docker Space on `app_port: 7860`. Push this directory to the
Space's git remote (`git push hf main`) and the build kicks off automatically.

## Request shape

```json
POST /api/score
{
  "age_years": 42,
  "gender": "F",
  "family_status": "Married",
  "num_children": 1,
  "has_car": true,
  "annual_income": 175000,
  "loan_amount": 550000,
  "annuity": 27000,
  "years_employed": 8,
  "ext_source_1": 0.65,
  "ext_source_2": 0.55,
  "ext_source_3": 0.45
}
```

## Response shape

```json
{
  "pd": 0.0834,
  "novascore": 692.4,
  "decision_band": "Silver",
  "policy": "Manual review recommended. Smaller limit, repayment coaching offered.",
  "fairness_adjusted_score": 712.9,
  "fairness_adjusted_band": "Gold",
  "fairness_adjustment_reason": "Per-group equal-opportunity threshold for age 26-40 (+20.5).",
  "raw_features_used": 80
}
```

Unknown applicant features (bureau aggregates, previous applications, etc.)
fall back to the training mean — they standardize to 0 in feature space and
contribute a neutral signal to the model. The published demo accepts the
narrow 12-field form above; production deployment should require richer
historical data.
