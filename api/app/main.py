"""NovaScore inference API — FastAPI app entry point.

Endpoints:
    GET  /                 — HTML landing pointing at /docs.
    GET  /api/health       — liveness probe with feature_count.
    POST /api/score        — score one applicant.

CORS is permissive (`*`) by default so the Vercel-hosted frontend can call us
during development; tighten the `NOVASCORE_CORS_ORIGINS` env var (comma-separated
list) for production.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from .inference import load_artifacts, score_applicant
from .schemas import HealthResponse, ScoreRequest, ScoreResponse

_HERE = Path(__file__).parent
_MODELS_DIR = Path(os.getenv("NOVASCORE_MODELS_DIR", _HERE.parent / "models"))

_origins = os.getenv("NOVASCORE_CORS_ORIGINS", "*").split(",")

app = FastAPI(
    title="NovaScore API",
    description=(
        "Equitable credit scoring service backed by a LightGBM model trained on the "
        "Home Credit Default Risk public dataset (test AUROC 0.7450). Applies the "
        "offline empirical calibration and per-group threshold-based fairness "
        "mitigation produced by the training pipeline."
    ),
    version="0.2.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    if not _MODELS_DIR.exists():
        raise RuntimeError(
            f"models directory {_MODELS_DIR} not found. "
            "Copy ml/results/ to api/models/ before starting the API."
        )
    load_artifacts(_MODELS_DIR)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root() -> str:
    return """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>NovaScore API</title>
    <style>
      body { font-family: -apple-system, system-ui, sans-serif; background:#0A1628; color:#E8DCC4;
             margin:0; min-height:100vh; display:flex; align-items:center; justify-content:center; }
      .card { padding:2.5rem 3rem; max-width:520px; }
      h1 { font-weight:600; letter-spacing:-0.02em; font-size:2.25rem; margin:0 0 .75rem; }
      p  { font-size:1rem; line-height:1.55; color:rgba(232,220,196,.7); margin:0 0 1.25rem; }
      a  { color:#C9A26F; text-decoration:none; border-bottom:1px solid rgba(201,162,111,.45); }
      a:hover { border-bottom-color:#C9A26F; }
    </style>
  </head>
  <body>
    <main class="card">
      <h1>NovaScore API</h1>
      <p>Calibrated credit scoring service.
         OpenAPI / Swagger UI at <a href="/docs">/docs</a>.
         Health probe at <a href="/api/health">/api/health</a>.</p>
    </main>
  </body>
</html>"""


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    b = load_artifacts(_MODELS_DIR)
    return HealthResponse(status="ok", model="lightgbm", feature_count=len(b.feature_columns))


@app.post("/api/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    try:
        b = load_artifacts(_MODELS_DIR)
        return score_applicant(req, b)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model artifact missing: {e}") from e
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(e)) from e
