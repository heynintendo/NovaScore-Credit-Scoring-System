"""NovaScore — equitable credit scoring engine for gig-economy partners.

3-L narrative:
- Learn: ingest mobility, delivery, financial, quality, and network signals.
- Lend: hybrid FT-Transformer + TCN + Node2Vec fusion to a calibrated score.
- Loop: audit and mitigate group-level disparities in the score's decisions.
"""

__version__ = "0.2.0"

# Calibration anchors and score range — single source of truth for the package.
SCORE_MIN: float = 300.0
SCORE_MAX: float = 950.0
PD_ANCHORS: tuple[float, float] = (0.01, 0.20)
SCORE_ANCHORS: tuple[float, float] = (900.0, 650.0)

# 90-day window, 13 weekly buckets.
WINDOW_DAYS: int = 90
T_WEEKS: int = 13
SEQ_FEATURES: tuple[str, ...] = (
    "trips",
    "dist",
    "dur",
    "cancels",
    "rating",
    "earnings",
    "spend",
    "txns",
    "merchants",
)

GRAPH_DIM: int = 64
