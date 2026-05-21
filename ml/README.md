# novascore (ml/)

Python package for the NovaScore credit scoring pipeline: data synthesis,
feature engineering, LightGBM baseline, hybrid FT-Transformer + TCN + Node2Vec
model, calibration to a 300–950 score, and fairness analysis.

## Install (editable)

```bash
pip install -e ml/
```

## CLI

```bash
novascore train          # generates synth data if missing, trains end-to-end
novascore evaluate       # rebuild plots and metrics from last checkpoint
novascore score --input examples/partner.json
```

See top-level README.md for the full project context.
