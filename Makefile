.PHONY: install train evaluate test lint format api-dev frontend-dev frontend-build clean

PYTHON ?= python3
VENV ?= .venv
PY := $(VENV)/bin/python

install:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e "ml/[dev]"

train:
	OMP_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
	$(PY) -m novascore.cli train --dataset home_credit --sample-n 60000 --n-trials 4 --epochs 10

evaluate:
	$(PY) -m novascore.cli evaluate --results-dir ml/results

recalibrate:
	$(PY) ml/scripts/recalibrate.py

test:
	OMP_NUM_THREADS=1 $(PY) -m pytest ml/tests/ -v

lint:
	$(PY) -m ruff check ml/novascore ml/tests
	$(PY) -m ruff format --check ml/novascore ml/tests

format:
	$(PY) -m ruff check --fix ml/novascore ml/tests
	$(PY) -m ruff format ml/novascore ml/tests

api-dev:
	cp -R ml/results api/models 2>/dev/null || true
	cd api && NOVASCORE_MODELS_DIR=$$(pwd)/models $$( cd .. && echo $$PWD )/$(VENV)/bin/python -m uvicorn app.main:app --reload --port 7860

api-docker:
	cd api && docker build -t novascore-api . && docker run -p 7860:7860 novascore-api

frontend-dev:
	cd frontend && npm install && NEXT_PUBLIC_API_URL=http://localhost:7860 npm run dev

frontend-build:
	cd frontend && npm install && npm run build

clean:
	rm -rf $(VENV) ml/novascore.egg-info ml/results/_archive_phase4_5_synth
	find . -type d -name __pycache__ -exec rm -rf {} +
