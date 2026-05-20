
PYTHON = python
STREAMLIT = streamlit

# ── Setup ──────────────────────────────────
install:
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip install -r requirements.txt

# ── Pipeline ───────────────────────────────
preprocess:
	$(PYTHON) -m src.preprocessing \
		--data-dir data/raw/ \
		--output-dir data/processed/

features:
	$(PYTHON) -m src.feature_engineering

train:
	$(PYTHON) -m src.models \
		--data-dir data/processed/ \
		--models-dir models/

evaluate:
	$(PYTHON) -m src.run_evaluation

shap:
	$(PYTHON) -m src.shap_analysis

predictions:
	$(PYTHON) -m src.save_predictions

# ── Run full pipeline in one command ───────
pipeline: preprocess features train evaluate predictions shap
	@echo "Pipeline complete. All models and data are ready."

# ── Tests ──────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-slow:
	$(PYTHON) -m pytest tests/ -v -m slow

test-integration:
	$(PYTHON) -m pytest tests/test_integration.py -v --tb=short -m slow

test-unit:
	$(PYTHON) -m pytest tests/ -v --tb=short --ignore=tests/test_integration.py

test-all:
	$(PYTHON) -m pytest tests/ -v --tb=short

coverage:
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=term-missing -v \
		--ignore=tests/test_integration.py

# ── Lint ───────────────────────────────────
lint:
	$(PYTHON) -m flake8 src/ tests/ dashboard/ \
		--max-line-length=88 \
		--exclude=__pycache__

# ── Dashboard ──────────────────────────────
serve:
	$(STREAMLIT) run dashboard/app.py \
		--server.port 8501 \
		--server.headless true \
		--server.enableCORS false \
		--server.enableXsrfProtection true \
		--logger.level warning

# ── Cleanup ────────────────────────────────
clean:
	rm -f ./models/*.pkl ./models/results_*.json ./models/results_*.csv
	rm -f ./data/processed/*.parquet
	rm -f pipeline.log

# ── Validation ─────────────────────────────
check: lint test-unit
	@echo "All checks passed."

# ── Reset pipeline artefacts only ──────────
reset:
	rm -f ./models/*.pkl ./models/results_*.js
	.on ./models/results_*.csv

# ── Full fresh start ───────────────────────
all: install pipeline test serve