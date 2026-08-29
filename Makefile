PYTHON := uv run python
PYTEST := uv run pytest
RUFF := uv run ruff
MYPY := uv run mypy
MLFLOW := uv run mlflow
UVICORN := uv run uvicorn

CONFIG ?= src/vaak/config/base.yaml
MLFLOW_DB ?= sqlite:///mlruns/mlflow.db
MLFLOW_PORT ?= 5000
API_PORT ?= 8000

.PHONY: help
help:
	@echo "Vaak development commands:"
	@echo ""
	@echo "  make install         Install dependencies"
	@echo "  make test            Run tests"
	@echo "  make test-cov        Run tests with coverage"
	@echo "  make lint            Run Ruff linting"
	@echo "  make lint-fix        Fix Ruff lint issues"
	@echo "  make format          Format code"
	@echo "  make format-check    Check formatting"
	@echo "  make typecheck       Run mypy"
	@echo "  make check           Run all quality checks"
	@echo "  make clean           Remove generated files"
	@echo ""
	@echo "  make prepare-asvspoof Build ASVspoof 2019 dataset manifest"
	@echo "  make train           Run model training (e.g. make train CONFIG=configs/base.yaml)"
	@echo "  make mlflow-ui       Start MLflow tracking server"
	@echo "  make serve           Start FastAPI inference server"

.PHONY: version
version:
	@uv version

.PHONY: bump-patch
bump-patch:
	uv version --bump patch

.PHONY: bump-minor
bump-minor:
	uv version --bump minor

.PHONY: bump-major
bump-major:
	uv version --bump major

.PHONY: install
install:
	uv sync

.PHONY: test
test:
	$(PYTEST)

.PHONY: test-cov
test-cov:
	$(PYTEST) --cov=vaak --cov-report=term-missing

.PHONY: format
format:
	$(RUFF) check . --fix
	$(RUFF) format .

.PHONY: format-check
format-check:
	$(RUFF) check .
	$(RUFF) format --check .

.PHONY: typecheck
typecheck:
	$(MYPY) src tests

.PHONY: check
check:
	$(RUFF) check .
	$(RUFF) format --check .
	$(MYPY) src tests
	$(PYTEST)

.PHONY: clean
clean:
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +

.PHONY: prepare-asvspoof
prepare-asvspoof:
	$(PYTHON) scripts/prepare_asvspoof.py

.PHONY: train
train:
	$(PYTHON) scripts/train.py --config $(CONFIG)

.PHONY: mlflow-ui
mlflow-ui:
	$(MLFLOW) ui --backend-store-uri $(MLFLOW_DB) --port $(MLFLOW_PORT)

.PHONY: serve
serve:
	$(UVICORN) vaak.api.server:app --reload --port $(API_PORT)