PYTHON := uv run python
PYTEST := uv run pytest
RUFF := uv run ruff
MYPY := uv run mypy

.PHONY: help
help:
	@echo "Vaak development commands:"
	@echo ""
	@echo "  make install      Install dependencies"
	@echo "  make test         Run tests"
	@echo "  make test-cov     Run tests with coverage"
	@echo "  make lint         Run Ruff linting"
	@echo "  make lint-fix     Fix Ruff lint issues"
	@echo "  make format       Format code"
	@echo "  make format-check Check formatting"
	@echo "  make typecheck    Run mypy"
	@echo "  make check        Run all quality checks"
	@echo "  make smoke        Run smoke test"
	@echo "  make clean        Remove generated files"


.PHONY: install
install:
	uv sync


.PHONY: test
test:
	$(PYTEST)


.PHONY: test-cov
test-cov:
	$(PYTEST) --cov=vaak --cov-report=term-missing


.PHONY: lint
lint:
	$(RUFF) check .


.PHONY: lint-fix
lint-fix:
	$(RUFF) check . --fix


.PHONY: format
format:
	$(RUFF) format .


.PHONY: format-check
format-check:
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


.PHONY: smoke
smoke:
	$(PYTHON) scripts/smoke_test.py


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