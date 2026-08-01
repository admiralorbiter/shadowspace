.PHONY: install test lint format typecheck check clean generate-datasets

install:
	pip install -r requirements.txt

test:
	pytest

lint:
	ruff check src tests app.py

format:
	ruff format src tests app.py

typecheck:
	mypy src

generate-datasets:
	python -m shadowspace datasets fetch --output data/bundles

check: lint typecheck test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
