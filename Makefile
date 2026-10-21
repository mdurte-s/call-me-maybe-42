.PHONY: install run debug clean lint lint-strict

PYTHON := uv run python
SRC_DIR := src

install:
	uv sync

run:
	$(PYTHON) -m $(SRC_DIR)

debug:
	$(PYTHON) -m pdb -c continue -m $(SRC_DIR)

clean:
	rm -rf __pycache__ $(SRC_DIR)/__pycache__ .mypy_cache .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

lint:
	uv run flake8 $(SRC_DIR)
	uv run mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs $(SRC_DIR)

lint-strict:
	uv run flake8 $(SRC_DIR)
	uv run mypy --strict $(SRC_DIR)
