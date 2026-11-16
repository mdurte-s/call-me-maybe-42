.PHONY: install run debug clean lint lint-strict

# Store Hugging Face cache in /tmp to prevent filling up home directory quota
export HF_HOME ?= /tmp/$(USER)/huggingface

PYTHON := uv run python
SRC_DIR := src
INPUT := data/input/function_calling_tests.json
FUNCTIONS := data/input/functions_definition.json
OUTPUT := data/output/function_calls.json

install:
	uv sync

run:
	@mkdir -p $(HF_HOME)
	$(PYTHON) -m $(SRC_DIR) --functions_definition $(FUNCTIONS) --input $(INPUT) --output $(OUTPUT)

visualize:
	@mkdir -p $(HF_HOME)
	$(PYTHON) -m $(SRC_DIR) --functions_definition $(FUNCTIONS) --input $(INPUT) --output $(OUTPUT) --visualize

debug:
	@mkdir -p $(HF_HOME)
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