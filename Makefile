.PHONY: venv sanity test clean install format lint check help

# Default target
help:
	@echo "Available targets:"
	@echo "  venv     - Create virtual environment and install package in editable mode"
	@echo "  sanity   - Run 20-question smoke test on ARC Easy"
	@echo "  test     - Run full test suite"
	@echo "  format   - Format code with Black and isort"
	@echo "  lint     - Run linting with Flake8"
	@echo "  check    - Run both formatting and linting checks"
	@echo "  clean    - Clean up build artifacts and cache"
	@echo "  install  - Install package in editable mode"

# Create virtual environment and install package
venv:
	python -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip
	. .venv/bin/activate && pip install -e .
	@echo ""
	@echo "Virtual environment created! Activate with:"
	@echo "source .venv/bin/activate"

# Install package in editable mode (for existing environment)
install:
	pip install -e .

# Format code with Black and isort
format:
	@echo "Running Black formatter..."
	black .
	@echo "Running isort import sorter..."
	isort .
	@echo "Code formatting complete!"

# Run linting with Flake8
lint:
	@echo "Running Flake8 linter..."
	flake8 .
	@echo "Linting complete!"

# Run both formatting and linting checks
check: format lint
	@echo "Code quality checks complete!"

# Run 20-question smoke test
sanity:
	python compare_methods.py \
		--model meta-llama/Llama-3.2-1B \
		--benchmark data/arc_easy_full.json \
		--limit 20 \
		--samplers dcbs greedy \
		--load-in-4bit \
		--no-cot

# Run full test suite
test:
	pytest tests/ -v

# Clean up build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete 