.PHONY: venv sanity test clean install format lint check help

help:
	@echo "Available targets:"
	@echo "  venv       - Create virtual environment and install package in editable mode"
	@echo "  sanity     - Run 20-question smoke test on ARC Easy"
	@echo "  cache-test - Compare DCBS performance with and without caching"
	@echo "  sweep      - Run parameter sweep for DCBS optimization"
	@echo "  full       - Run complete evaluation with all methods"
	@echo "  test       - Run full test suite"
	@echo "  format     - Format code with Black and isort"
	@echo "  lint       - Run linting with Flake8"
	@echo "  check      - Run both formatting and linting checks"
	@echo "  clean      - Clean up build artifacts and cache"
	@echo "  install    - Install package in editable mode"

venv:
	python -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip
	. .venv/bin/activate && pip install -e .
	@echo ""
	@echo "Virtual environment created! Activate with:"
	@echo "source .venv/bin/activate"

install:
	pip install -e .

format:
	@echo "Running Black formatter..."
	black .
	@echo "Running isort import sorter..."
	isort .
	@echo "Code formatting complete!"

lint:
	@echo "Running Flake8 linter..."
	flake8 .
	@echo "Linting complete!"

check: format lint
	@echo "Code quality checks complete!"

sanity:
	python compare_methods.py \
		--model meta-llama/Llama-3.2-1B \
		--benchmark data/arc_easy_full.json \
		--limit 20 \
		--samplers dcbs greedy \
		--load-in-4bit \
		--no-cot

cache-test:
	@echo "Testing DCBS with caching enabled..."
	python compare_methods.py \
		--model meta-llama/Llama-3.2-1B \
		--limit 50 \
		--samplers dcbs greedy \
		--load-in-4bit
	@echo "Testing DCBS with caching disabled..."
	python compare_methods.py \
		--model meta-llama/Llama-3.2-1B \
		--limit 50 \
		--samplers dcbs greedy \
		--load-in-4bit \
		--disable-cache

sweep:
	python compare_methods.py \
		--model meta-llama/Llama-3.2-1B \
		--sweep-k 4 8 16 \
		--sweep-top-n 20 50 100 \
		--limit 100 \
		--load-in-4bit

full:
	python compare_methods.py \
		--model meta-llama/Llama-3.2-1B \
		--benchmark data/arc_easy_full.json \
		--save-details \
		--output-format both

test:
	pytest tests/ -v

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete 