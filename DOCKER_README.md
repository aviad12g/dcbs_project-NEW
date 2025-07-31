# DCBS Project - Docker Deployment

## Quick Start

### 1. Build and Run (Simple)
```bash
# Build the Docker image
docker build -t dcbs-project .

# Run with default settings (5 examples, greedy + dcbs)
docker run --rm -v $(pwd)/results:/app/results dcbs-project

# Run with custom parameters
docker run --rm -v $(pwd)/results:/app/results dcbs-project \
  python compare_methods.py --limit 10 --samplers greedy dcbs hier_loop
```

### 2. Using Docker Compose (Recommended)
```bash
# Run main evaluation
docker-compose up dcbs

# Run Streamlit web interface
docker-compose up streamlit
# Access at: http://localhost:8501

# Run Jupyter for analysis
docker-compose up jupyter  
# Access at: http://localhost:8888
```

## Common Commands

### Quick Evaluation (5 examples)
```bash
docker run --rm -v $(pwd)/results:/app/results dcbs-project \
  python compare_methods.py --limit 5 --samplers greedy dcbs
```

### Full Comparison (30 examples)
```bash
docker run --rm -v $(pwd)/results:/app/results dcbs-project \
  python compare_methods.py --limit 30 --samplers greedy dcbs hier_loop
```

### Debug Mode
```bash
docker run --rm -v $(pwd)/results:/app/results dcbs-project \
  python compare_methods.py --limit 3 --samplers dcbs --debug-mode
```

## Volume Mounts

The Docker setup automatically mounts:
- `./results` → `/app/results` (evaluation outputs)
- `./data` → `/app/data` (datasets)
- `./mlruns` → `/app/mlruns` (MLflow tracking)

## GPU Support

To enable GPU support, add `--gpus all`:
```bash
docker run --rm --gpus all -v $(pwd)/results:/app/results dcbs-project
```

## Troubleshooting

### Out of Memory
- Reduce `--limit` parameter
- Use `--load-in-4bit true` for model quantization

### Slow Performance
- Ensure GPU is available with `--gpus all`
- Reduce batch size with `--batch-size 1`

### Permission Issues
```bash
# Fix volume permissions
sudo chown -R $USER:$USER results/ mlruns/
```