# Deterministic Category Based Sampling

A comprehensive evaluation harness for testing different LLM sampling strategies on multiple-choice tasks, featuring a clean implementation of Deterministic Category Based Sampling alongside standard methods with optimized KV caching and conversation flow.

## Overview

Deterministic Category Based Sampling is a novel sampling method that combines the benefits of diverse sampling with clustering of token embeddings. It enables exploration of different semantic categories during sampling while maintaining deterministic selection for reproducibility.

The algorithm works in multiple steps:
1. Select top-n tokens by probability (n=50)
2. Cluster token embeddings into semantic categories using DBSCAN (eps=0.3, min_samples=2)
3. Select the cluster with highest total probability mass (deterministic greedy selection)
4. Select the highest probability token from within the selected cluster (deterministic greedy selection)

This approach balances exploration of diverse semantic spaces with consistent, reproducible results through purely deterministic selection.

## Latest Evaluation Results

**Dataset**: ARC Easy (50 examples)  
**Model**: Llama 3.2 1B Instruct  
**Configuration**: Chain-of-thought enabled, DBSCAN clustering (eps=0.3), independent inference per sampler

### Accuracy Comparison

| Method | Accuracy | 95% CI | Correct/Total | Avg Time (ms) |
|--------|----------|--------|---------------|---------------|
| **DCBS** | **78.0%** | (64.8, 87.2) | 39/50 | 5424.9 |
| **Top-p** | 66.0% | (52.2, 77.6) | 33/50 | 4510.4 |
| **Greedy** | 64.0% | (50.1, 75.9) | 32/50 | 4289.8 |
| **Random** | 20.0% | (11.2, 33.0) | 10/50 | 8915.8 |

### Statistical Significance Analysis

**Paired T-Test Results (50 examples):**

| Comparison | t-statistic | p-value | Significance |
|------------|-------------|---------|--------------|
| DCBS vs Greedy | 2.447 | 0.0180 | * |
| DCBS vs Top-p | 1.630 | 0.1095 | ns |
| DCBS vs Random | 6.096 | < 0.001 | *** |
| Top-p vs Random | 4.608 | < 0.001 | *** |
| Greedy vs Random | 4.416 | < 0.001 | *** |

*Significance levels: *** p < 0.001, ** p < 0.01, * p < 0.05, ns = not significant*

### Key Findings

- **DCBS shows statistically significant improvement** over Greedy sampling (p = 0.018)
- **All sophisticated methods significantly outperform** random baseline (p < 0.001)
- **DCBS demonstrates substantial practical advantage** with 78.0% accuracy (+14% over Greedy, +12% over Top-p)
- **DBSCAN clustering effectiveness** confirmed with eps=0.3, min_samples=2 configuration
- **Performance optimization trade-off** identified: shared logits broke DCBS clustering functionality

### Research Insights

After resolving a critical performance optimization bug that caused identical results across samplers, DCBS now demonstrates its intended clustering advantage. The algorithm successfully identifies semantic token categories and selects from the most promising cluster, leading to measurably improved accuracy on ARC Easy reasoning tasks.

## System Architecture

The following diagram illustrates the DCBS evaluation framework and algorithm flow:

```mermaid
graph TD
    A["Input Text<br/>Question + Multiple Choice Options"] --> B["Language Model<br/>Llama 3.2 1B Instruct<br/>Chain-of-Thought Generation"]
    B --> C["Token Logits<br/>Probability Distribution P(t|context)"]
    
    C --> D["Greedy Sampling<br/>t* = argmax P(t|context)"]
    C --> E["Top-p Nucleus Sampling<br/>Sample from top tokens where<br/>Σ P(t) ≥ p (p=0.9)"]
    C --> F["DCBS Algorithm<br/>Deterministic Category-Based Selection"]
    C --> G["Random Baseline<br/>Uniform Random Selection"]
    
    F --> H["Step 1: Candidate Selection<br/>Select top-n tokens by probability<br/>(n=50)"]
    H --> I["Step 2: Embedding Extraction<br/>Extract token embeddings<br/>e(t) ∈ R^d from embedding layer"]
    I --> J["Step 3: Clustering<br/>DBSCAN Algorithm<br/>ε=0.3, min_samples=2"]
    J --> K["Step 4: Cluster Scoring<br/>Mass(C_i) = Σ P(t) for t ∈ C_i"]
    K --> L["Step 5: Cluster Selection<br/>C* = argmax Mass(C_i)"]
    L --> M["Step 6: Token Selection<br/>t* = argmax P(t) for t ∈ C*"]
    
    D --> N["Prediction Outputs"]
    E --> N
    M --> N
    G --> N
    
    N --> O["Performance Evaluation<br/>DCBS: 78.0% accuracy<br/>Top-p: 66.0% accuracy<br/>Greedy: 64.0% accuracy<br/>Random: 20.0% accuracy"]
    
    classDef input fill:#f9f9f9,stroke:#333,stroke-width:2px
    classDef model fill:#e8f4fd,stroke:#1976d2,stroke-width:2px
    classDef sampler fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef dcbs fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef output fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    
    class A input
    class B model
    class C model
    class D,E,G sampler
    class F,H,I,J,K,L,M dcbs
    class N,O output
```

### Evaluation Results Visualizations

![Accuracy Comparison](results/accuracy_by_method.png)
![Performance Comparison](results/timing_comparison.png)
![DCBS Optimization Impact](results/dcbs_optimization_impact.png)

## KV Caching Implementation

**Current Status**: Production-ready KV caching with optimized conversation flow

### Implementation Features

1. **Two-Step Conversation Flow**:
   - Step 1: LLM generates reasoning response
   - Step 2: LLM generates final answer with cached context
   - Proper `add_generation_prompt=True` usage
   - Never completes user messages, only assistant messages

2. **Performance Optimization**:
   - KV cache reuse between reasoning and final answer steps
   - Token-by-token generation with cache persistence
   - GPU memory efficient implementation
   - Deterministic results with fixed random seeds

3. **Configuration Options**:
   ```bash
   # Enable caching (default behavior)
   python compare_methods.py --model meta-llama/Llama-3.2-1B-Instruct
   
   # Programmatic configuration
   from dcbs import DCBSSampler
   sampler = DCBSSampler.create_default(k=8, top_n=50, enable_caching=True)
   ```

### Performance Metrics

- **Processing time**: ~4ms average per example (500 examples)
- **Memory efficiency**: Optimized for 11GB GPU memory
- **Throughput**: ~150 examples/minute on RTX 4060 Laptop GPU
- **Cache effectiveness**: Minimal overhead for conversation flow

## Project Structure

```
dcbs_project-NEW/
├── dcbs/                    # Core sampling package
│   ├── samplers/           # All sampler implementations
│   ├── clustering.py       # Clustering abstractions and implementations
│   ├── cache_manager.py    # Thread-safe caching system
│   └── optimizations.py    # Advanced optimization features
├── src/                    # Source code
│   ├── evaluation_core/    # Core evaluation logic and framework
│   ├── visualization.py    # Chart generation with statistical analysis
│   ├── token_utils.py     # Token handling utilities
│   └── errors.py          # Error handling and logging
├── tests/                  # Comprehensive test suite
├── data/                   # Benchmark datasets
├── results/               # Generated results and charts
├── docs/                  # Documentation
├── compare_methods.py     # MAIN EVALUATION SCRIPT (unified framework)
├── fisher_test.py         # Statistical significance analysis
└── requirements.txt       # Python dependencies
```

## Sampling Methods

The project implements four sampling strategies with a unified interface:

### 1. **Greedy Sampling**
- **Algorithm**: Always selects the highest probability token (argmax)
- **Characteristics**: Fully deterministic, fastest execution
- **Performance**: 68.2% accuracy, 0.36ms average time

### 2. **Top-p (Nucleus) Sampling**
- **Algorithm**: Samples from the smallest set of tokens whose cumulative probability ≥ p
- **Characteristics**: Stochastic sampling with controlled diversity
- **Configuration**: `p=0.9` (default)
- **Performance**: 67.2% accuracy, 0.93ms average time

### 3. **Deterministic Category Based Sampling**
- **Algorithm**: Clusters tokens by embeddings, selects best cluster, then best token
- **Characteristics**: Deterministic, semantically-aware, novel approach
- **Configuration**: DBSCAN clustering (eps=0.3, min_samples=2), `top_n=50` candidates
- **Performance**: 78.0% accuracy, 5424.9ms average time

### 4. **Random Sampling**
- **Algorithm**: Uniform random selection from allowed tokens
- **Characteristics**: Maximum stochasticity, serves as baseline
- **Performance**: 25.2% accuracy, 0.01ms average time

## Installation

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd dcbs_project-NEW

# Set up virtual environment and install
make venv

# Authenticate with HuggingFace
huggingface-cli login

# Run 20-question smoke test
make sanity
```

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Usage

### Quick Start

```bash
# Run comparative evaluation on ARC Easy (default)
python compare_methods.py

# Specify custom model and dataset
python compare_methods.py \
    --model "meta-llama/Llama-3.2-1B-Instruct" \
    --benchmark "data/arc_easy_full.json" \
    --limit 500

# Run with 4-bit quantization for faster inference
python compare_methods.py \
    --model "meta-llama/Llama-3.2-1B-Instruct" \
    --load-in-4bit \
    --limit 100

# Run specific samplers only
python compare_methods.py \
    --samplers dcbs greedy \
    --limit 50
```

### Advanced Features

```bash
# Parameter sweeping for DCBS optimization
python compare_methods.py \
    --sweep-k 4 8 16 \
    --sweep-top-n 20 50 100 \
    --limit 100

# Memory profiling and detailed output
python compare_methods.py \
    --memory-profiling \
    --save-details \
    --output-format both \
    --limit 500

# Statistical significance analysis
python fisher_test.py
```

### Command Line Options

```bash
usage: compare_methods.py [-h] [--model MODEL] [--benchmark BENCHMARK]
                         [--output-dir OUTPUT_DIR] [--limit LIMIT]
                         [--top-p TOP_P] [--k K] [--top-n TOP_N]
                         [--no-cot] [--log-level {DEBUG,INFO,WARNING,ERROR}]
                         [--save-details] [--load-in-4bit]
                         [--samplers {greedy,top-p,dcbs,random} ...]

Options:
  --model MODEL         HuggingFace model name or path (default: meta-llama/Llama-3.2-1B-Instruct)
  --benchmark BENCHMARK Path to benchmark JSON file (default: data/arc_easy_full.json)
  --output-dir OUTPUT_DIR Output directory for results
  --limit LIMIT         Limit number of examples for testing
  --top-p TOP_P         Top-p value for nucleus sampling (default: 0.9)
  --k K                 Number of clusters for DCBS (default: 8)
  --top-n TOP_N         Top-n tokens for DCBS clustering (default: 50)
  --no-cot             Disable chain-of-thought reasoning
  --save-details       Save detailed per-example results
  --load-in-4bit       Load model with 4-bit quantization
  --samplers           Specify which samplers to evaluate (default: all)
```

### Using the Sampler Classes

```python
import torch
from dcbs import (
    GreedySampler, TopPSampler, RandomSampler, DCBSSampler,
    KMeansClusterer, TopNCandidateSelector, SamplingContext
)

# Setup
logits = torch.tensor([1.0, 3.0, 2.0, 0.5, 2.5])
filter_tokens = {1, 2, 4}

# Greedy sampling
greedy = GreedySampler()
token = greedy.sample(logits, filter_tokens=filter_tokens)

# Top-p sampling
top_p = TopPSampler(p=0.9)
token = top_p.sample(logits, filter_tokens=filter_tokens)

# Deterministic Category Based Sampling
clusterer = KMeansClusterer(k=8)
candidate_selector = TopNCandidateSelector(top_n=50)
dcbs = DCBSSampler(clusterer, candidate_selector)
context = SamplingContext(embedding_layer=model.get_input_embeddings())
token = dcbs.sample(logits, filter_tokens=filter_tokens, context=context)

# Or use the convenience factory method
from dcbs import DCBSSamplerFactory
dcbs_default = DCBSSamplerFactory.create_default(k=8, top_n=50)
token = dcbs_default.sample(logits, filter_tokens=filter_tokens, context=context)
```

## Generated Outputs

The evaluation produces:

1. **Main accuracy chart** (`results/accuracy_by_method.png`)
2. **Timing comparison** (`results/timing_comparison.png`) 
3. **DCBS optimization impact** (`results/dcbs_optimization_impact.png`)
4. **Results summary** (`results/results_summary.md`)
5. **Raw data** (`results/evaluation_results_YYYYMMDD_HHMMSS.json`)
6. **Statistical analysis** (`results/fisher_exact_tests.txt`)

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_samplers.py -v
pytest tests/test_integration.py -v

# Run with coverage
pytest tests/ --cov=dcbs --cov=src
```

### Code Quality

```bash
# Format code
black dcbs/ src/ tests/

# Lint code
flake8 dcbs/ src/ tests/

# Type checking
mypy dcbs/ src/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-sampler`)
3. Make your changes with tests
4. Run the test suite (`pytest tests/`)
5. Submit a pull request

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Use `--limit` to reduce evaluation size
2. **Model loading errors**: Check HuggingFace token and model access
3. **Template issues**: Verify model supports chat formatting
4. **Slow performance**: Consider smaller models or reduced `top_n`

### Debug Mode

```bash
python compare_methods.py --log-level DEBUG --limit 10
```

## Acknowledgments

- HuggingFace Transformers team for excellent model libraries
- scikit-learn for robust clustering implementations
- The research community for foundational work on sampling methods
