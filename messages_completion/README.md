# Messages Completion Module

A self-contained module for converting messages to completions with deterministic, batch-invariant processing.

## Quick Start (GPU, Greedy Mode)

```python
from messages_completion import MessageCompleter, HuggingFaceModelInterface

# Initialize model (requires GPU)
model = HuggingFaceModelInterface("meta-llama/Meta-Llama-3-8B-Instruct")
model.set_seed(42)  # For deterministic results

# Create completer
comp = MessageCompleter(model, max_new_tokens=32)

# Single conversation
convos = [[{"role":"system","content":"You are a helpful assistant."},
           {"role":"user","content":"Say hello in one word."}]]
result = comp.complete(convos, use_batching=False, return_logprobs=False)
print(result.text)  # Single conversation returns CompletionResult

# Batch processing (deterministic, batch-invariant)
convos = [
    [{"role":"system","content":"You are terse."},
     {"role":"user","content":"2+2?"}],
    [{"role":"system","content":"You are terse."},
     {"role":"user","content":"Capital of France?"}]
]

# Sequential processing
seq_results = []
for convo in convos:
    out = comp.complete([convo], use_batching=False, return_logprobs=False)
    seq_results.append(out.token_ids)

# Batch processing  
batch_result = comp.complete(convos, use_batching=True, return_logprobs=False)
batch_results = [c.token_ids for c in batch_result.completions]

# Results are identical (batch invariance)
assert seq_results == batch_results
print("Batch invariance verified!")
```

## Core API

### MessageCompleter

The main interface for deterministic completion:

```python
comp = MessageCompleter(model, max_new_tokens=32)

# Single conversation -> CompletionResult
result = comp.complete([conversation], use_batching=False)
print(result.text)
print(result.token_ids)

# Multiple conversations -> BatchCompletionResult  
result = comp.complete(conversations, use_batching=True)
for completion in result.completions:
    print(completion.text)
```

### HuggingFaceModelInterface

GPU-optimized model interface with determinism:

```python
model = HuggingFaceModelInterface("meta-llama/Meta-Llama-3-8B-Instruct")
model.set_seed(42)  # Deterministic generation

# GPU determinism settings are automatically applied:
# - attn_implementation="eager" (no flash attention)
# - torch.backends.cudnn.deterministic = True
# - torch.use_deterministic_algorithms(True)
# - Disabled TF32 for exact reproducibility
```

## Key Features

- **Deterministic**: Same input always produces same output
- **Batch Invariant**: Sequential and batch processing produce identical results  
- **GPU Optimized**: Efficient CUDA operations with determinism guarantees
- **Greedy Sampling**: Uses `do_sample=False` for deterministic generation
- **Self-contained**: No external dependencies on parent project

## Testing

Run the batch invariance test:

```bash
python -m pytest messages_completion/tests/test_batch_invariance.py::test_batch_invariance -v
```

This test verifies that sequential processing (N calls with batch=1) produces identical results to batch processing (1 call with batch=N).

## Requirements

- PyTorch with CUDA support
- Transformers library
- GPU with sufficient memory for your chosen model