# Messages Completion Module

A self-contained module for converting messages to completions with deterministic, batch-invariant processing.

## Quick Start (GPU, Greedy Mode)

```python
from messages_completion import MessageCompleter, CompletionConfig

# Initialize model (requires GPU)
comp = MessageCompleter(CompletionConfig(model_name="meta-llama/Meta-Llama-3-8B-Instruct", max_new_tokens=32))

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

### MessageCompleter

GPU-optimized model interface with determinism:

```python
comp = MessageCompleter(CompletionConfig(model_name="meta-llama/Meta-Llama-3-8B-Instruct"))
```

## Key Features

- **Deterministic**: Same input always produces same output
- **Batch Invariant**: Sequential and batch processing produce identical results  
- **GPU Optimized**: Efficient CUDA operations with determinism guarantees
- **Standard Decoding via model.generate**: Greedy (`do_sample=False`) and Top‑p (`do_sample=True`, `top_p`, `temperature`) use the model’s generate API directly.
- **No Template Injection**: This module does not add messages or prompts; callers must supply the exact messages to complete (e.g., system/instructions).
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
