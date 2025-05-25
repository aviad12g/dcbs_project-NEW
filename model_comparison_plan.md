# Model Choice Comparison Plan

## Option 1: Base Model (`meta-llama/Llama-3.2-1B`)
- **Our Results**: 41.1% accuracy  (reasonable for base)
- **Research Value**: Tests sampling on pure language modeling
- **Publications**: Compare with TinyLlama-1.1B (~25-30%), OLMo-1B (~32%)

## Option 2: Instruct Model (`meta-llama/Llama-3.2-1B-Instruct`) 
- **Expected Results**: ~65-70% accuracy
- **Research Value**: Standard benchmark comparison
- **Publications**: Compare with published ARC-Easy results

## Option 3: Both Models 
1. **Base model**: Shows fundamental sampling differences
2. **Instruct model**: Shows practical performance
3. **Analysis**: How does instruction tuning affect sampling method rankings?

## Commands to Run:

### Base Model (Fixed Timing):
```bash
python run_arc_eval_fixed.py \
  --model meta-llama/Llama-3.2-1B \
  --limit 100 \
  --output results/arc_base_fixed.json
```

### Instruct Model (Fixed Timing):
```bash
python run_arc_eval_fixed.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --limit 100 \
  --output results/arc_instruct_fixed.json
```

### Cache Comparison (DCBS efficiency test):
```bash
python run_arc_eval_fixed.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --limit 50 \
  --disable_cache \
  --output results/arc_no_cache.json
``` 