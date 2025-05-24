# Winogrande 1.1 Dataset Integration

This directory is for storing Winogrande 1.1 dataset files for use with the DCBS evaluation framework.

## Dataset Information

Winogrande 1.1 is a large-scale adversarial dataset for the Winograd Schema Challenge. It's designed to be challenging for modern language models and focuses on commonsense reasoning.

## Getting the Dataset

You can download the Winogrande 1.1 dataset from:
- https://winogrande.allenai.org/
- Or directly from Hugging Face: https://huggingface.co/datasets/winogrande

Download the dev.jsonl, train.jsonl, or test.jsonl files and place them in this directory.

## Converting to DCBS Format

The DCBS evaluation framework expects a different JSON format than the JSONL format used by Winogrande. Use the provided conversion script:

```bash
python -m src.convert_winogrande data/winogrande/dev.jsonl data/winogrande/dev.json
```

## Using with DCBS Evaluation

After conversion, you can use the converted file with the DCBS evaluation script:

```bash
python -m src.run_dcbs_eval --config configs/study_config.yaml --benchmark data/winogrande/dev.json --out_csv results/dev_evaluation.csv
```

## Benefits of Using Winogrande 1.1

1. **Standard Benchmark**: Winogrande is a widely-used benchmark for evaluating natural language understanding.
2. **Proper Splits**: Using the official dev set for parameter tuning and test set for final evaluation ensures reliable results.
3. **Challenging Examples**: The dataset contains challenging examples that test sophisticated reasoning abilities.

## Format Conversion Details

The conversion script transforms the JSONL format (one JSON object per line) to a single JSON array with the following structure:

```json
[
  {
    "id": "winogrande_123",
    "sentence": "The trophy doesn't fit into the brown suitcase because it's too large.",
    "option1": "trophy",
    "option2": "suitcase",
    "correct_option": "1"
  },
  ...
]
``` 