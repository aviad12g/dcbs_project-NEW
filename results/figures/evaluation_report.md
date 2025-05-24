# DCBS Evaluation Results

Total examples evaluated: 3801

## Accuracy Comparison

| Method | Accuracy | Avg Time (ms) |
|--------|----------|---------------|
| greedy | 50.04% | 0.97 |
| top-p | 49.78% | 1.56 |
| dcbs | 50.36% | 5.51 |
| random | 48.43% | 0.01 |

## Key Observations

1. **DCBS vs Other Methods**: DCBS outperforms other methods in accuracy.

2. **Performance Trade-off**: DCBS takes approximately 5.7x longer than greedy sampling.

3. **Overall Results**: All methods perform similarly in terms of accuracy, suggesting the model's knowledge is the limiting factor rather than the sampling method.

## Generated Visualizations

1. `method_accuracy_comparison.png`: Bar chart comparing accuracy across methods
2. `method_time_comparison.png`: Bar chart comparing execution time across methods
3. `accuracy_vs_time.png`: Scatter plot showing accuracy vs execution time trade-off
4. `parameter_accuracy_impact.png`: Line chart showing how parameters affect accuracy
5. `parameter_time_impact.png`: Line chart showing how parameters affect execution time