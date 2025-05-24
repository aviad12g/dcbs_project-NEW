# Evaluation Results Summary

| Method | Accuracy (%) | 95% CI | Correct/Total | Avg Time (ms) |
|--------|--------------|--------|---------------|---------------|
| Greedy | 80.00 | (44.9, 100.0) | 4/5 | 2.67 |
| Top-P | 80.00 | (44.9, 100.0) | 4/5 | 11.38 |
| Dcbs | 80.00 | (44.9, 100.0) | 4/5 | 0.18 |
| Random | 60.00 | (17.1, 100.0) | 3/5 | 0.02 |

**Total Examples:** 5
**Model:** meta-llama/Llama-3.2-1B

**Notes:**
- CI = Confidence Interval (binomial proportion)
- Baseline random performance is 50%
- * indicates p < 0.05, ** indicates p < 0.01