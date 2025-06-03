# Evaluation Results Summary

| Method | Accuracy (%) | 95% CI | Correct/Total | Avg Time (ms) |
|--------|--------------|--------|---------------|---------------|
| Greedy | 64.00 | (50.1, 75.9) | 32/50 | 4289.85 |
| Top_P | 66.00 | (52.2, 77.6) | 33/50 | 4510.35 |
| Dcbs | 78.00 | (64.8, 87.2) | 39/50 | 5424.93 |
| Random | 20.00 | (11.2, 33.0) | 10/50 | 8915.82 |

**Total Examples:** 50
**Model:** meta-llama/Llama-3.2-1B-Instruct

**Notes:**
- CI = Confidence Interval (binomial proportion)
- Baseline random performance is 50%
- * indicates p < 0.05, ** indicates p < 0.01