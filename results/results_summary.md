# Evaluation Results Summary

| Method | Accuracy (%) | 95% CI | Correct/Total | Avg Time (ms) |
|--------|--------------|--------|---------------|---------------|
| Greedy | 60.00 | (29.6, 90.4) | 6/10 | 3.05 |
| Top-P | 50.00 | (19.0, 81.0) | 5/10 | 12.58 |
| Dcbs | 60.00 | (29.6, 90.4) | 6/10 | 0.11 |
| Random | 50.00 | (19.0, 81.0) | 5/10 | 0.01 |

**Total Examples:** 10
**Model:** meta-llama/Llama-3.2-1B-Instruct

**Notes:**
- CI = Confidence Interval (binomial proportion)
- Baseline random performance is 50%
- * indicates p < 0.05, ** indicates p < 0.01