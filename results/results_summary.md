# Evaluation Results Summary

| Method | Accuracy (%) | 95% CI | Correct/Total | Avg Time (ms) |
|--------|--------------|--------|---------------|---------------|
| Greedy | 51.00 | (41.3, 60.6) | 51/100 | 0.47 |
| Top_P | 49.00 | (39.4, 58.7) | 49/100 | 0.91 |
| Dcbs | 53.00 | (43.3, 62.5) | 53/100 | 1.72 |
| Random | 22.00 | (15.0, 31.1) | 22/100 | 0.00 |

**Total Examples:** 100
**Model:** meta-llama/Llama-3.2-1B-Instruct

**Notes:**
- CI = Confidence Interval (binomial proportion)
- Baseline random performance is 50%
- * indicates p < 0.05, ** indicates p < 0.01