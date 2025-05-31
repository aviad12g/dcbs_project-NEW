# Evaluation Results Summary

| Method | Accuracy (%) | 95% CI | Correct/Total | Avg Time (ms) |
|--------|--------------|--------|---------------|---------------|
| Greedy | 66.00 | (56.3, 74.5) | 66/100 | 0.18 |
| Top_P | 65.00 | (55.3, 73.6) | 65/100 | 1.43 |
| Dcbs | 66.00 | (56.3, 74.5) | 66/100 | 1.55 |
| Random | 22.00 | (15.0, 31.1) | 22/100 | 0.00 |

**Total Examples:** 100
**Model:** meta-llama/Llama-3.2-1B-Instruct

**Notes:**
- CI = Confidence Interval (binomial proportion)
- Baseline random performance is 50%
- * indicates p < 0.05, ** indicates p < 0.01