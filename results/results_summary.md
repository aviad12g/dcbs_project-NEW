# Evaluation Results Summary

| Method | Accuracy (%) | 95% CI | Correct/Total | Avg Time (ms) |
|--------|--------------|--------|---------------|---------------|
| Greedy | 68.20 | (64.0, 72.1) | 341/500 | 0.36 |
| Top_P | 67.20 | (63.0, 71.2) | 336/500 | 0.93 |
| Dcbs | 68.60 | (64.4, 72.5) | 343/500 | 3.72 |
| Random | 25.20 | (21.6, 29.2) | 126/500 | 0.01 |

**Total Examples:** 500
**Model:** meta-llama/Llama-3.2-1B-Instruct

**Notes:**
- CI = Confidence Interval (binomial proportion)
- Baseline random performance is 50%
- * indicates p < 0.05, ** indicates p < 0.01