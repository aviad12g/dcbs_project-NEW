@echo off
echo Running DCBS evaluation...
if not exist results mkdir results
python -m src.run_dcbs_eval --config configs/study_config.yaml --benchmark data/bench_wino_test.json --out_csv results/dcbs_eval_results.csv --log_level CRITICAL --limit 5 > eval_output.txt 2>&1
echo Evaluation complete. Check results at results/dcbs_eval_results.csv and logs in eval_output.txt 