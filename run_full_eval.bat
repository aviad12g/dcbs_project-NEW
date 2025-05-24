@echo off
echo Running full DCBS evaluation on COMPLETE benchmark...
if not exist results mkdir results
python -m src.run_dcbs_eval --config configs/study_config.yaml --benchmark data/bench_wino.json --out_csv results/dcbs_full_benchmark_results.csv --log_level CRITICAL > full_benchmark_output.txt 2>&1
echo Evaluation complete. Check results at results/dcbs_full_benchmark_results.csv 