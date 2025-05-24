@echo off
echo === RUNNING DCBS EVALUATION ON FULL BENCHMARK ===
echo This may take a while...

REM Check if running just a subset for speed
if "%1"=="quick" (
    set BENCHMARK=data/bench_wino_1000.json
    set OUTPUT=results/dcbs_1000_results.csv
    echo Using subset of 1000 examples for quicker testing
) else (
    set BENCHMARK=data/bench_wino.json
    set OUTPUT=results/dcbs_full_benchmark_results.csv
    echo Using complete benchmark with all examples
)

REM Ensure results directory exists
if not exist results mkdir results

REM Run the evaluation
echo Running evaluation...
python -m src.run_dcbs_eval --config configs/study_config.yaml --benchmark %BENCHMARK% --out_csv %OUTPUT% --log_level CRITICAL > evaluation_log.txt 2>&1

REM Check if evaluation was successful
if not exist %OUTPUT% (
    echo ERROR: Evaluation failed! Check evaluation_log.txt for details.
    exit /b 1
)

REM Analyze and visualize the results
echo.
echo === ANALYZING RESULTS AND GENERATING VISUALIZATIONS ===
python visualize_results.py %OUTPUT%

echo.
echo === EVALUATION COMPLETE ===
echo Results saved to: %OUTPUT%
echo Check the PNG files in the results directory for visualizations. 