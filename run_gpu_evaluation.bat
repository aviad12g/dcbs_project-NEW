@echo off
REM GPU evaluation script for DCBS
REM This script runs the unified DCBS evaluation framework on GPU

echo Starting DCBS GPU evaluation...
echo Using unified evaluation framework
echo.

REM Check if Python is available
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found in PATH, trying conda environment...
    call conda activate dcbs_env 2>nul
    if %errorlevel% neq 0 (
        echo Could not activate conda environment. Please ensure Python is available.
        pause
        exit /b 1
    )
)

REM Run evaluation using the unified framework
%USERPROFILE%\anaconda3\python.exe compare_methods.py ^
    --model "microsoft/DialoGPT-medium" ^
    --benchmark "data/arc_easy_sample.json" ^
    --output-dir "results/gpu_evaluation" ^
    --limit 100 ^
    --log-level INFO ^
    --output-format both ^
    --save-details

echo.
if %errorlevel% equ 0 (
    echo Evaluation completed successfully!
    echo Results saved to results/gpu_evaluation/
) else (
    echo Evaluation failed with error code %errorlevel%
)

echo.
pause 