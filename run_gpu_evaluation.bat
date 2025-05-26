@echo off
echo Starting GPU-accelerated DCBS evaluation...
echo.

echo Python path: %USERPROFILE%\anaconda3\python.exe
echo Current directory: %CD%
echo.

echo Running evaluation with:
echo - Model: unsloth/Llama-3.2-1B-Instruct
echo - Limit: 5 examples (scientific test)
echo - DCBS k=3, top_n=50
echo - Top-p: 0.9
echo.

%USERPROFILE%\anaconda3\python.exe src/chat_eval.py ^
  --model "unsloth/Llama-3.2-1B-Instruct" ^
  --benchmark "data/arc_easy_processed.json" ^
  --output "results/gpu_scientific_dcbs.json" ^
  --limit 5 ^
  --top_p 0.9 ^
  --k 3 ^
  --top_n 50

echo.
echo Evaluation completed. Check results/gpu_scientific_dcbs.json
pause 