@echo off
echo Starting Llama 3.2 1B evaluation on ARC Easy...
echo.
python run_arc_eval.py --model meta-llama/Llama-3.2-1B --data data/arc_easy_full.json --limit 50 --output results/arc_llama_test.json
echo.
echo Evaluation complete!
pause 