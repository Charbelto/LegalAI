#!/bin/bash
# Bash script to run the complete Legal AI experiment pipeline

export LEGALAI_DETERMINISTIC=1
export LEGALAI_LLM_SEED=42
export JUDGE_MODEL=${JUDGE_MODEL:-llama3.1:8b}

echo "=== Starting Experiment Pipeline ==="

# 1. Smoke test gate
echo "[1/4] Running smoke test gate..."
.venv/bin/python benchmark.py --smoke
if [ $? -ne 0 ]; then
    echo "Smoke test failed! Exiting pipeline."
    exit 1
fi
echo "Smoke test passed successfully."

# 2. Full benchmark run
echo "[2/4] Running full benchmark (R = 5 repeats, 8 modes)..."
.venv/bin/python benchmark.py --resume
if [ $? -ne 0 ]; then
    echo "Benchmark run failed! Exiting pipeline."
    exit 1
fi

# 3. Statistical Analysis
echo "[3/4] Running aggregation & significance analysis..."
.venv/bin/python analyze_results.py
if [ $? -ne 0 ]; then
    echo "Analysis failed! Exiting pipeline."
    exit 1
fi

# 4. Generate Visualizations & Tables
echo "[4/4] Generating publication-grade charts and LaTeX tables..."
.venv/bin/python evaluate_workflows.py
if [ $? -ne 0 ]; then
    echo "Chart generation failed! Exiting pipeline."
    exit 1
fi

echo "=== Experiment Pipeline Completed Successfully ==="
