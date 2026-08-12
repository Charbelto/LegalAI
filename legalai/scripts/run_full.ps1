# PowerShell script to run the complete Legal AI Multi-Agent vs Single-Agent experiment

$env:LEGALAI_DETERMINISTIC="1"
$env:LEGALAI_LLM_SEED="42"
$env:JUDGE_MODEL="llama3.1:8b"

Write-Host "=== Starting Experiment Pipeline ===" -ForegroundColor Green

# 1. Smoke test gate
Write-Host "[1/4] Running smoke test gate..." -ForegroundColor Yellow
.venv\Scripts\python.exe benchmark.py --smoke
if ($LASTEXITCODE -ne 0) {
    Write-Error "Smoke test failed! Exiting pipeline."
    Exit $LASTEXITCODE
}
Write-Host "Smoke test passed successfully." -ForegroundColor Green

# 2. Full benchmark run
Write-Host "[2/4] Running full benchmark (R = 5 repeats, 8 modes)..." -ForegroundColor Yellow
.venv\Scripts\python.exe benchmark.py --resume
if ($LASTEXITCODE -ne 0) {
    Write-Error "Benchmark run failed! Exiting pipeline."
    Exit $LASTEXITCODE
}

# 3. Statistical Analysis
Write-Host "[3/4] Running aggregation & significance analysis..." -ForegroundColor Yellow
.venv\Scripts\python.exe analyze_results.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Analysis failed! Exiting pipeline."
    Exit $LASTEXITCODE
}

# 4. Generate Visualizations & Tables
Write-Host "[4/4] Generating publication-grade charts and LaTeX tables..." -ForegroundColor Yellow
.venv\Scripts\python.exe evaluate_workflows.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Chart generation failed! Exiting pipeline."
    Exit $LASTEXITCODE
}

Write-Host "=== Experiment Pipeline Completed Successfully ===" -ForegroundColor Green
