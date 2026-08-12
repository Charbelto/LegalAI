# =============================================================================
# One-command clean experiment run.
#
#   .\run_experiment.ps1                       full run: 30 queries x 3 topologies
#                                              x 3 repeats x 2 arms = 540 runs
#   .\run_experiment.ps1 -Arms peft            PEFT arm only (270 runs, drops RQ2)
#   .\run_experiment.ps1 -Smoke                3 runs per arm, separate output files
#   .\run_experiment.ps1 -SkipBenchmark        re-analyse the existing runs file only
#   .\run_experiment.ps1 -GenerationProvider ollama    pre-pivot shared-model arm
#
# The two ARMS are the RQ2 ablation: 'peft' loads each expert's LoRA adapter,
# 'base' runs the identical base models untuned. They cannot share one server
# process (six models will not fit in 8GB), so each arm gets its own benchmark
# invocation and its own server; benchmark.py cross-checks the arm the server
# reports and aborts on a mismatch rather than mislabelling rows.
#
# Produces: benchmark_runs.jsonl (both arms, with an 'arm' column),
#           run_meta.json + run_meta_<arm>.json, analysis_summary.csv,
#           by_query_type.csv, significance.csv, results.json,
#           metrics_table.tex, metrics_table_ablation.tex, paper_figures/*.png
# =============================================================================

param(
    [switch]$Smoke,
    [switch]$SkipBenchmark,
    [switch]$BenchmarkOnly,   # generate answers now, analyse later once golds are reviewed
    [switch]$Resume,
    [string]$JudgeProvider = "",          # "ollama" | "openai" | "deepseek"; empty = whatever .env says
    [string]$JudgeModel = "",             # empty = whatever .env says
    [string]$GenerationProvider = "",     # "local_peft" | "ollama" | "deepseek"; empty = whatever .env
                                           # says. This is the system under test, not the judge - see
                                           # config.GENERATION_PROVIDER.
    [ValidateSet("both", "peft", "base")]
    [string]$Arms = "both",               # which experimental arm(s) to benchmark. "both" is the
                                           # default because RQ2 needs the untuned control.
    [int]$Repeats = 3,
    [int]$Concurrency = 0                 # max in-flight benchmark requests. 0 = auto: 8 for
                                           # deepseek generation, 1 (sequential) for ollama and
                                           # local_peft. Do not raise it for local_peft: one GPU
                                           # serialises anyway and client-side concurrency only
                                           # corrupts the latency measurements.
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== Legal AI experiment run ===" -ForegroundColor Cyan

# --- Interpreter ------------------------------------------------------------
# Use the venv's Python explicitly rather than whatever `python` happens to
# resolve to. This is not defensive tidiness: on this machine bare `python` is
# the system interpreter, which has torch but NOT accelerate, so the server it
# started could not honour `device_map` and every expert model silently failed to
# load. The benchmark then recorded 3/3 runs as errors with a message about a
# missing package that is in fact installed - in the venv.
#
# benchmark.py spawns the server with sys.executable, so fixing the interpreter
# here fixes the server subprocess too.
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Warning "No venv at $py; falling back to '$((Get-Command python -ErrorAction SilentlyContinue).Source)'."
    Write-Warning "If the local PEFT stack is installed in a venv, this run will fail to load models."
    $py = "python"
}
Write-Host "interpreter: $py"

# Fail fast if the interpreter cannot see the packages local generation needs,
# instead of discovering it one failed run at a time.
$effectiveProviderForCheck = if ($env:GENERATION_PROVIDER) { $env:GENERATION_PROVIDER } else { "" }
if ($effectiveProviderForCheck -ne "ollama" -and $effectiveProviderForCheck -ne "deepseek") {
    $missing = & $py -c "import importlib.util as u; print(','.join(m for m in ('torch','transformers','peft','accelerate','bitsandbytes') if u.find_spec(m) is None))"
    if ($missing) {
        Write-Error "Interpreter '$py' is missing required packages for local generation: $missing`nInstall them:`n    $py -m pip install torch --index-url https://download.pytorch.org/whl/cu126`n    $py -m pip install -r requirements-finetune.txt"
        exit 1
    }
    Write-Host "      local generation stack present (torch, transformers, peft, accelerate, bitsandbytes)" -ForegroundColor DarkGray
}

# --- Environment -------------------------------------------------------------
# Sampling (not greedy) so that repeats carry generation variance and the
# confidence intervals mean something.
$env:LEGALAI_DETERMINISTIC   = "0"
$env:LEGALAI_NUM_CTX         = "8192"
$env:LEGALAI_ENABLE_COMPL_AI = "0"      # canned answers must never enter an experiment
$env:LEGALAI_ENABLE_EURLEX_LIVE = "0"   # live legal search makes runs non-reproducible
$env:BENCH_REPEATS           = "$Repeats"
# Judge settings come from legalai\.env unless overridden on the command line.
if ($JudgeProvider) { $env:JUDGE_PROVIDER = $JudgeProvider }
if ($JudgeModel)    { $env:JUDGE_MODEL    = $JudgeModel }

# Generation (the system under test) - same override pattern as the judge.
# Default stays whatever legalai\.env says (GENERATION_PROVIDER=ollama unless
# you changed it), so a bare `.\run_experiment.ps1` keeps testing the local
# privacy-preserving setup. Pass -GenerationProvider deepseek for the
# comparison arm.
if ($GenerationProvider) { $env:GENERATION_PROVIDER = $GenerationProvider }

# Resolve the provider ONCE, from legalai\.env when not overridden, and export it
# so that what this script prints is what the server subprocess actually loads.
#
# Without this the two could disagree silently: the script fell back to a literal
# default for display while the server read .env, so a bare `-Smoke` announced
# "generation_provider=local_peft" and then started a DeepSeek server. Reading the
# file here means the banner below is the truth rather than an assumption.
if (-not $env:GENERATION_PROVIDER) {
    $envFile = Join-Path $PSScriptRoot ".env"
    if (Test-Path $envFile) {
        $match = Select-String -Path $envFile -Pattern '^\s*GENERATION_PROVIDER\s*=\s*(\S+)' | Select-Object -First 1
        if ($match) { $env:GENERATION_PROVIDER = $match.Matches[0].Groups[1].Value }
    }
}
if (-not $env:GENERATION_PROVIDER) { $env:GENERATION_PROVIDER = "local_peft" }

# Without this, "parallel" topologies are silently serialised by Ollama and the
# parallel-vs-sequential comparison measures queueing rather than structure.
# (Irrelevant to DeepSeek generation, but harmless to leave set - embeddings
# still run on local Ollama either way.)
if (-not $env:OLLAMA_NUM_PARALLEL) { $env:OLLAMA_NUM_PARALLEL = "4" }

$effectiveGeneration = if ($env:GENERATION_PROVIDER) { $env:GENERATION_PROVIDER } else { "local_peft" }
$effectiveConcurrency = if ($Concurrency -gt 0) { $Concurrency } elseif ($effectiveGeneration -eq "deepseek") { 8 } else { 1 }
Write-Host "generation_provider=$effectiveGeneration  deterministic=$($env:LEGALAI_DETERMINISTIC)  num_ctx=$($env:LEGALAI_NUM_CTX)  OLLAMA_NUM_PARALLEL=$($env:OLLAMA_NUM_PARALLEL)  concurrency=$effectiveConcurrency"

# Which arms to run. The ablation only exists for local_peft: the other providers
# have no adapters to switch off, so forcing two arms there would produce two
# identical passes labelled differently.
$armList = if ($effectiveGeneration -ne "local_peft") {
    Write-Host "      (generation_provider=$effectiveGeneration has no PEFT adapters; single pass, arm label 'peft' is not applicable)" -ForegroundColor DarkGray
    @("peft")
} elseif ($Arms -eq "both") { @("peft", "base") } else { @($Arms) }
Write-Host "arms=$($armList -join ', ')  repeats=$Repeats"

if ($effectiveGeneration -eq "deepseek" -and -not $env:DEEPSEEK_API_KEY) {
    Write-Warning "GENERATION_PROVIDER=deepseek but DEEPSEEK_API_KEY is not set in the environment or legalai\.env."
    $answer = Read-Host "Continue anyway? (y/N)"
    if ($answer -ne "y") { exit 1 }
}

# DeepSeek generation is deliberately no longer the default: DeepSeek is the
# judge, and a model scoring its own answers is self-preference biased. This was
# a disclosed limitation of the pre-pivot paper and the pivot's whole point is
# that it no longer applies.
if ($effectiveGeneration -eq "deepseek" -and (-not $env:JUDGE_PROVIDER -or $env:JUDGE_PROVIDER -eq "deepseek")) {
    Write-Warning "GENERATION_PROVIDER=deepseek with a DeepSeek judge means the judge is grading its own output."
    $answer = Read-Host "Continue anyway? (y/N)"
    if ($answer -ne "y") { exit 1 }
}

# For the PEFT arm every adapter must exist, or the run would silently measure
# base models under a 'peft' label. benchmark.py's preflight also refuses, but
# failing here costs seconds instead of a server start per arm.
if ($effectiveGeneration -eq "local_peft" -and $armList -contains "peft" -and -not $SkipBenchmark) {
    $missing = @()
    foreach ($role in @("legal", "news", "general_qa")) {
        if (-not (Test-Path "adapters\$role\adapter_config.json")) { $missing += $role }
    }
    if ($missing.Count -gt 0) {
        Write-Error "No LoRA adapter for: $($missing -join ', '). Train them first:`n    python finetune\prepare_datasets.py`n    python finetune\train_qlora.py`nOr run the control arm only: .\run_experiment.ps1 -Arms base"
        exit 1
    }
    Write-Host "      adapters present for legal, news, general_qa" -ForegroundColor DarkGray
}

# --- Preconditions -----------------------------------------------------------
Write-Host "`n[1/6] Checking Ollama..." -ForegroundColor Yellow
try {
    $tags = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5
    $models = $tags.models | ForEach-Object { $_.name }
    Write-Host "      models: $($models -join ', ')"
    # Only a local judge needs to be pulled; a hosted judge is checked separately.
    $effectiveProvider = if ($env:JUDGE_PROVIDER) { $env:JUDGE_PROVIDER } else { "ollama" }
    $effectiveJudge = $env:JUDGE_MODEL
    if (-not $BenchmarkOnly -and $effectiveProvider -eq "ollama" -and $effectiveJudge -and ($models -notcontains $effectiveJudge)) {
        Write-Warning "Local judge model '$effectiveJudge' is not pulled. Run: ollama pull $effectiveJudge"
        Write-Warning "The judge must differ from the model under test AND from the model that drafted the reference answers, or scores are self-preference biased."
        $answer = Read-Host "Continue anyway? (y/N)"
        if ($answer -ne "y") { exit 1 }
    }
} catch {
    Write-Error "Ollama is not reachable at localhost:11434. Start it first."
    exit 1
}

# The judge only runs during analysis, so a missing judge model does not block
# the benchmark itself.
if ($BenchmarkOnly) {
    Write-Host "      (benchmark-only: judge model not needed yet)" -ForegroundColor DarkGray
}

Write-Host "`n[2/6] Running validity tests..." -ForegroundColor Yellow
& $py -m pytest tests -q
if ($LASTEXITCODE -ne 0) {
    Write-Error "Validity tests failed. Fix these before spending GPU hours on a run."
    exit 1
}

# --- Benchmark ---------------------------------------------------------------
if (-not $SkipBenchmark) {
    Write-Host "`n[3/6] Benchmark ($($armList.Count) arm(s))..." -ForegroundColor Yellow
    $started = Get-Date
    $armIndex = 0
    foreach ($arm in $armList) {
        Write-Host "`n      --- arm: $arm ---" -ForegroundColor Cyan
        $benchArgs = @("--arm", $arm)
        if ($Smoke)  { $benchArgs += "--smoke" }
        if ($Resume) { $benchArgs += "--resume" }
        # The first arm truncates the runs file; later arms append, so both arms
        # end up in one benchmark_runs.jsonl distinguished by the 'arm' column.
        # With -Resume the file is always appended to anyway.
        if ($armIndex -gt 0 -and -not $Resume) { $benchArgs += "--append" }
        if ($effectiveConcurrency -gt 1) { $benchArgs += "--concurrency"; $benchArgs += "$effectiveConcurrency" }

        $armStarted = Get-Date
        # -u: unbuffered. Redirected to a file, Python block-buffers its output,
        # so a multi-hour run shows nothing until it exits - and nothing at all if
        # it is interrupted, including the reason it aborted.
        & $py -u benchmark.py @benchArgs
        if ($LASTEXITCODE -ne 0) { Write-Error "Benchmark failed on arm '$arm'."; exit 1 }
        Write-Host "      arm '$arm' took $((Get-Date) - $armStarted)"
        $armIndex++
    }
    Write-Host "      benchmark took $((Get-Date) - $started) in total"
} else {
    Write-Host "`n[3/6] Benchmark skipped (-SkipBenchmark)" -ForegroundColor DarkGray
}

if ($Smoke) {
    Write-Host "`nSmoke run complete -> benchmark_runs_smoke.jsonl" -ForegroundColor Green
    Write-Host "`nInspect the answers before committing to the full run:" -ForegroundColor Green
    Write-Host "    python -c ""import json; [print(r['arm'].ljust(5), r['mode'].ljust(10), round(r['elapsed_s'],1), '|', repr(r.get('response',''))[:120]) for r in map(json.loads, open('benchmark_runs_smoke.jsonl', encoding='utf-8')) if r.get('success')]""" -ForegroundColor Gray
    Write-Host "`nAny of these means STOP, not proceed:" -ForegroundColor Yellow
    Write-Host "  * The abstention sentence in every mode  -> retrieval is empty (check chunk count and Ollama)." -ForegroundColor Yellow
    Write-Host "  * peft and base answers identical        -> the adapters are inert or the arm switch is not taking effect." -ForegroundColor Yellow
    Write-Host "  * All modes identical within an arm      -> sampling is off; LEGALAI_DETERMINISTIC must be 0." -ForegroundColor Yellow
    Write-Host "  * A canned GPAI provider-tier template   -> COMPL-AI is enabled and is bypassing the graph." -ForegroundColor Yellow
    Write-Host "`nAlso multiply the elapsed_s values out: 540 runs at several minutes each is a multi-day job." -ForegroundColor Green
    Write-Host "Then run without -Smoke." -ForegroundColor Green
    exit 0
}

if ($BenchmarkOnly) {
    Write-Host "`nAnswers generated. Analysis skipped (-BenchmarkOnly)." -ForegroundColor Green
    Write-Host "Review the gold answers and annotate relevance, then run:" -ForegroundColor Green
    Write-Host "    .\run_experiment.ps1 -SkipBenchmark" -ForegroundColor Green
    exit 0
}

# --- Judge preflight ---------------------------------------------------------
# One live call before spending on ~1200: confirms the model id, that it returns
# strict JSON, and what the full pass will cost.
Write-Host "`n[3.5/6] Judge preflight..." -ForegroundColor Yellow
& $py llm_judge.py --check
if ($LASTEXITCODE -ne 0) {
    Write-Error "Judge preflight failed. Fix the judge configuration in .env before analysing."
    exit 1
}
$proceed = Read-Host "Proceed with the full judging pass? (y/N)"
if ($proceed -ne "y") {
    Write-Host "Stopped before judging. Re-run with -SkipBenchmark when ready." -ForegroundColor Yellow
    exit 0
}

# --- Analysis ----------------------------------------------------------------
Write-Host "`n[4/6] Analysis (LLM judge runs here; this is slow)..." -ForegroundColor Yellow
& $py analyze_results.py
if ($LASTEXITCODE -ne 0) { Write-Error "Analysis failed."; exit 1 }

Write-Host "`n[5/6] Charts and LaTeX table..." -ForegroundColor Yellow
& $py evaluate_workflows.py
& $py make_paper_figures.py
& $py make_topology_figure.py   # needs no run data, but keeps figures/ in one place

Write-Host "`n[6/6] Summary" -ForegroundColor Yellow
& $py scripts\print_summary.py

& $py llm_judge.py --spend

# Archive this pass so a later judge pass (or a later generation-provider pass)
# cannot overwrite it. Label carries both axes so a deepseek-generation run and
# an ollama-generation run stay distinguishable even when judged by the same model.
$provider = if ($env:JUDGE_PROVIDER) { $env:JUDGE_PROVIDER } else { "ollama" }
Write-Host "`n[7/7] Snapshotting run..." -ForegroundColor Yellow
& $py scripts\snapshot_run.py --label "gen-${effectiveGeneration}_judge-$provider"

Write-Host "`nDone. Upload to Overleaf: metrics_table.tex and paper_figures\*.png" -ForegroundColor Green
Write-Host "Then fill the [TBD] placeholders in main.tex from results.json." -ForegroundColor Green
