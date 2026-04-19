param(
  [switch]$Tune,
  [switch]$AllModels,
  [int]$Limit = 0
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
  throw "Virtual environment not found at .venv. Run: python -m venv .venv"
}

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
  throw "Ollama CLI not found. Install Ollama and ensure it is in PATH."
}

Write-Host "Installing/updating dependencies..."
& $pythonExe -m pip install -r requirements.txt

Write-Host "Ensuring Ollama models are present..."
& ollama pull qwen2.5:14b
& ollama pull nomic-embed-text

Write-Host "Running tests..."
& $pythonExe -m pytest -q

if ($Tune) {
  Write-Host "Running train-set tuning..."
  & $pythonExe scripts/tune_on_train.py
}

$pipelines = @("rag", "no_rag", "rag_pretrained", "rag_pretrained_web")
if ($AllModels) {
  $models = @("qwen", "gemma", "gemini")
} else {
  $models = @("qwen")
}

$cmdArgs = @("main.py", "--models") + $models + @("--pipelines") + $pipelines
if ($Limit -gt 0) {
  $cmdArgs += @("--limit", "$Limit")
}

Write-Host "Running main pipeline..."
& $pythonExe @cmdArgs

Write-Host "Done. Reports are in outputs/reports."

