param(
  [switch]$Tune,
  [switch]$AllModels,
  [int]$Limit = 0
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$envPath = Join-Path $projectRoot ".env"
if (Test-Path -LiteralPath $envPath) {
  Get-Content -LiteralPath $envPath | ForEach-Object {
    $line = $_.Trim()
    if (-not $line -or $line.StartsWith("#")) { return }
    $parts = $line -split "=", 2
    if ($parts.Count -ne 2) { return }
    $key = $parts[0].Trim()
    $val = $parts[1].Trim().Trim('"').Trim("'")
    if ($key) {
      $existing = [Environment]::GetEnvironmentVariable($key, "Process")
      if (-not $existing) {
        [Environment]::SetEnvironmentVariable($key, $val, "Process")
      }
    }
  }
}

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

if (-not $env:TAVILY_API_KEY) {
  throw "TAVILY_API_KEY is not set. It is required for rag_pretrained_web pipeline."
}

if (-not $env:FIRECRAWL_API_KEY) {
  throw "FIRECRAWL_API_KEY is not set. It is required for rag_pretrained_web pipeline."
}

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
