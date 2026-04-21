param(
  [switch]$Tune,
  [switch]$AllModels,
  [int]$Limit = 0,
  [ValidateSet("test", "train", "both")]
  [string]$EvalSplit = "test",
  [switch]$UseKimiCloudJudge,
  [ValidateSet("hybrid", "llamaindex")]
  [string]$RetrievalBackend = "llamaindex",
  [string]$ExtraTestQuestions = "",
  [Alias("EnableRAGatouilleReranker")]
  [switch]$EnableReranker,
  [Alias("DisableRAGatouilleReranker")]
  [switch]$DisableReranker,
  [switch]$EnableWebReranker,
  [switch]$DisableWebReranker
)

$ErrorActionPreference = "Stop"

function Get-OllamaBaseUrl([string]$hostValue) {
  if ([string]::IsNullOrWhiteSpace($hostValue)) {
    return "http://127.0.0.1:11434"
  }
  $trimmed = $hostValue.Trim()
  if ($trimmed -match "^https?://") {
    return $trimmed.TrimEnd("/")
  }
  return ("http://" + $trimmed).TrimEnd("/")
}

function Invoke-OllamaProbe([string]$baseUrl, [string]$model, [string]$prompt, [string]$expectedToken) {
  $uri = "$baseUrl/api/generate"
  $payload = @{
    model = $model
    prompt = $prompt
    stream = $false
    options = @{ temperature = 0 }
  } | ConvertTo-Json -Depth 6

  try {
    $resp = Invoke-RestMethod -Method Post -Uri $uri -ContentType "application/json" -Body $payload -TimeoutSec 120
    $text = [string]($resp.response)
    if ([string]::IsNullOrWhiteSpace($text)) {
      throw "empty_response"
    }
    if ($text -notmatch [regex]::Escape($expectedToken)) {
      throw ("unexpected_response:`n" + $text)
    }
  } catch {
    $detail = $_.Exception.Message
    if ($_.ErrorDetails -and $_.ErrorDetails.Message) {
      $detail = "$detail | $($_.ErrorDetails.Message)"
    }
    throw "Ollama probe failed for model '$model': $detail"
  }
}

function Invoke-OllamaEmbedProbe([string]$baseUrl, [string]$model) {
  $payload = @{
    model = $model
    input = @("search_query: health check")
    keep_alive = "0s"
  } | ConvertTo-Json -Depth 6

  try {
    $resp = Invoke-RestMethod -Method Post -Uri "$baseUrl/api/embed" -ContentType "application/json" -Body $payload -TimeoutSec 180
    $embeddings = $resp.embeddings
    if (-not $embeddings -or $embeddings.Count -lt 1) {
      throw "empty_embeddings"
    }
  } catch {
    try {
      $legacyPayload = @{
        model = $model
        prompt = "search_query: health check"
        keep_alive = "0s"
      } | ConvertTo-Json -Depth 6
      $legacy = Invoke-RestMethod -Method Post -Uri "$baseUrl/api/embeddings" -ContentType "application/json" -Body $legacyPayload -TimeoutSec 180
      if (-not $legacy.embedding) {
        throw "empty_legacy_embedding"
      }
      return
    } catch {
      $detail = $_.Exception.Message
      if ($_.ErrorDetails -and $_.ErrorDetails.Message) {
        $detail = "$detail | $($_.ErrorDetails.Message)"
      }
      throw "Ollama embed probe failed for model '$model': $detail"
    }
  }
}

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
      # Always prefer explicit values from .env in this run context so stale shell values
      # (for example placeholder OPENROUTER_API_KEY) do not override user-provided keys.
      if ($val -ne "") {
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

Write-Host "Preflight: checking Python dependencies..."
$dependencyCheck = @'
import importlib
import sys

modules = [
    "chromadb",
    "yaml",
    "rank_bm25",
    "duckduckgo_search",
    "openpyxl",
    "pandas",
    "numpy",
    "sklearn",
    "sentence_transformers",
    "dotenv",
    "psutil",
    "tavily",
    "firecrawl",
    "streamlit",
    "llama_index.core",
    "llama_index.vector_stores.chroma",
]

missing = []
for mod in modules:
    try:
        importlib.import_module(mod)
    except Exception as exc:
        missing.append(f"{mod}: {exc.__class__.__name__}")

if missing:
    print("DEPENDENCY_CHECK_FAILED")
    for item in missing:
        print(item)
    sys.exit(1)

print("DEPENDENCY_CHECK_OK")
'@
$dependencyCheck | & $pythonExe -
if ($LASTEXITCODE -ne 0) {
  throw "Dependency check failed. Install missing packages with: .\.venv\Scripts\python -m pip install -r requirements.txt"
}

$env:USE_KIMI_CLOUD_JUDGE = if ($UseKimiCloudJudge) { "1" } else { "0" }
$modelPlanScript = @'
import json
import os
import yaml

with open("config/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

models_cfg = cfg.get("models", {}) or {}
qwen_model = str(models_cfg.get("qwen", "qwen2.5:14b")).strip()
embed_model = str((cfg.get("embeddings", {}) or {}).get("model", "nomic-embed-text")).strip()
required = []
if qwen_model:
    required.append(qwen_model)
if embed_model:
    required.append(embed_model)

use_cloud = os.getenv("USE_KIMI_CLOUD_JUDGE", "0") == "1"
if use_cloud:
    required.append("kimi-k2.5:cloud")
else:
    eval_judge = (cfg.get("evaluation", {}) or {}).get("judge", {}) or {}
    web_judge = (cfg.get("web_validator", {}) or {}).get("judge", {}) or {}
    if str(eval_judge.get("provider", "ollama")).lower() == "ollama":
        model = str(eval_judge.get("model", "")).strip()
        if model:
            required.append(model)
    if str(web_judge.get("provider", "ollama")).lower() == "ollama":
        model = str(web_judge.get("model", "")).strip()
        if model:
            required.append(model)

out = {
    "qwen_model": qwen_model,
    "embedding_model": embed_model,
    "required_models": sorted(set(required)),
}
print(json.dumps(out))
'@
$modelPlanJson = $modelPlanScript | & $pythonExe -
if ($LASTEXITCODE -ne 0) {
  throw "Failed to parse config/config.yaml for required model checks."
}
$modelPlan = $modelPlanJson | ConvertFrom-Json
$qwenModel = [string]$modelPlan.qwen_model
$embeddingModel = [string]$modelPlan.embedding_model
$requiredModels = @($modelPlan.required_models)
if ([string]::IsNullOrWhiteSpace($qwenModel)) {
  throw "models.qwen is empty in config/config.yaml."
}
if ([string]::IsNullOrWhiteSpace($embeddingModel)) {
  throw "embeddings.model is empty in config/config.yaml."
}

Write-Host "Preflight: checking Ollama server connectivity..."
$configuredHost = [Environment]::GetEnvironmentVariable("OLLAMA_HOST", "Process")
$configuredBase = [Environment]::GetEnvironmentVariable("OLLAMA_BASE_URL", "Process")
$candidateInputs = @()
if ($configuredHost) { $candidateInputs += $configuredHost }
if ($configuredBase) { $candidateInputs += $configuredBase }
$candidateInputs += @(
  "http://127.0.0.1:11435",
  "127.0.0.1:11435",
  "http://localhost:11435",
  "http://127.0.0.1:11434",
  "127.0.0.1:11434",
  "http://localhost:11434"
)
$candidateInputs = $candidateInputs | Select-Object -Unique

$resolvedOllamaBaseUrl = $null
$ollamaTags = $null
foreach ($candidate in $candidateInputs) {
  $baseUrl = Get-OllamaBaseUrl $candidate
  try {
    $resp = Invoke-RestMethod -Method Get -Uri "$baseUrl/api/tags" -TimeoutSec 8
    if ($resp -and $resp.models -is [System.Array]) {
      $resolvedOllamaBaseUrl = $baseUrl
      $ollamaTags = $resp
      break
    }
  } catch {
    continue
  }
}

if (-not $resolvedOllamaBaseUrl) {
  throw "Ollama server is not reachable. Start it first (example: `$env:OLLAMA_HOST='127.0.0.1:11435'; ollama serve)."
}
$uri = [Uri]$resolvedOllamaBaseUrl
[Environment]::SetEnvironmentVariable("OLLAMA_BASE_URL", $resolvedOllamaBaseUrl, "Process")
[Environment]::SetEnvironmentVariable("OLLAMA_HOST", "$($uri.Host):$($uri.Port)", "Process")
Write-Host "Preflight: connected to Ollama via OLLAMA_BASE_URL=$resolvedOllamaBaseUrl"

Write-Host "Preflight: checking required Ollama models..."
$names = @()
if ($ollamaTags -and $ollamaTags.models) {
  $names = @($ollamaTags.models | ForEach-Object { [string]$_.name })
}
foreach ($model in $requiredModels) {
  $expected = @($model, "$model`:latest")
  $present = $false
  foreach ($name in $names) {
    if ($expected -contains $name) {
      $present = $true
      break
    }
  }
  if (-not $present) {
    throw "Required Ollama model not found locally: $model. Pull it first with: ollama pull $model"
  }
}

$env:RERANKER_FORCE_ENABLE = if ($EnableReranker) { "1" } else { "0" }
$env:RERANKER_FORCE_DISABLE = if ($DisableReranker) { "1" } else { "0" }
Write-Host "Preflight: checking cross-encoder cache (no downloads allowed)..."
$crossEncoderCheck = @'
import os
import sys
import yaml
from sentence_transformers import CrossEncoder

with open("config/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

rer_cfg = cfg.get("reranker", {})
enabled = bool(rer_cfg.get("enabled", False))
if os.getenv("RERANKER_FORCE_ENABLE") == "1":
    enabled = True
if os.getenv("RERANKER_FORCE_DISABLE") == "1":
    enabled = False

models_to_check = []
if enabled:
    models_to_check.append(str(rer_cfg.get("model", "cross-encoder/ms-marco-MiniLM-L6-v2")))

web_cfg = cfg.get("web_validator", {})
web_rerank_enabled = bool(web_cfg.get("rerank_enabled", True))
if web_rerank_enabled and bool(web_cfg.get("cross_encoder_local_files_only", True)):
    models_to_check.append(
        str(cfg.get("embeddings", {}).get("cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L6-v2"))
    )

models_to_check = sorted(set(models_to_check))
for model_name in models_to_check:
    try:
        CrossEncoder(model_name, local_files_only=True)
    except Exception as exc:
        print("CROSS_ENCODER_CACHE_MISSING")
        print(model_name)
        print(type(exc).__name__)
        sys.exit(1)

print("CROSS_ENCODER_CACHE_OK")
'@
$crossEncoderCheck | & $pythonExe -
if ($LASTEXITCODE -ne 0) {
  throw (
    "Cross-encoder model cache check failed. Download each configured model once, for example: " +
    ".\.venv\Scripts\python -c `"from sentence_transformers import CrossEncoder; " +
    "CrossEncoder('<MODEL_NAME>', local_files_only=False); print('CE_OK')`""
  )
}

if (-not $env:TAVILY_API_KEY) {
  throw "TAVILY_API_KEY is not set. It is required for rag_pretrained_web pipeline."
}

if (-not $env:FIRECRAWL_API_KEY) {
  throw "FIRECRAWL_API_KEY is not set. It is required for rag_pretrained_web pipeline."
}

if ($UseKimiCloudJudge) {
  Write-Host "Preflight: probing Kimi cloud judge..."
  Invoke-OllamaProbe -baseUrl $resolvedOllamaBaseUrl -model "kimi-k2.5:cloud" -prompt "Reply only: KIMI_OK" -expectedToken "KIMI_OK"
}

Write-Host "Preflight: probing Qwen generation model..."
Invoke-OllamaProbe -baseUrl $resolvedOllamaBaseUrl -model $qwenModel -prompt "Reply only: QWEN_OK" -expectedToken "QWEN_OK"

Write-Host "Preflight: probing embedding model..."
Invoke-OllamaEmbedProbe -baseUrl $resolvedOllamaBaseUrl -model $embeddingModel

Write-Host "Running tests..."
& $pythonExe -m pytest -q
if ($LASTEXITCODE -ne 0) {
  throw "Tests failed. Stopping pipeline run."
}

if ($Tune) {
  Write-Host "Running train-set tuning..."
  & $pythonExe scripts/tune_on_train.py
  if ($LASTEXITCODE -ne 0) {
    throw "Train tuning failed. Stopping pipeline run."
  }
}

$pipelines = @("rag", "no_rag", "rag_pretrained", "rag_pretrained_web")
if ($AllModels) {
  $models = @("qwen", "gemma", "gemini")
} else {
  $models = @("qwen")
}

$cmdArgs = @("main.py", "--models") + $models + @("--pipelines") + $pipelines + @("--eval-split", $EvalSplit, "--retrieval-backend", $RetrievalBackend)
if ($Limit -gt 0) {
  $cmdArgs += @("--limit", "$Limit")
}
if ($UseKimiCloudJudge) {
  $cmdArgs += @("--use-kimi-cloud-judge")
}
if ($EnableReranker) {
  $cmdArgs += @("--enable-reranker")
}
if ($DisableReranker) {
  $cmdArgs += @("--disable-reranker")
}
if ($EnableWebReranker) {
  $cmdArgs += @("--enable-web-reranker")
}
if ($DisableWebReranker) {
  $cmdArgs += @("--disable-web-reranker")
}
if (-not [string]::IsNullOrWhiteSpace($ExtraTestQuestions)) {
  $cmdArgs += @("--extra-test-questions", $ExtraTestQuestions)
}

Write-Host "Running main pipeline..."
& $pythonExe @cmdArgs
if ($LASTEXITCODE -ne 0) {
  throw "Main pipeline failed."
}

Write-Host "Done. Reports are in outputs/reports."
