param(
  [string]$ProgressFile = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$streamlitExe = Join-Path $projectRoot ".venv\Scripts\streamlit.exe"
if (-not (Test-Path -LiteralPath $streamlitExe)) {
  throw "streamlit is not installed in .venv. Run: .\.venv\Scripts\python -m pip install -r requirements.txt"
}

$appPath = Join-Path $projectRoot "ui\reviewer_app.py"
if (-not (Test-Path -LiteralPath $appPath)) {
  throw "UI app not found at ui/reviewer_app.py"
}

if ($ProgressFile) {
  & $streamlitExe run $appPath -- --progress-file $ProgressFile
} else {
  & $streamlitExe run $appPath
}
