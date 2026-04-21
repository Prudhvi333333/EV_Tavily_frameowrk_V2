param(
  [string]$ProgressFile = "",
  [ValidateSet("dash", "streamlit")]
  [string]$Framework = "dash",
  [int]$Port = 8050
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if ($Framework -eq "dash") {
  $pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
  if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "python is not available in .venv. Run: python -m venv .venv"
  }
  & $pythonExe -c "import dash, dash_bootstrap_components, plotly; print('DASH_DEPS_OK')" | Out-Null
  if ($LASTEXITCODE -ne 0) {
    throw "Dash UI dependencies are missing. Run: .\.venv\Scripts\python -m pip install dash dash-bootstrap-components plotly"
  }
  $appPath = Join-Path $projectRoot "ui\reviewer_dash_app.py"
  if (-not (Test-Path -LiteralPath $appPath)) {
    throw "Dash UI app not found at ui/reviewer_dash_app.py"
  }
  if ($ProgressFile) {
    & $pythonExe $appPath --progress-file $ProgressFile --port $Port
  } else {
    & $pythonExe $appPath --port $Port
  }
} else {
  $streamlitExe = Join-Path $projectRoot ".venv\Scripts\streamlit.exe"
  if (-not (Test-Path -LiteralPath $streamlitExe)) {
    throw "streamlit is not installed in .venv. Run: .\.venv\Scripts\python -m pip install -r requirements.txt"
  }
  $appPath = Join-Path $projectRoot "ui\reviewer_app.py"
  if (-not (Test-Path -LiteralPath $appPath)) {
    throw "Streamlit UI app not found at ui/reviewer_app.py"
  }
  if ($ProgressFile) {
    & $streamlitExe run $appPath -- --progress-file $ProgressFile
  } else {
    & $streamlitExe run $appPath
  }
}
