$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    python -m venv (Join-Path $ProjectRoot ".venv")
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
& $VenvPython -m unittest discover -s (Join-Path $ProjectRoot "tests")

Write-Host "Setup complete. Start the app with:"
Write-Host ".\.venv\Scripts\python.exe -m streamlit run streamlit_app.py"
