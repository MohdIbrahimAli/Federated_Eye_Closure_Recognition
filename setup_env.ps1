$ErrorActionPreference = 'Stop'

function Resolve-Python312 {
    if ($env:PY312 -and (Test-Path $env:PY312)) {
        return @{ exe = $env:PY312; args = @() }
    }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return @{ exe = 'py'; args = @('-3.12') }
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return @{ exe = 'python'; args = @() }
    }

    throw 'Python not found. Install Python 3.12 or set $env:PY312 to python.exe path.'
}

$pyInfo = Resolve-Python312
$exe = $pyInfo.exe
$args = $pyInfo.args

Write-Host "Using interpreter launcher: $exe $($args -join ' ')" -ForegroundColor Cyan

& $exe @args -m venv .venv

$venvPython = Join-Path $PSScriptRoot '.venv\\Scripts\\python.exe'
if (!(Test-Path $venvPython)) {
    throw "Virtual environment creation failed: $venvPython not found"
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt

& $venvPython -c "import cv2, mediapipe; print('OK: cv2', cv2.__version__); print('OK: mediapipe', mediapipe.__version__)"

Write-Host 'Environment ready. In VS Code: Ctrl+Shift+P -> Python: Select Interpreter -> .venv\\Scripts\\python.exe' -ForegroundColor Green
