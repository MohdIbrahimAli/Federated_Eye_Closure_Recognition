$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$venvPython = Join-Path $scriptDir '.venv\Scripts\python.exe'
$py = if (Test-Path $venvPython) { $venvPython } elseif ($env:PY312) { $env:PY312 } elseif (Get-Command py -ErrorAction SilentlyContinue) { 'py -3.12' } elseif (Get-Command python -ErrorAction SilentlyContinue) { 'python' } else { throw 'Python 3.12 not found. Set $env:PY312 to your python.exe path.' }
Invoke-Expression "$py server.py"