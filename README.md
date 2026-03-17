# Federated Physiognomy + Rapid Eye Closure Simulation

This project simulates a federated learning workflow for face recognition (physiognomy-style facial profiling) and rapid eye-closure (blink) detection.

## Key properties
- 1 central server (`server.py`) + 2 independent clients (`client.py`)
- Facial data never leaves the client machine/folder
- Server only receives lightweight model updates (no raw images, no embeddings)
- Two clients can run simultaneously with isolated local profile stores
- Blink password can be used to lock/unlock files locally
- Low-light and high-exposure frame optimization in the live pipeline

## Python version
Use **Python 3.12** for the full project.

## Quick fix for `cv2` import errors
```powershell
.\setup_env.ps1
```
This creates `.venv`, installs dependencies, and verifies `cv2` + `mediapipe` imports.

## Manual install
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `py` is not available, set:
```powershell
$env:PY312="C:\\Path\\To\\Python312\\python.exe"
```

## Run
1. Start server:
```powershell
.\run_server.ps1
```

2. Start client 1 and client 2 in separate terminals:
```powershell
.\run_client1.ps1
.\run_client2.ps1
```

Or launch all 3 windows automatically:
```powershell
.\run_demo.ps1
```

## Client controls
- `r`: Register current face as a local identity
- `s`: Start/stop face recognition mode
- `b`: Set blink password
- `v`: Verify blink password
- `l`: Lock a file (encrypt)
- `u`: Unlock a file (decrypt)
- `q`: Quit

## Privacy and isolation
- Client local data is stored under each data directory (`data/client1`, `data/client2`).
- Client-1 identities are not visible to Client-2 and vice versa.
- The central server aggregates only scalar tuning values for thresholds.

## Note for single webcam setups
Some drivers do not allow one physical camera stream to be opened by two apps at the same time. If this happens during presentation, use a virtual camera or two camera sources.
