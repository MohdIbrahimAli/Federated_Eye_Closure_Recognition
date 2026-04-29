# Federated Eye Closure Recognition

A privacy-preserving biometric system built on **Federated Learning**. Each client observes its user locally (face recognition + blink detection), sends only statistical summaries to a central server, and receives back improved model parameters — raw images and face embeddings never leave the device.

---

## How to Run

### Step 1 — Set up the environment (once)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> Requires **Python 3.12**. Run `py -3.12 --version` to confirm it is installed.

---

### Step 2 — Start the server (Terminal 1)

```powershell
.\run_server.ps1
```

The server starts a Federated Learning hub at `127.0.0.1:9000`. Keep this terminal open — it prints every FL aggregation round with the full population statistics.

---

### Step 3 — Start the clients (Terminal 2 and Terminal 3)

```powershell
# Terminal 2
.\run_client1.ps1

# Terminal 3
.\run_client2.ps1
```

Each client opens a GUI window with a live camera feed and a **Federated Learning Activity** panel.

**Or launch all three windows at once:**

```powershell
.\run_demo.ps1
```

---

### What you will see

**Server terminal** prints an FL round every 15 seconds:

```
[FL Round 3]  clients=['client1', 'client2']  total_frames=2700
  Population stats (weighted avg across 2 client(s)):
    client1: frames=450  ear_open=0.3104  blinks/min=18.5  brightness=127.3  face_rate=0.92  rec_acc=0.85
    client2: frames=380  ear_open=0.2891  blinks/min=14.2  brightness=118.6  face_rate=0.88  rec_acc=0.72
  Global model v3:  ear_thr=0.2158  rec_thr=0.3384  mean_ear_open=0.3007  blinks/min=16.6  brightness=123.4
```

**Client GUI** shows a live "Federated Learning Activity" panel:

- **Last summary sent** — the 6-field statistical summary this client just uploaded
- **Global model received** — the aggregated thresholds the server sent back
- **Next FL push in** — countdown to the next federated round

---

## How the Federated Learning Works

```
Client 1                       Server                        Client 2
────────                       ──────                        ────────
Observe frames locally         Weighted FedAvg               Observe frames locally
  ↓                              ↓                             ↓
Compute summary:               global_ear_open =             Compute summary:
  mean_ear_open                  Σ(frames_i × ear_i)           mean_ear_open
  blink_rate            ──→      ─────────────────   ←──       blink_rate
  mean_brightness                  Σ(frames_i)                 mean_brightness
  face_detection_rate                                          face_detection_rate
  recognition_accuracy           Derive thresholds:           recognition_accuracy
  frames_processed               ear_thr = ear_open × 0.72    frames_processed
                                 rec_thr = f(brightness,
                       ──→         accuracy)           ←──
Apply updated thresholds       Return global model           Apply updated thresholds
```

### What each client sends (summary)

| Field | Description |
|---|---|
| `mean_ear_open` | Average Eye Aspect Ratio when eyes are open — the personal baseline |
| `blink_rate` | Blinks per minute observed this interval |
| `mean_brightness` | Average frame brightness (0–255) |
| `face_detection_rate` | Fraction of frames where a face was detected |
| `recognition_accuracy` | Fraction of recognition attempts that identified a known face |
| `frames_processed` | Total frames this interval — used as the **FedAvg weight** |

No raw images, no face embeddings, and no identity names are ever sent to the server.

### What the server computes (Weighted FedAvg)

Each client is weighted by `frames_processed` (analogous to local dataset size in standard FedAvg):

```
global_mean_ear_open = Σ( frames_i × mean_ear_open_i ) / Σ( frames_i )
global_blink_rate    = Σ( frames_i × blink_rate_i    ) / Σ( frames_i )
global_brightness    = Σ( frames_i × mean_brightness_i) / Σ( frames_i )
```

Thresholds are then **derived** from the population statistics:

```
ear_blink_threshold  = global_mean_ear_open × 0.72
    # A blink = eyes ~72% closed relative to the observed open-eye baseline.
    # Adapts to each user's face size, camera distance, and lighting.

recognition_threshold = 0.33
                      + (130 - global_brightness) / 1000   # darker → more lenient
                      - (global_rec_accuracy - 0.5) × 0.04 # higher accuracy → stricter
```

### What clients receive back

```json
{
  "version": 5,
  "ear_blink_threshold": 0.2158,
  "recognition_threshold": 0.3384,
  "global_mean_ear_open": 0.3007,
  "global_blink_rate": 16.6,
  "global_brightness": 123.4,
  "total_frames_seen": 4500
}
```

Both thresholds take effect immediately on the next frame — blink detection and face recognition improve without any raw data leaving the device.

---

## Client GUI Features

| Feature | How to use |
|---|---|
| **Register Face** | Click → enter a name → hold your face still for ~2 seconds |
| **Enable Recognition** | Toggle on to identify faces against local registered profiles |
| **Set Blink Password** | Records your blink pattern over 10 seconds as a biometric key |
| **Verify Blink Password** | Replays your pattern to authenticate |
| **Lock File** | Encrypts any file using your blink pattern as the key (Fernet + PBKDF2) |
| **Unlock File** | Decrypts a `.lock` file by verifying your blink pattern |

---

## Privacy and Isolation

- Each client's face profiles are stored locally under `data/client1` and `data/client2`.
- Client-1 identities are never visible to Client-2 or the server.
- The server stores only scalar statistics per client — no biometric data.
- File encryption keys are derived locally and never transmitted.

---

## Note for single-webcam setups

Some drivers do not allow one physical camera to be opened by two processes simultaneously. If both clients are needed on one machine, use a virtual camera tool (e.g. OBS Virtual Camera) as a second source, or run clients on separate machines pointing at the same server IP.
