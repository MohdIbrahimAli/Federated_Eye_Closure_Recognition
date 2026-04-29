import json
import socketserver
import threading
import time
from dataclasses import dataclass, field
from typing import Dict

HOST = "127.0.0.1"
PORT = 9000


@dataclass
class GlobalModel:
    version: int = 1
    # Derived thresholds sent back to clients
    recognition_threshold: float = 0.33
    ear_blink_threshold: float = 0.21
    # Aggregated population statistics (learned via FedAvg)
    global_mean_ear_open: float = 0.30
    global_blink_rate: float = 15.0
    global_brightness: float = 130.0
    global_face_detection_rate: float = 0.80
    global_recognition_accuracy: float = 0.0
    updates_seen: int = 0
    total_frames_seen: int = 0


@dataclass
class ClientSummary:
    """One round of local statistics from a single client."""
    frames_processed: int = 0
    mean_ear_open: float = 0.30
    blink_rate: float = 15.0
    mean_brightness: float = 130.0
    face_detection_rate: float = 0.80
    recognition_accuracy: float = 0.0
    last_seen: float = 0.0


@dataclass
class ServerState:
    clients: Dict[str, float] = field(default_factory=dict)
    client_summaries: Dict[str, ClientSummary] = field(default_factory=dict)
    model: GlobalModel = field(default_factory=GlobalModel)
    lock: threading.Lock = field(default_factory=threading.Lock)


STATE = ServerState()


def _fedavg(summaries: Dict[str, ClientSummary]) -> GlobalModel:
    """
    Weighted Federated Averaging (FedAvg).

    Each client's contribution is weighted by frames_processed — analogous to
    standard FedAvg where clients are weighted by local dataset size.

    From the aggregated population statistics we then derive the two adaptive
    thresholds that every client will use:

      ear_blink_threshold  = global_mean_ear_open * 0.72
          A blink is detected when EAR drops to ~72 % of the open-eye baseline.
          This factor (0.72) is a well-established heuristic from EAR literature.
          By grounding it in the population's actual open-eye EAR we avoid the
          hard-coded 0.21 that works poorly under different lighting or face sizes.

      recognition_threshold = 0.33 + brightness_correction - accuracy_correction
          Brighter scenes → tighter threshold (faces more distinguishable).
          Higher global recognition accuracy → we can afford a slightly stricter
          threshold; lower accuracy → loosen it to reduce false-unknowns.
    """
    total_frames = sum(s.frames_processed for s in summaries.values())
    if total_frames == 0:
        return STATE.model  # no data yet, keep current model

    def wavg(attr: str) -> float:
        return sum(getattr(s, attr) * s.frames_processed for s in summaries.values()) / total_frames

    global_ear_open = wavg("mean_ear_open")
    global_blink_rate = wavg("blink_rate")
    global_brightness = wavg("mean_brightness")
    global_face_rate = wavg("face_detection_rate")
    global_rec_acc = wavg("recognition_accuracy")

    # Derive thresholds from population statistics
    ear_thr = max(0.16, min(0.28, global_ear_open * 0.72))
    brightness_correction = (130.0 - global_brightness) / 1000.0
    accuracy_correction = (global_rec_acc - 0.5) * 0.04  # up to ±0.02
    rec_thr = max(0.22, min(0.45, 0.33 + brightness_correction - accuracy_correction))

    m = STATE.model
    m.global_mean_ear_open = round(global_ear_open, 5)
    m.global_blink_rate = round(global_blink_rate, 3)
    m.global_brightness = round(global_brightness, 2)
    m.global_face_detection_rate = round(global_face_rate, 4)
    m.global_recognition_accuracy = round(global_rec_acc, 4)
    m.ear_blink_threshold = round(ear_thr, 5)
    m.recognition_threshold = round(rec_thr, 5)
    m.total_frames_seen += total_frames
    return m


class HubHandler(socketserver.StreamRequestHandler):
    def handle(self):
        line = self.rfile.readline().decode("utf-8").strip()
        if not line:
            self._reply({"status": "error", "message": "Empty payload"})
            return

        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            self._reply({"status": "error", "message": "Invalid JSON"})
            return

        action = payload.get("action")
        if action == "register_client":
            self._register(payload)
        elif action == "get_global_model":
            self._global_model()
        elif action == "submit_update":
            self._submit_update(payload)
        elif action == "health":
            self._reply({"status": "ok", "time": time.time()})
        else:
            self._reply({"status": "error", "message": f"Unknown action: {action}"})

    def _register(self, payload):
        cid = payload.get("client_id")
        if not cid:
            self._reply({"status": "error", "message": "Missing client_id"})
            return

        with STATE.lock:
            STATE.clients[cid] = time.time()
            model = STATE.model

        print(f"[Server] Client registered: {cid}  |  active clients: {list(STATE.clients.keys())}")
        self._reply({
            "status": "ok",
            "client_id": cid,
            "global_model": self._model_dict(model),
        })

    def _global_model(self):
        with STATE.lock:
            model = STATE.model
            active_clients = list(STATE.clients.keys())

        self._reply({
            "status": "ok",
            "global_model": self._model_dict(model),
            "active_clients": active_clients,
        })

    def _submit_update(self, payload):
        cid = payload.get("client_id")
        update = payload.get("update", {})
        if not cid:
            self._reply({"status": "error", "message": "Missing client_id"})
            return

        with STATE.lock:
            STATE.clients[cid] = time.time()

            # Store this client's local summary
            summary = ClientSummary(
                frames_processed=int(update.get("frames_processed", 0)),
                mean_ear_open=float(update.get("mean_ear_open", 0.30)),
                blink_rate=float(update.get("blink_rate", 15.0)),
                mean_brightness=float(update.get("mean_brightness", 130.0)),
                face_detection_rate=float(update.get("face_detection_rate", 0.80)),
                recognition_accuracy=float(update.get("recognition_accuracy", 0.0)),
                last_seen=time.time(),
            )
            STATE.client_summaries[cid] = summary

            # Weighted FedAvg across all clients that have submitted at least once
            active = {k: v for k, v in STATE.client_summaries.items() if v.frames_processed > 0}
            model = _fedavg(active)
            model.updates_seen += 1
            model.version += 1

        # Log the FL round clearly
        print(
            f"\n[FL Round {model.version}]  clients={list(active.keys())}  "
            f"total_frames={model.total_frames_seen}"
        )
        print(
            f"  Population stats (weighted avg across {len(active)} client(s)):"
        )
        for k, s in active.items():
            print(
                f"    {k}: frames={s.frames_processed}  ear_open={s.mean_ear_open:.4f}"
                f"  blinks/min={s.blink_rate:.1f}  brightness={s.mean_brightness:.1f}"
                f"  face_rate={s.face_detection_rate:.2f}  rec_acc={s.recognition_accuracy:.2f}"
            )
        print(
            f"  Global model v{model.version}:  "
            f"ear_thr={model.ear_blink_threshold:.4f}  rec_thr={model.recognition_threshold:.4f}  "
            f"mean_ear_open={model.global_mean_ear_open:.4f}  "
            f"blinks/min={model.global_blink_rate:.1f}  "
            f"brightness={model.global_brightness:.1f}"
        )

        self._reply({"status": "ok", "global_model": self._model_dict(model)})

    @staticmethod
    def _model_dict(model: GlobalModel) -> dict:
        return {
            "version": model.version,
            "recognition_threshold": model.recognition_threshold,
            "ear_blink_threshold": model.ear_blink_threshold,
            "global_mean_ear_open": model.global_mean_ear_open,
            "global_blink_rate": model.global_blink_rate,
            "global_brightness": model.global_brightness,
            "global_face_detection_rate": model.global_face_detection_rate,
            "global_recognition_accuracy": model.global_recognition_accuracy,
            "updates_seen": model.updates_seen,
            "total_frames_seen": model.total_frames_seen,
        }

    def _reply(self, payload):
        self.wfile.write((json.dumps(payload) + "\n").encode("utf-8"))


def main():
    print(f"[Server] Federated Learning Hub starting at {HOST}:{PORT}")
    print("[Server] Waiting for clients to connect and submit summaries...")
    with socketserver.ThreadingTCPServer((HOST, PORT), HubHandler) as server:
        server.daemon_threads = True
        server.serve_forever()


if __name__ == "__main__":
    main()
