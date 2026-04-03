import json
import socketserver
import threading
import time
import sys
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

HOST = "127.0.0.1"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9000

def average_weights(current, new, n):
    """Recursively average two nested weight structures."""
    if isinstance(current, list) and isinstance(new, list):
        return [average_weights(c, nw, n) for c, nw in zip(current, new)]
    else:
        return (current * n + new) / (n + 1)
@dataclass
class GlobalModel:
    version: int = 1
    recognition_threshold: float = 0.33
    ear_blink_threshold: float = 0.21
    model_weights: List[List[float]] = field(default_factory=list)
    updates_seen: int = 0


@dataclass
class ServerState:
    clients: Dict[str, float] = field(default_factory=dict)
    model: GlobalModel = field(default_factory=GlobalModel)
    lock: threading.Lock = field(default_factory=threading.Lock)


STATE = ServerState()


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

        self._reply(
            {
                "status": "ok",
                "client_id": cid,
                "global_model": {
                    "version": model.version,
                    "recognition_threshold": model.recognition_threshold,
                    "ear_blink_threshold": model.ear_blink_threshold,
                    "model_weights": model.model_weights,
                },
            }
        )

    def _global_model(self):
        with STATE.lock:
            model = STATE.model
            active_clients = list(STATE.clients.keys())

        self._reply(
            {
                "status": "ok",
                "global_model": {
                    "version": model.version,
                    "recognition_threshold": model.recognition_threshold,
                    "ear_blink_threshold": model.ear_blink_threshold,
                    "model_weights": model.model_weights,
                    "updates_seen": model.updates_seen,
                },
                "active_clients": active_clients,
            }
        )

    def _submit_update(self, payload):
        cid = payload.get("client_id")
        update = payload.get("update", {})
        if not cid:
            self._reply({"status": "error", "message": "Missing client_id"})
            return

        with STATE.lock:
            STATE.clients[cid] = time.time()
            model = STATE.model

            rec_thr = float(update.get("recognition_threshold", model.recognition_threshold))
            new_weights = update.get("model_weights", [])

            # Federated aggregation of scalar parameters and model weights.
            n = model.updates_seen
            model.recognition_threshold = (model.recognition_threshold * n + rec_thr) / (n + 1)
            if new_weights and not model.model_weights:
                model.model_weights = new_weights
            elif new_weights:
                # Average the weights recursively
                model.model_weights = [average_weights(layer_a, layer_b, n) for layer_a, layer_b in zip(model.model_weights, new_weights)]
            model.updates_seen += 1
            model.version += 1

            response_model = {
                "version": model.version,
                "recognition_threshold": model.recognition_threshold,
                "ear_blink_threshold": model.ear_blink_threshold,
                "model_weights": model.model_weights,
                "updates_seen": model.updates_seen,
            }

        self._reply({"status": "ok", "global_model": response_model})

    def _reply(self, payload):
        self.wfile.write((json.dumps(payload) + "\n").encode("utf-8"))


def main():
    print(f"[Server] Starting federated hub at {HOST}:{PORT}")
    with socketserver.ThreadingTCPServer((HOST, PORT), HubHandler) as server:
        server.daemon_threads = True
        server.serve_forever()


if __name__ == "__main__":
    main()
