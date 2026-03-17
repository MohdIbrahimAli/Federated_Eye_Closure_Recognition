import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from common import (
    LEFT_EYE_IDX,
    RIGHT_EYE_IDX,
    blink_pattern_from_timestamps,
    build_face_embedding,
    compute_ear,
    cosine_distance,
    enhance_frame,
    ensure_dir,
    load_json,
    lock_file_with_secret,
    salted_hash,
    save_json,
    send_message,
    unlock_file_with_secret,
    verify_salted_hash,
)


class LocalStore:
    def __init__(self, data_dir: Path):
        self.data_dir = ensure_dir(data_dir)
        self.profiles_path = self.data_dir / "profiles.json"
        self.security_path = self.data_dir / "security.json"
        self.profiles: Dict[str, List[float]] = load_json(self.profiles_path, {})
        self.security: Dict[str, str] = load_json(self.security_path, {})

    def save_profiles(self):
        save_json(self.profiles_path, self.profiles)

    def save_security(self):
        save_json(self.security_path, self.security)

    def add_profile(self, name: str, samples: List[np.ndarray]):
        centroid = np.mean(np.stack(samples), axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        self.profiles[name] = centroid.tolist()
        self.save_profiles()

    def recognize(self, emb: np.ndarray, threshold: float) -> Tuple[str, float]:
        if not self.profiles:
            return "NoLocalData", 999.0

        best_name = "Unknown"
        best_dist = 999.0
        for name, ref in self.profiles.items():
            ref_vec = np.array(ref, dtype=np.float32)
            d = cosine_distance(emb, ref_vec)
            if d < best_dist:
                best_dist = d
                best_name = name

        if best_dist > threshold:
            return "Unknown", best_dist
        return best_name, best_dist

    def set_blink_password(self, pattern: str):
        salt, digest = salted_hash(pattern)
        self.security["blink_salt"] = salt
        self.security["blink_hash"] = digest
        self.save_security()

    def verify_blink_password(self, pattern: str) -> bool:
        salt = self.security.get("blink_salt")
        digest = self.security.get("blink_hash")
        if not salt or not digest:
            return False
        return verify_salted_hash(pattern, salt, digest)

    def has_blink_password(self) -> bool:
        return "blink_salt" in self.security and "blink_hash" in self.security


class FederatedClient:
    def __init__(self, args):
        self.args = args
        self.client_id = args.client_id
        self.store = LocalStore(Path(args.data_dir))

        self.recognition_threshold = 0.33
        self.ear_blink_threshold = 0.21
        self.server_model_version = 1
        self.last_federated_push = 0.0

        self.recognition_mode = False
        self.latest_name = "Unknown"
        self.latest_dist = 999.0
        self.rapid_eye_closure = False

        self.registering_name: Optional[str] = None
        self.registration_samples: List[np.ndarray] = []
        self.registration_target = 24

        self.eye_closed = False
        self.last_blink_ts = 0.0
        self.blink_timestamps: List[float] = []
        self.rapid_window = deque(maxlen=20)

        self.pattern_capture_mode: Optional[str] = None  # set | verify
        self.pattern_capture_start = 0.0
        self.pattern_capture_timestamps: List[float] = []
        self.pattern_window_seconds = 10.0
        self.pending_file_action: Optional[Tuple[str, Path]] = None
        self.last_message = "Ready"

        self.frame_brightness: deque = deque(maxlen=90)

    def register_to_server(self):
        try:
            resp = send_message(
                self.args.host,
                self.args.port,
                {"action": "register_client", "client_id": self.client_id},
            )
            if resp.get("status") == "ok":
                gm = resp.get("global_model", {})
                self.recognition_threshold = float(gm.get("recognition_threshold", self.recognition_threshold))
                self.ear_blink_threshold = float(gm.get("ear_blink_threshold", self.ear_blink_threshold))
                self.server_model_version = int(gm.get("version", self.server_model_version))
                print(f"[Client:{self.client_id}] Registered. Server model v{self.server_model_version}")
            else:
                print(f"[Client:{self.client_id}] Register failed: {resp}")
        except Exception as exc:
            print(f"[Client:{self.client_id}] Server unavailable ({exc}). Running local-only mode.")

    def push_federated_update(self):
        now = time.time()
        if now - self.last_federated_push < 15.0:
            return

        if self.frame_brightness:
            mean_b = float(np.mean(self.frame_brightness))
            # Tiny adaptation proposal from local environment.
            local_rec = max(0.22, min(0.45, 0.33 + ((130.0 - mean_b) / 1000.0)))
        else:
            local_rec = self.recognition_threshold

        local_ear = max(0.16, min(0.28, self.ear_blink_threshold))

        payload = {
            "action": "submit_update",
            "client_id": self.client_id,
            "update": {
                "recognition_threshold": local_rec,
                "ear_blink_threshold": local_ear,
            },
        }

        try:
            resp = send_message(self.args.host, self.args.port, payload)
            if resp.get("status") == "ok":
                gm = resp["global_model"]
                self.recognition_threshold = float(gm["recognition_threshold"])
                self.ear_blink_threshold = float(gm["ear_blink_threshold"])
                self.server_model_version = int(gm["version"])
        except Exception:
            pass
        self.last_federated_push = now

    def start_registration(self):
        name = input("Enter local identity name for registration: ").strip()
        if not name:
            print("Name cannot be empty.")
            return
        self.registering_name = name
        self.registration_samples = []
        self.last_message = f"Registering {name}... keep face centered"
        print(f"[Client:{self.client_id}] Registration started for '{name}'")

    def start_pattern_capture(self, mode: str):
        self.pattern_capture_mode = mode
        self.pattern_capture_start = time.time()
        self.pattern_capture_timestamps = []
        if mode == "set":
            self.last_message = "Blink password capture started (10s)"
        else:
            self.last_message = "Blink verification started (10s)"

    def handle_blink_event(self, now_ts: float):
        self.blink_timestamps.append(now_ts)
        self.rapid_window.append(now_ts)

        while self.rapid_window and (now_ts - self.rapid_window[0] > 2.0):
            self.rapid_window.popleft()
        self.rapid_eye_closure = len(self.rapid_window) >= 3

        if self.pattern_capture_mode and (now_ts - self.pattern_capture_start) <= self.pattern_window_seconds:
            self.pattern_capture_timestamps.append(now_ts)

    def finalize_pattern_capture_if_needed(self):
        if not self.pattern_capture_mode:
            return

        elapsed = time.time() - self.pattern_capture_start
        if elapsed < self.pattern_window_seconds:
            return

        pattern = blink_pattern_from_timestamps(
            self.pattern_capture_timestamps,
            window_seconds=self.pattern_window_seconds,
            buckets=5,
        )

        mode = self.pattern_capture_mode
        self.pattern_capture_mode = None

        if mode == "set":
            self.store.set_blink_password(pattern)
            self.last_message = f"Blink password set. Pattern signature: {pattern}"
            print(f"[Client:{self.client_id}] Blink password set.")
            return

        is_valid = self.store.verify_blink_password(pattern)
        if is_valid:
            self.last_message = "Blink verification passed"
            print(f"[Client:{self.client_id}] Blink verification passed.")
            if self.pending_file_action:
                action, path = self.pending_file_action
                self.pending_file_action = None
                self._run_file_action(action, path, pattern)
        else:
            self.last_message = "Blink verification failed"
            self.pending_file_action = None
            print(f"[Client:{self.client_id}] Blink verification failed.")

    def _run_file_action(self, action: str, path: Path, pattern_secret: str):
        try:
            if action == "lock":
                out = lock_file_with_secret(path, pattern_secret)
                self.last_message = f"File locked: {out.name}"
                print(f"[Client:{self.client_id}] Locked: {out}")
            elif action == "unlock":
                out = unlock_file_with_secret(path, pattern_secret)
                self.last_message = f"File unlocked: {out.name}"
                print(f"[Client:{self.client_id}] Unlocked: {out}")
        except Exception as exc:
            self.last_message = f"File action failed: {exc}"
            print(f"[Client:{self.client_id}] File action failed: {exc}")

    def run(self):
        self.register_to_server()

        cap = cv2.VideoCapture(self.args.camera, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Unable to open camera.")
            sys.exit(1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
        cap.set(cv2.CAP_PROP_FPS, 30)

        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        win_name = f"FederatedClient-{self.client_id}"
        print(
            "Controls: [r] register  [s] recognize  [b] set blink pass  [v] verify blink  "
            "[l] lock file  [u] unlock file  [q] quit"
        )

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            frame = enhance_frame(frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.frame_brightness.append(float(np.mean(gray)))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            now_ts = time.time()
            face_found = False

            if results.multi_face_landmarks:
                face_found = True
                lms = results.multi_face_landmarks[0].landmark
                h, w = frame.shape[:2]

                emb = build_face_embedding(lms, w, h)

                left_ear = compute_ear(lms, w, h, LEFT_EYE_IDX)
                right_ear = compute_ear(lms, w, h, RIGHT_EYE_IDX)
                ear = (left_ear + right_ear) / 2.0

                if ear < self.ear_blink_threshold and not self.eye_closed:
                    self.eye_closed = True
                elif ear >= self.ear_blink_threshold and self.eye_closed:
                    self.eye_closed = False
                    if now_ts - self.last_blink_ts > 0.12:
                        self.last_blink_ts = now_ts
                        self.handle_blink_event(now_ts)

                if self.registering_name:
                    self.registration_samples.append(emb)
                    if len(self.registration_samples) >= self.registration_target:
                        self.store.add_profile(self.registering_name, self.registration_samples)
                        self.last_message = f"Registered local identity: {self.registering_name}"
                        print(f"[Client:{self.client_id}] Saved local profile '{self.registering_name}'")
                        self.registering_name = None
                        self.registration_samples = []

                if self.recognition_mode:
                    self.latest_name, self.latest_dist = self.store.recognize(emb, self.recognition_threshold)

                cv2.putText(frame, f"EAR: {ear:.3f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 230, 60), 2)

            if not face_found and self.registering_name:
                cv2.putText(
                    frame,
                    "No face detected for registration",
                    (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            self.finalize_pattern_capture_if_needed()
            self.push_federated_update()

            rec_text = f"Recognized: {self.latest_name} ({self.latest_dist:.3f})" if self.recognition_mode else "Recognition: OFF"
            rec_color = (20, 220, 20) if self.latest_name not in ("Unknown", "NoLocalData") else (30, 170, 255)

            cv2.putText(frame, f"Client: {self.client_id}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"ServerModel: v{self.server_model_version}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 180), 2)
            cv2.putText(frame, rec_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rec_color, 2)
            cv2.putText(
                frame,
                f"RapidEyeClosure: {'YES' if self.rapid_eye_closure else 'NO'}",
                (10, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255) if self.rapid_eye_closure else (120, 255, 120),
                2,
            )

            if self.pattern_capture_mode:
                remaining = max(0.0, self.pattern_window_seconds - (time.time() - self.pattern_capture_start))
                cv2.putText(
                    frame,
                    f"Blink capture ({self.pattern_capture_mode}) {remaining:.1f}s",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (60, 220, 255),
                    2,
                )

            cv2.putText(frame, self.last_message[:90], (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2)

            cv2.imshow(win_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("r"):
                self.start_registration()
            if key == ord("s"):
                self.recognition_mode = not self.recognition_mode
                self.last_message = f"Recognition {'ON' if self.recognition_mode else 'OFF'}"
            if key == ord("b"):
                self.start_pattern_capture("set")
            if key == ord("v"):
                if not self.store.has_blink_password():
                    self.last_message = "Set blink password first (press b)"
                else:
                    self.pending_file_action = None
                    self.start_pattern_capture("verify")
            if key == ord("l"):
                if not self.store.has_blink_password():
                    self.last_message = "Set blink password first (press b)"
                else:
                    path = Path(input("File path to lock: ").strip()).expanduser()
                    if not path.exists() or not path.is_file():
                        self.last_message = "Invalid file path"
                    else:
                        self.pending_file_action = ("lock", path)
                        self.start_pattern_capture("verify")
            if key == ord("u"):
                if not self.store.has_blink_password():
                    self.last_message = "Set blink password first (press b)"
                else:
                    path = Path(input(".lock file path to unlock: ").strip()).expanduser()
                    if not path.exists() or not path.is_file():
                        self.last_message = "Invalid lock file path"
                    else:
                        self.pending_file_action = ("unlock", path)
                        self.start_pattern_capture("verify")

        face_mesh.close()
        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="Federated client for local physiognomy + blink detection")
    p.add_argument("--client-id", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--data-dir", default="./data/client")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    return p.parse_args()


def main():
    args = parse_args()
    app = FederatedClient(args)
    app.run()


if __name__ == "__main__":
    main()