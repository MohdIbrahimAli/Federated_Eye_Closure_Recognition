import argparse
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk

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

FL_PUSH_INTERVAL = 15.0  # seconds between federated updates


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

        # Global model parameters (received from server)
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

        self.pattern_capture_mode: Optional[str] = None
        self.pattern_capture_start = 0.0
        self.pattern_capture_timestamps: List[float] = []
        self.pattern_window_seconds = 10.0
        self.pending_file_action: Optional[Tuple[str, Path]] = None
        self.last_message = "Ready"
        self.server_state = "Connecting"
        self.current_ear = 0.0

        self.frame_brightness: deque = deque(maxlen=90)

        # ── Federated Learning local statistics ──────────────────────────────
        # Collected each interval, then sent to the server as a summary.
        # The server aggregates summaries from all clients via weighted FedAvg
        # and returns updated global model parameters.
        self._fl_reset_interval()
        self._fl_interval_start = time.time()

        # Last FL summary sent and response received (for GUI display)
        self.fl_last_summary: dict = {}
        self.fl_last_response: dict = {}

    def _fl_reset_interval(self):
        """Zero out local observation counters at the start of each FL round."""
        self._fl_frames_processed = 0
        self._fl_frames_with_face = 0
        self._fl_ear_open_sum = 0.0
        self._fl_ear_open_count = 0
        self._fl_blink_count = 0
        self._fl_recognition_attempts = 0
        self._fl_recognition_success = 0

    def record_frame(self, face_found: bool, ear: float, recognized: bool, attempted_recognition: bool):
        """Called once per frame from the GUI loop to accumulate local stats."""
        self._fl_frames_processed += 1
        if face_found:
            self._fl_frames_with_face += 1
            if ear >= self.ear_blink_threshold:
                # Eyes are open — record the open-eye EAR baseline
                self._fl_ear_open_sum += ear
                self._fl_ear_open_count += 1
        if attempted_recognition:
            self._fl_recognition_attempts += 1
            if recognized:
                self._fl_recognition_success += 1

    def set_status(self, message: str):
        self.last_message = message

    def register_to_server(self):
        try:
            resp = send_message(
                self.args.host,
                self.args.port,
                {"action": "register_client", "client_id": self.client_id},
            )
            if resp.get("status") == "ok":
                self._apply_global_model(resp.get("global_model", {}))
                self.server_state = "Connected"
                self.set_status(f"Connected to hub. Model v{self.server_model_version}")
            else:
                self.server_state = "Unavailable"
                self.set_status("Hub registration failed. Running in local-only mode.")
        except Exception:
            self.server_state = "Unavailable"
            self.set_status("Hub unavailable. Running in local-only mode.")

    def _apply_global_model(self, gm: dict):
        """Apply global model parameters received from the server."""
        self.recognition_threshold = float(gm.get("recognition_threshold", self.recognition_threshold))
        self.ear_blink_threshold = float(gm.get("ear_blink_threshold", self.ear_blink_threshold))
        self.server_model_version = int(gm.get("version", self.server_model_version))
        self.fl_last_response = gm

    def push_federated_update(self):
        """
        Build a local statistical summary and send it to the federated server.

        Summary fields:
          mean_ear_open        – average EAR when eyes were open this interval
          blink_rate           – blinks per minute observed locally
          mean_brightness      – mean frame brightness (0–255)
          face_detection_rate  – fraction of frames where a face was detected
          recognition_accuracy – fraction of recognition attempts that succeeded
          frames_processed     – total frames this interval (used as FedAvg weight)

        The server runs weighted FedAvg across all connected clients and derives:
          ear_blink_threshold  ← global_mean_ear_open × 0.72
          recognition_threshold ← calibrated from brightness + accuracy
        """
        now = time.time()
        if now - self.last_federated_push < FL_PUSH_INTERVAL:
            return

        interval_secs = now - self._fl_interval_start

        mean_ear_open = (
            self._fl_ear_open_sum / self._fl_ear_open_count
            if self._fl_ear_open_count > 0
            else 0.30
        )
        blink_rate = (self._fl_blink_count / interval_secs * 60.0) if interval_secs > 0 else 0.0
        mean_brightness = float(np.mean(self.frame_brightness)) if self.frame_brightness else 130.0
        face_detection_rate = (
            self._fl_frames_with_face / self._fl_frames_processed
            if self._fl_frames_processed > 0
            else 0.0
        )
        recognition_accuracy = (
            self._fl_recognition_success / self._fl_recognition_attempts
            if self._fl_recognition_attempts > 0
            else 0.0
        )

        summary = {
            "mean_ear_open": round(mean_ear_open, 5),
            "blink_rate": round(blink_rate, 3),
            "mean_brightness": round(mean_brightness, 2),
            "face_detection_rate": round(face_detection_rate, 4),
            "recognition_accuracy": round(recognition_accuracy, 4),
            "frames_processed": self._fl_frames_processed,
        }
        self.fl_last_summary = summary

        payload = {
            "action": "submit_update",
            "client_id": self.client_id,
            "update": summary,
        }

        try:
            resp = send_message(self.args.host, self.args.port, payload)
            if resp.get("status") == "ok":
                self._apply_global_model(resp["global_model"])
                self.server_state = "Connected"
                self.set_status(
                    f"FL round {self.server_model_version} complete. "
                    f"ear_thr={self.ear_blink_threshold:.3f}  rec_thr={self.recognition_threshold:.3f}"
                )
        except Exception:
            self.server_state = "Unavailable"

        self._fl_reset_interval()
        self._fl_interval_start = now
        self.last_federated_push = now

    def start_registration(self, name: str) -> bool:
        name = name.strip()
        if not name:
            return False
        self.registering_name = name
        self.registration_samples = []
        self.set_status(f"Registering {name}. Keep the face centered.")
        return True

    def toggle_recognition(self):
        self.recognition_mode = not self.recognition_mode
        self.set_status(f"Recognition {'enabled' if self.recognition_mode else 'disabled'}.")

    def start_pattern_capture(self, mode: str):
        self.pattern_capture_mode = mode
        self.pattern_capture_start = time.time()
        self.pattern_capture_timestamps = []
        if mode == "set":
            self.set_status("Blink password capture started for 10 seconds.")
        else:
            self.set_status("Blink verification started for 10 seconds.")

    def handle_blink_event(self, now_ts: float):
        self.blink_timestamps.append(now_ts)
        self.rapid_window.append(now_ts)
        self._fl_blink_count += 1  # contribute to FL blink_rate stat

        while self.rapid_window and (now_ts - self.rapid_window[0] > 2.0):
            self.rapid_window.popleft()
        self.rapid_eye_closure = len(self.rapid_window) >= 3

        if self.pattern_capture_mode and (now_ts - self.pattern_capture_start) <= self.pattern_window_seconds:
            self.pattern_capture_timestamps.append(now_ts)

    def finalize_pattern_capture_if_needed(self) -> Optional[Tuple[str, str]]:
        if not self.pattern_capture_mode:
            return None

        elapsed = time.time() - self.pattern_capture_start
        if elapsed < self.pattern_window_seconds:
            return None

        pattern = blink_pattern_from_timestamps(
            self.pattern_capture_timestamps,
            window_seconds=self.pattern_window_seconds,
            buckets=5,
        )

        mode = self.pattern_capture_mode
        self.pattern_capture_mode = None

        if mode == "set":
            self.store.set_blink_password(pattern)
            self.set_status(f"Blink password saved. Pattern signature: {pattern}")
            return ("info", "Blink password saved successfully.")

        is_valid = self.store.verify_blink_password(pattern)
        if not is_valid:
            self.pending_file_action = None
            self.set_status("Blink verification failed.")
            return ("error", "Blink verification failed.")

        self.set_status("Blink verification passed.")
        if self.pending_file_action:
            action, path = self.pending_file_action
            self.pending_file_action = None
            return self._run_file_action(action, path, pattern)
        return ("info", "Blink verification passed.")

    def queue_lock(self, path: Path):
        self.pending_file_action = ("lock", path)
        self.start_pattern_capture("verify")

    def queue_unlock(self, path: Path):
        self.pending_file_action = ("unlock", path)
        self.start_pattern_capture("verify")

    def _run_file_action(self, action: str, path: Path, pattern_secret: str) -> Tuple[str, str]:
        try:
            if action == "lock":
                out = lock_file_with_secret(path, pattern_secret)
                self.set_status(f"File locked: {out.name}")
                return ("info", f"File locked successfully:\n{out}")
            if action == "unlock":
                out = unlock_file_with_secret(path, pattern_secret)
                self.set_status(f"File unlocked: {out.name}")
                return ("info", f"File unlocked successfully:\n{out}")
            return ("error", f"Unknown file action: {action}")
        except Exception as exc:
            self.set_status(f"File action failed: {exc}")
            return ("error", f"File action failed:\n{exc}")


class FederatedClientGUI:
    def __init__(self, client: FederatedClient):
        self.client = client
        self.cap: Optional[cv2.VideoCapture] = None
        self.face_mesh = None
        self.closed = False
        self.video_image = None
        self.dialog_cooldown_until = 0.0

        self.root = tk.Tk()
        self.root.title(f"Federated Client - {self.client.client_id}")
        self.root.geometry("1280x900")
        self.root.configure(bg="#f4f6fb")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.status_var = tk.StringVar(value="Starting client...")
        self.server_var = tk.StringVar(value="Hub: connecting")
        self.model_var = tk.StringVar(value="Model v1")
        self.recognition_var = tk.StringVar(value="Recognition: OFF")
        self.identity_var = tk.StringVar(value="Recognized: Unknown")
        self.eye_var = tk.StringVar(value="EAR: 0.000")
        self.rapid_var = tk.StringVar(value="Rapid eye closure: NO")
        self.capture_var = tk.StringVar(value="Blink capture: idle")
        self.profiles_var = tk.StringVar(value=self._profiles_text())

        # Federated Learning activity display
        self.fl_sent_var = tk.StringVar(value="Sent summary: (waiting for first push)")
        self.fl_recv_var = tk.StringVar(value="Global model: (not yet received)")
        self.fl_countdown_var = tk.StringVar(value="Next FL push in: --s")

        self._build_layout()

    def _build_layout(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("Card.TFrame", background="#ffffff")
        style.configure("Panel.TLabelframe", background="#ffffff")
        style.configure("Panel.TLabelframe.Label", background="#ffffff", foreground="#1c274c")
        style.configure("Info.TLabel", background="#ffffff", foreground="#1f2d3d", font=("Segoe UI", 10))
        style.configure("FL.TLabel", background="#ffffff", foreground="#0d47a1", font=("Segoe UI", 9))
        style.configure("Title.TLabel", background="#f4f6fb", foreground="#14213d", font=("Segoe UI", 16, "bold"))

        container = ttk.Frame(self.root, padding=16)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=3)
        container.columnconfigure(1, weight=2)
        container.rowconfigure(1, weight=1)

        header = ttk.Label(
            container,
            text=f"Federated Learning Client Dashboard - {self.client.client_id}",
            style="Title.TLabel",
        )
        header.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

        video_card = ttk.Frame(container, style="Card.TFrame", padding=12)
        video_card.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        video_card.columnconfigure(0, weight=1)
        video_card.rowconfigure(0, weight=1)

        self.video_label = tk.Label(video_card, bg="#101827", bd=0, relief=tk.FLAT)
        self.video_label.grid(row=0, column=0, sticky="nsew")

        side = ttk.Frame(container)
        side.grid(row=1, column=1, sticky="nsew")
        side.columnconfigure(0, weight=1)

        # ── Live Summary ─────────────────────────────────────────────────────
        summary = ttk.LabelFrame(side, text="Live Summary", style="Panel.TLabelframe", padding=12)
        summary.grid(row=0, column=0, sticky="ew")

        for idx, variable in enumerate([
            self.server_var,
            self.model_var,
            self.recognition_var,
            self.identity_var,
            self.eye_var,
            self.rapid_var,
            self.capture_var,
        ]):
            ttk.Label(summary, textvariable=variable, style="Info.TLabel").grid(
                row=idx, column=0, sticky="w", pady=2
            )

        # ── Client Actions ────────────────────────────────────────────────────
        actions = ttk.LabelFrame(side, text="Client Actions", style="Panel.TLabelframe", padding=12)
        actions.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)

        ttk.Button(actions, text="Register Face", command=self.on_register).grid(
            row=0, column=0, sticky="ew", padx=(0, 6), pady=6
        )
        self.recognition_button = ttk.Button(
            actions, text="Enable Recognition", command=self.on_toggle_recognition
        )
        self.recognition_button.grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=6)
        ttk.Button(actions, text="Set Blink Password", command=self.on_set_blink_password).grid(
            row=1, column=0, sticky="ew", padx=(0, 6), pady=6
        )
        ttk.Button(actions, text="Verify Blink Password", command=self.on_verify_blink_password).grid(
            row=1, column=1, sticky="ew", padx=(6, 0), pady=6
        )
        ttk.Button(actions, text="Lock File", command=self.on_lock_file).grid(
            row=2, column=0, sticky="ew", padx=(0, 6), pady=6
        )
        ttk.Button(actions, text="Unlock File", command=self.on_unlock_file).grid(
            row=2, column=1, sticky="ew", padx=(6, 0), pady=6
        )

        # ── Federated Learning Activity ───────────────────────────────────────
        fl_frame = ttk.LabelFrame(
            side, text="Federated Learning Activity", style="Panel.TLabelframe", padding=12
        )
        fl_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        fl_frame.columnconfigure(0, weight=1)

        ttk.Label(fl_frame, textvariable=self.fl_countdown_var, style="FL.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 4)
        )
        ttk.Label(
            fl_frame,
            text="Last summary sent to server:",
            style="Info.TLabel",
            font=("Segoe UI", 9, "bold"),
        ).grid(row=1, column=0, sticky="w")
        ttk.Label(
            fl_frame, textvariable=self.fl_sent_var, style="FL.TLabel", wraplength=370, justify=tk.LEFT
        ).grid(row=2, column=0, sticky="w", pady=(0, 6))

        ttk.Label(
            fl_frame,
            text="Global model received from server:",
            style="Info.TLabel",
            font=("Segoe UI", 9, "bold"),
        ).grid(row=3, column=0, sticky="w")
        ttk.Label(
            fl_frame, textvariable=self.fl_recv_var, style="FL.TLabel", wraplength=370, justify=tk.LEFT
        ).grid(row=4, column=0, sticky="w")

        # ── Local Privacy Store ───────────────────────────────────────────────
        storage = ttk.LabelFrame(side, text="Local Privacy Store", style="Panel.TLabelframe", padding=12)
        storage.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(
            storage,
            text=f"Data folder: {Path(self.client.args.data_dir).resolve()}",
            style="Info.TLabel",
            wraplength=380,
        ).grid(row=0, column=0, sticky="w", pady=(0, 6))
        ttk.Label(
            storage, textvariable=self.profiles_var, style="Info.TLabel", wraplength=380, justify=tk.LEFT
        ).grid(row=1, column=0, sticky="w")

        # ── Status ────────────────────────────────────────────────────────────
        status = ttk.LabelFrame(side, text="Status", style="Panel.TLabelframe", padding=12)
        status.grid(row=4, column=0, sticky="nsew", pady=(12, 0))
        side.rowconfigure(4, weight=1)
        ttk.Label(
            status, textvariable=self.status_var, style="Info.TLabel", wraplength=380, justify=tk.LEFT
        ).grid(row=0, column=0, sticky="nw")

    def _profiles_text(self) -> str:
        names = sorted(self.client.store.profiles.keys())
        if not names:
            return "Registered identities: none"
        return "Registered identities: " + ", ".join(names)

    def _show_dialog(self, kind: str, message: str):
        now = time.time()
        if now < self.dialog_cooldown_until:
            return
        self.dialog_cooldown_until = now + 0.75
        if kind == "error":
            messagebox.showerror("Federated Client", message, parent=self.root)
        else:
            messagebox.showinfo("Federated Client", message, parent=self.root)

    def update_summary(self):
        self.status_var.set(self.client.last_message)
        self.server_var.set(f"Hub: {self.client.server_state}")
        self.model_var.set(
            f"Model v{self.client.server_model_version} | "
            f"face_thr {self.client.recognition_threshold:.3f} | "
            f"blink_thr {self.client.ear_blink_threshold:.3f}"
        )
        self.recognition_var.set(f"Recognition: {'ON' if self.client.recognition_mode else 'OFF'}")
        self.identity_var.set(f"Recognized: {self.client.latest_name} ({self.client.latest_dist:.3f})")
        self.eye_var.set(f"EAR: {self.client.current_ear:.3f}")
        self.rapid_var.set(f"Rapid eye closure: {'YES' if self.client.rapid_eye_closure else 'NO'}")

        if self.client.pattern_capture_mode:
            remaining = max(
                0.0,
                self.client.pattern_window_seconds - (time.time() - self.client.pattern_capture_start),
            )
            self.capture_var.set(f"Blink capture: {self.client.pattern_capture_mode} ({remaining:.1f}s)")
        else:
            self.capture_var.set("Blink capture: idle")

        self.profiles_var.set(self._profiles_text())
        self.recognition_button.configure(
            text="Disable Recognition" if self.client.recognition_mode else "Enable Recognition"
        )

        # FL countdown
        secs_since_push = time.time() - self.client.last_federated_push
        next_push = max(0.0, FL_PUSH_INTERVAL - secs_since_push)
        self.fl_countdown_var.set(f"Next FL push in: {next_push:.0f}s")

        # FL last summary sent
        s = self.client.fl_last_summary
        if s:
            self.fl_sent_var.set(
                f"ear_open={s.get('mean_ear_open', 0):.4f}  "
                f"blinks/min={s.get('blink_rate', 0):.1f}  "
                f"brightness={s.get('mean_brightness', 0):.1f}\n"
                f"face_rate={s.get('face_detection_rate', 0):.2f}  "
                f"rec_acc={s.get('recognition_accuracy', 0):.2f}  "
                f"frames={s.get('frames_processed', 0)}"
            )

        # FL last global model received
        g = self.client.fl_last_response
        if g:
            self.fl_recv_var.set(
                f"v{g.get('version', '?')}  "
                f"ear_thr={g.get('ear_blink_threshold', 0):.4f}  "
                f"rec_thr={g.get('recognition_threshold', 0):.4f}\n"
                f"global_ear_open={g.get('global_mean_ear_open', 0):.4f}  "
                f"blinks/min={g.get('global_blink_rate', 0):.1f}\n"
                f"brightness={g.get('global_brightness', 0):.1f}  "
                f"total_frames={g.get('total_frames_seen', 0)}"
            )

    def on_register(self):
        name = simpledialog.askstring("Register Face", "Enter a local identity name:", parent=self.root)
        if name is None:
            return
        if not self.client.start_registration(name):
            self._show_dialog("error", "Identity name cannot be empty.")
        self.update_summary()

    def on_toggle_recognition(self):
        self.client.toggle_recognition()
        self.update_summary()

    def on_set_blink_password(self):
        self.client.start_pattern_capture("set")
        self.update_summary()

    def on_verify_blink_password(self):
        if not self.client.store.has_blink_password():
            self._show_dialog("error", "Set a blink password first.")
            return
        self.client.pending_file_action = None
        self.client.start_pattern_capture("verify")
        self.update_summary()

    def on_lock_file(self):
        if not self.client.store.has_blink_password():
            self._show_dialog("error", "Set a blink password first.")
            return
        selected = filedialog.askopenfilename(title="Select a file to lock", parent=self.root)
        if not selected:
            return
        self.client.queue_lock(Path(selected))
        self.update_summary()

    def on_unlock_file(self):
        if not self.client.store.has_blink_password():
            self._show_dialog("error", "Set a blink password first.")
            return
        selected = filedialog.askopenfilename(
            title="Select a locked file",
            filetypes=[("Locked files", "*.lock"), ("All files", "*.*")],
            parent=self.root,
        )
        if not selected:
            return
        self.client.queue_unlock(Path(selected))
        self.update_summary()

    def on_close(self):
        self.closed = True
        if self.face_mesh is not None:
            self.face_mesh.close()
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

    def start(self):
        self.client.register_to_server()
        self.update_summary()

        self.cap = cv2.VideoCapture(self.client.args.camera, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self._show_dialog("error", "Unable to open the selected camera.")
            self.on_close()
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.client.args.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.client.args.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.root.after(0, self.update_frame)
        self.root.mainloop()

    def update_frame(self):
        if self.closed or self.cap is None or self.face_mesh is None:
            return

        ok, frame = self.cap.read()
        if not ok:
            self.client.set_status("Camera stream ended.")
            self.update_summary()
            self.root.after(60, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        frame = enhance_frame(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.client.frame_brightness.append(float(np.mean(gray)))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        now_ts = time.time()
        face_found = False
        ear = 0.0
        recognized = False
        attempted_recognition = False

        if results.multi_face_landmarks:
            face_found = True
            lms = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]

            emb = build_face_embedding(lms, w, h)

            left_ear = compute_ear(lms, w, h, LEFT_EYE_IDX)
            right_ear = compute_ear(lms, w, h, RIGHT_EYE_IDX)
            ear = (left_ear + right_ear) / 2.0
            self.client.current_ear = ear

            if ear < self.client.ear_blink_threshold and not self.client.eye_closed:
                self.client.eye_closed = True
            elif ear >= self.client.ear_blink_threshold and self.client.eye_closed:
                self.client.eye_closed = False
                if now_ts - self.client.last_blink_ts > 0.12:
                    self.client.last_blink_ts = now_ts
                    self.client.handle_blink_event(now_ts)

            if self.client.registering_name:
                self.client.registration_samples.append(emb)
                if len(self.client.registration_samples) >= self.client.registration_target:
                    self.client.store.add_profile(
                        self.client.registering_name, self.client.registration_samples
                    )
                    self.client.set_status(f"Registered local identity: {self.client.registering_name}")
                    self.client.registering_name = None
                    self.client.registration_samples = []

            if self.client.recognition_mode:
                attempted_recognition = True
                self.client.latest_name, self.client.latest_dist = self.client.store.recognize(
                    emb, self.client.recognition_threshold
                )
                recognized = self.client.latest_name not in ("Unknown", "NoLocalData", "RecognitionOff")
            else:
                self.client.latest_name = "RecognitionOff"
                self.client.latest_dist = 999.0

            cv2.putText(frame, f"EAR: {ear:.3f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 230, 60), 2)
            cv2.putText(
                frame, f"Client: {self.client.client_id}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            cv2.putText(
                frame,
                f"Recognized: {self.client.latest_name}",
                (12, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (90, 255, 140)
                if self.client.latest_name not in ("Unknown", "NoLocalData", "RecognitionOff")
                else (0, 215, 255),
                2,
            )
        else:
            self.client.current_ear = 0.0
            if self.client.registering_name:
                cv2.putText(
                    frame,
                    "No face detected for registration",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        # Accumulate local FL stats for this frame
        self.client.record_frame(face_found, ear, recognized, attempted_recognition)

        dialog = self.client.finalize_pattern_capture_if_needed()
        self.client.push_federated_update()
        self.update_summary()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        image = image.resize((820, 620))
        self.video_image = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self.video_image)

        if dialog:
            self._show_dialog(dialog[0], dialog[1])

        self.root.after(20, self.update_frame)


def parse_args():
    p = argparse.ArgumentParser(description="Federated client GUI for local physiognomy and blink detection")
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
    client = FederatedClient(args)
    gui = FederatedClientGUI(client)
    gui.start()


if __name__ == "__main__":
    main()
