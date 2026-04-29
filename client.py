import argparse
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog
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


# ── Premium colour palette ─────────────────────────────────────────────────
_BG       = "#f8fafc"
_CARD     = "#ffffff"
_BORDER   = "#e2e8f0"
_ACCENT   = "#2563eb"
_ACCENT_H = "#1d4ed8"
_TEXT1    = "#1e293b"
_TEXT2    = "#475569"
_MUTED    = "#94a3b8"
_SUC_BG   = "#f0fdf4"
_SUC_BR   = "#bbf7d0"
_SUC_TX   = "#15803d"
_SUC_DOT  = "#22c55e"
_WARN_BG  = "#fff7ed"
_WARN_BR  = "#fed7aa"
_WARN_TX  = "#c2410c"
_WARN_DOT = "#f97316"
_FL_BG    = "#eff6ff"
_FL_BR    = "#bfdbfe"
_FL_TX    = "#3b82f6"
_DARK_BAR = "#1e293b"
_AMBER    = "#d97706"
_FONT     = "Segoe UI"


class FederatedClientGUI:
    def __init__(self, client: FederatedClient):
        self.client = client
        self.cap: Optional[cv2.VideoCapture] = None
        self.face_mesh = None
        self.closed = False
        self.video_image = None
        self.dialog_cooldown_until = 0.0
        self._last_status = ""
        self._frame_count = 0

        self.root = tk.Tk()
        self.root.title(f"Federated Eye Closure Recognition — {self.client.client_id}")
        self.root.geometry("1280x820")
        self.root.configure(bg=_BG)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Top-bar StringVars
        self.hub_status_var      = tk.StringVar(value="Connecting...")
        self.ear_var             = tk.StringVar(value="—")
        self.blink_rate_var      = tk.StringVar(value="—")
        self.recognized_var      = tk.StringVar(value="—")
        self.rapid_var           = tk.StringVar(value="NO")

        # FL panel StringVars
        self.fl_countdown_var    = tk.StringVar(value="—")
        self.fl_ear_open_var     = tk.StringVar(value="—")
        self.fl_blink_rate_var   = tk.StringVar(value="—")
        self.fl_face_rate_var    = tk.StringVar(value="—")
        self.fl_ear_thr_var      = tk.StringVar(value="—")
        self.fl_rec_thr_var      = tk.StringVar(value="—")
        self.fl_frames_var       = tk.StringVar(value="—")

        # Status panel StringVars
        self.model_var           = tk.StringVar(value="v1  ·  face_thr 0.330  ·  blink_thr 0.210")
        self.blink_capture_var   = tk.StringVar(value="Idle")
        self.recognition_mode_var = tk.StringVar(value="OFF")
        self.brightness_var      = tk.StringVar(value="—")
        self.profiles_var        = tk.StringVar(value=self._profiles_text())

        # Bottom bar
        self.frame_count_var     = tk.StringVar(value="Frames: 0")

        # Label refs updated dynamically
        self._ear_label       = None
        self._recognized_label = None
        self._rapid_label     = None
        self._hub_pill        = None
        self._hub_dot         = None
        self._hub_label_w     = None
        self._recog_dot       = None
        self._sb_conn         = None

        self._build_layout()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_layout(self):
        self._build_topbar()
        tk.Frame(self.root, bg=_BORDER, height=2).pack(fill=tk.X)

        main = tk.Frame(self.root, bg=_BG)
        main.pack(fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(main, bg="#0f172a", bd=0, relief=tk.FLAT)
        self.video_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Frame(main, bg=_BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y)

        sidebar = tk.Frame(main, bg=_CARD, width=390)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)

        self._build_statusbar()

    def _build_topbar(self):
        bar = tk.Frame(self.root, bg=_CARD)
        bar.pack(fill=tk.X)

        inner = tk.Frame(bar, bg=_CARD)
        inner.pack(side=tk.LEFT, fill=tk.Y, padx=16)

        # Hub connection pill
        self._hub_pill = tk.Frame(
            inner, bg=_SUC_BG, padx=10, pady=4,
            highlightthickness=1, highlightbackground=_SUC_BR,
        )
        self._hub_pill.pack(side=tk.LEFT, padx=(0, 12), pady=10)

        self._hub_dot = tk.Label(self._hub_pill, text="●", fg=_SUC_DOT, bg=_SUC_BG,
                                  font=(_FONT, 9))
        self._hub_dot.pack(side=tk.LEFT, padx=(0, 4))

        self._hub_label_w = tk.Label(self._hub_pill, textvariable=self.hub_status_var,
                                      fg=_SUC_TX, bg=_SUC_BG, font=(_FONT, 11, "bold"))
        self._hub_label_w.pack(side=tk.LEFT)

        # Stat chips
        for chip_text, var, ref in [
            ("EAR",          self.ear_var,        "_ear_label"),
            ("BLINKS / MIN", self.blink_rate_var, None),
            ("RECOGNIZED",   self.recognized_var, "_recognized_label"),
            ("RAPID CLOSURE",self.rapid_var,      "_rapid_label"),
        ]:
            chip = tk.Frame(inner, bg=_BG, padx=12, pady=2,
                            highlightthickness=1, highlightbackground=_BORDER)
            chip.pack(side=tk.LEFT, padx=6, pady=10)
            tk.Label(chip, text=chip_text, fg=_MUTED, bg=_BG,
                     font=(_FONT, 8, "bold")).pack(anchor="w")
            val = tk.Label(chip, textvariable=var, fg=_ACCENT, bg=_BG,
                           font=(_FONT, 14, "bold"))
            val.pack(anchor="w")
            if ref:
                setattr(self, ref, val)

        # Vertical divider
        tk.Frame(inner, bg=_BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y, pady=10, padx=8)

        # ⚡ Actions dropdown button
        self.actions_btn = tk.Button(
            inner, text="  ⚡  Actions  ▾  ",
            bg=_ACCENT, fg="#ffffff",
            activebackground=_ACCENT_H, activeforeground="#ffffff",
            font=(_FONT, 12, "bold"), relief=tk.FLAT, bd=0,
            cursor="hand2", padx=4, pady=6,
            command=self._show_actions_menu,
        )
        self.actions_btn.pack(side=tk.LEFT, pady=10)

    def _show_actions_menu(self):
        rec_label = (
            "  👁   Disable Recognition"
            if self.client.recognition_mode
            else "  👁   Enable Recognition"
        )
        menu = tk.Menu(
            self.root, tearoff=0,
            bg=_CARD, fg=_TEXT1,
            activebackground=_FL_BG, activeforeground=_ACCENT,
            font=(_FONT, 11), relief=tk.FLAT, bd=1,
        )
        menu.add_command(label="  IDENTITY", state=tk.DISABLED,
                         font=(_FONT, 8, "bold"), foreground=_MUTED)
        menu.add_command(label="  👤   Register Face",   command=self.on_register)
        menu.add_command(label=rec_label,                command=self.on_toggle_recognition)
        menu.add_separator()
        menu.add_command(label="  BLINK PASSWORD", state=tk.DISABLED,
                         font=(_FONT, 8, "bold"), foreground=_MUTED)
        menu.add_command(label="  🔑   Set Blink Password",    command=self.on_set_blink_password)
        menu.add_command(label="  ✅   Verify Blink Password",  command=self.on_verify_blink_password)
        menu.add_separator()
        menu.add_command(label="  FILE ENCRYPTION", state=tk.DISABLED,
                         font=(_FONT, 8, "bold"), foreground=_MUTED)
        menu.add_command(label="  🔒   Lock File",   command=self.on_lock_file)
        menu.add_command(label="  🔓   Unlock File", command=self.on_unlock_file)

        btn = self.actions_btn
        menu.post(btn.winfo_rootx(), btn.winfo_rooty() + btn.winfo_height() + 2)

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _build_sidebar(self, parent):
        self._build_fl_panel(parent)
        tk.Frame(parent, bg=_BORDER, height=1).pack(fill=tk.X)
        self._build_status_panel(parent)
        tk.Frame(parent, bg=_BORDER, height=1).pack(fill=tk.X)
        self._build_log_panel(parent)

    def _build_fl_panel(self, parent):
        frame = tk.Frame(parent, bg=_FL_BG)
        frame.pack(fill=tk.X)

        tk.Label(frame, text="⚡  FEDERATED LEARNING", fg=_FL_TX, bg=_FL_BG,
                 font=(_FONT, 8, "bold")).pack(anchor="w", padx=16, pady=(12, 8))

        # Countdown card
        cd = tk.Frame(frame, bg=_CARD, highlightthickness=1, highlightbackground=_FL_BR)
        cd.pack(fill=tk.X, padx=16, pady=(0, 10))
        cd_row = tk.Frame(cd, bg=_CARD)
        cd_row.pack(fill=tk.X, padx=12, pady=8)
        tk.Label(cd_row, text="Next FL push in", fg=_TEXT2, bg=_CARD,
                 font=(_FONT, 10)).pack(side=tk.LEFT)
        tk.Label(cd_row, textvariable=self.fl_countdown_var, fg=_ACCENT, bg=_CARD,
                 font=(_FONT, 18, "bold")).pack(side=tk.RIGHT)

        # Data rows
        for label_text, var, color in [
            ("Last ear_open sent",  self.fl_ear_open_var,  _TEXT1),
            ("Blinks / min sent",   self.fl_blink_rate_var,_TEXT1),
            ("Face detection rate", self.fl_face_rate_var, _TEXT1),
            ("Global ear_thr ←",   self.fl_ear_thr_var,   _ACCENT),
            ("Global rec_thr ←",   self.fl_rec_thr_var,   _ACCENT),
            ("Frames this round",   self.fl_frames_var,    _TEXT1),
        ]:
            row = tk.Frame(frame, bg=_FL_BG)
            row.pack(fill=tk.X, padx=16, pady=1)
            tk.Label(row, text=label_text, fg=_TEXT2, bg=_FL_BG,
                     font=(_FONT, 10)).pack(side=tk.LEFT)
            tk.Label(row, textvariable=var, fg=color, bg=_FL_BG,
                     font=(_FONT, 10, "bold")).pack(side=tk.RIGHT)

        tk.Frame(frame, bg=_FL_BG, height=10).pack()

    def _build_status_panel(self, parent):
        frame = tk.Frame(parent, bg=_CARD)
        frame.pack(fill=tk.X)

        tk.Label(frame, text="LIVE STATUS", fg=_MUTED, bg=_CARD,
                 font=(_FONT, 8, "bold")).pack(anchor="w", padx=16, pady=(12, 6))

        # Model badge
        badge = tk.Frame(frame, bg=_BG, highlightthickness=1, highlightbackground=_BORDER)
        badge.pack(fill=tk.X, padx=16, pady=(0, 8))
        tk.Label(badge, textvariable=self.model_var, fg=_TEXT2, bg=_BG,
                 font=(_FONT, 10)).pack(anchor="w", padx=10, pady=6)

        # Status rows
        for dot_color, label_text, var, ref in [
            (_SUC_DOT, "Blink capture",    self.blink_capture_var,    None),
            (_MUTED,   "Recognition mode", self.recognition_mode_var, "_recog_dot"),
            (_TEXT2,   "Brightness",       self.brightness_var,       None),
        ]:
            row = tk.Frame(frame, bg=_CARD)
            row.pack(fill=tk.X, padx=16, pady=2)
            dot = tk.Label(row, text="●", fg=dot_color, bg=_CARD, font=(_FONT, 9))
            dot.pack(side=tk.LEFT, padx=(0, 6))
            if ref:
                setattr(self, ref, dot)
            tk.Label(row, text=label_text, fg=_TEXT2, bg=_CARD,
                     font=(_FONT, 11)).pack(side=tk.LEFT)
            tk.Label(row, textvariable=var, fg=_TEXT1, bg=_CARD,
                     font=(_FONT, 11, "bold")).pack(side=tk.RIGHT)

        tk.Label(frame, text="REGISTERED IDENTITIES", fg=_MUTED, bg=_CARD,
                 font=(_FONT, 8, "bold")).pack(anchor="w", padx=16, pady=(10, 4))
        tk.Label(frame, textvariable=self.profiles_var, fg=_TEXT2, bg=_CARD,
                 font=(_FONT, 10), wraplength=350, justify=tk.LEFT).pack(
                     anchor="w", padx=16, pady=(0, 12))

    def _build_log_panel(self, parent):
        frame = tk.Frame(parent, bg=_CARD)
        frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(frame, text="ACTIVITY LOG", fg=_MUTED, bg=_CARD,
                 font=(_FONT, 8, "bold")).pack(anchor="w", padx=16, pady=(12, 6))

        wrap = tk.Frame(frame, bg=_CARD)
        wrap.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(
            wrap, bg=_CARD, fg=_TEXT2, font=(_FONT, 10),
            relief=tk.FLAT, padx=16, state=tk.DISABLED,
            wrap=tk.WORD, selectbackground=_FL_BG, cursor="arrow",
        )
        sb = tk.Scrollbar(wrap, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.log_text.tag_configure("ts",   foreground=_MUTED)
        self.log_text.tag_configure("ok",   foreground="#22c55e")
        self.log_text.tag_configure("info", foreground=_ACCENT)
        self.log_text.tag_configure("warn", foreground=_AMBER)

    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg=_DARK_BAR, height=28)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        bar.pack_propagate(False)

        left = tk.Frame(bar, bg=_DARK_BAR)
        left.pack(side=tk.LEFT, padx=16)

        self._sb_conn = tk.Label(left, text="● Connecting", fg="#4ade80", bg=_DARK_BAR,
                                  font=(_FONT, 9))
        self._sb_conn.pack(side=tk.LEFT, pady=5)

        for txt in [" | ", f"{self.client.client_id} · {self.client.args.host}:{self.client.args.port}",
                    " | ", str(Path(self.client.args.data_dir).resolve())]:
            c = "#334155" if txt.strip() == "|" else "#64748b"
            tk.Label(left, text=txt, fg=c, bg=_DARK_BAR, font=(_FONT, 9)).pack(side=tk.LEFT)

        right = tk.Frame(bar, bg=_DARK_BAR)
        right.pack(side=tk.RIGHT, padx=16)
        tk.Label(right, textvariable=self.frame_count_var, fg="#64748b", bg=_DARK_BAR,
                 font=(_FONT, 9)).pack(side=tk.RIGHT, pady=5)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _profiles_text(self) -> str:
        names = sorted(self.client.store.profiles.keys())
        if not names:
            return "None registered"
        return "  ·  ".join(f"👤 {n}" for n in names)

    def _show_dialog(self, kind: str, message: str):
        now = time.time()
        if now < self.dialog_cooldown_until:
            return
        self.dialog_cooldown_until = now + 0.75
        if kind == "error":
            messagebox.showerror("Federated Client", message, parent=self.root)
        else:
            messagebox.showinfo("Federated Client", message, parent=self.root)

    def _append_log(self, message: str, level: str = "normal"):
        ts = time.strftime("%H:%M:%S")
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, ts + "  ", "ts")
        self.log_text.insert(tk.END, message + "\n", level)
        self.log_text.configure(state=tk.DISABLED)
        self.log_text.see(tk.END)

    # ── Update loop ───────────────────────────────────────────────────────────

    def update_summary(self):
        now = time.time()

        # Hub status pill
        connected = self.client.server_state == "Connected"
        if connected:
            self.hub_status_var.set("Hub Connected")
            pill_bg, dot_fg, lbl_fg, pill_br = _SUC_BG, _SUC_DOT, _SUC_TX, _SUC_BR
            if self._sb_conn:
                self._sb_conn.configure(text="● Connected", fg="#4ade80")
        else:
            self.hub_status_var.set("Hub Unavailable")
            pill_bg, dot_fg, lbl_fg, pill_br = _WARN_BG, _WARN_DOT, _WARN_TX, _WARN_BR
            if self._sb_conn:
                self._sb_conn.configure(text="● Unavailable", fg=_AMBER)

        if self._hub_pill:
            self._hub_pill.configure(bg=pill_bg, highlightbackground=pill_br)
        if self._hub_dot:
            self._hub_dot.configure(bg=pill_bg, fg=dot_fg)
        if self._hub_label_w:
            self._hub_label_w.configure(bg=pill_bg, fg=lbl_fg)

        # Stat chips
        ear = self.client.current_ear
        self.ear_var.set(f"{ear:.3f}" if ear > 0 else "—")

        recent_blinks = sum(1 for t in self.client.blink_timestamps if t > now - 60)
        self.blink_rate_var.set(str(recent_blinks))

        if self.client.recognition_mode:
            name = self.client.latest_name
            if name in ("Unknown", "NoLocalData", "RecognitionOff"):
                self.recognized_var.set("Unknown")
                if self._recognized_label:
                    self._recognized_label.configure(fg=_AMBER)
            else:
                self.recognized_var.set(name)
                if self._recognized_label:
                    self._recognized_label.configure(fg=_ACCENT)
        else:
            self.recognized_var.set("—")
            if self._recognized_label:
                self._recognized_label.configure(fg=_MUTED)

        rapid = self.client.rapid_eye_closure
        self.rapid_var.set("YES" if rapid else "NO")
        if self._rapid_label:
            self._rapid_label.configure(fg=_AMBER if rapid else _MUTED)

        # FL panel
        secs_since = now - self.client.last_federated_push
        self.fl_countdown_var.set(f"{max(0.0, FL_PUSH_INTERVAL - secs_since):.0f}s")

        s = self.client.fl_last_summary
        if s:
            self.fl_ear_open_var.set(f"{s.get('mean_ear_open', 0):.4f}")
            self.fl_blink_rate_var.set(f"{s.get('blink_rate', 0):.1f}")
            self.fl_face_rate_var.set(f"{s.get('face_detection_rate', 0):.2f}")
            self.fl_frames_var.set(str(s.get('frames_processed', 0)))

        g = self.client.fl_last_response
        if g:
            self.fl_ear_thr_var.set(f"{g.get('ear_blink_threshold', 0):.4f}")
            self.fl_rec_thr_var.set(f"{g.get('recognition_threshold', 0):.4f}")

        # Status panel
        gv = self.client.server_model_version
        ft = self.client.recognition_threshold
        bt = self.client.ear_blink_threshold
        self.model_var.set(f"v{gv}   ·   face_thr {ft:.3f}   ·   blink_thr {bt:.3f}")

        if self.client.pattern_capture_mode:
            rem = max(0.0, self.client.pattern_window_seconds - (now - self.client.pattern_capture_start))
            self.blink_capture_var.set(f"{self.client.pattern_capture_mode} ({rem:.1f}s)")
        else:
            self.blink_capture_var.set("Idle")

        mode_on = self.client.recognition_mode
        self.recognition_mode_var.set("ON" if mode_on else "OFF")
        if self._recog_dot:
            self._recog_dot.configure(fg=_ACCENT if mode_on else _MUTED)

        brightness = float(np.mean(self.client.frame_brightness)) if self.client.frame_brightness else 0.0
        self.brightness_var.set(f"{brightness:.1f}")
        self.profiles_var.set(self._profiles_text())

        # Frame counter
        self.frame_count_var.set(f"Frames: {self._frame_count:,}")

        # Activity log — append only when message changes
        msg = self.client.last_message
        if msg != self._last_status:
            self._last_status = msg
            lvl = ("ok"   if "complete" in msg.lower() or "success" in msg.lower() else
                   "info" if any(k in msg.lower() for k in ("connect", "register", "enabled", "disabled")) else
                   "warn" if "fail" in msg.lower() or "error" in msg.lower() else
                   "normal")
            self._append_log(msg, lvl)

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

        self._frame_count += 1

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        vw = self.video_label.winfo_width() or 890
        vh = self.video_label.winfo_height() or 660
        image = image.resize((vw, vh))
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
