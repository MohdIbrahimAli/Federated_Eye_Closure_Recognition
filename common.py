import base64
import hashlib
import hmac
import json
import os
import socket
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np
from cryptography.fernet import Fernet
import tensorflow as tf
from tensorflow.keras import layers, models

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]


def create_face_cnn_model(input_shape=(128, 128, 3), embedding_dim=128):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(embedding_dim, activation='linear'),  # Embedding layer
    ])
    return model


def build_face_embedding_cnn(face_image: np.ndarray, model: tf.keras.Model) -> np.ndarray:
    # Preprocess image: resize to 128x128, normalize
    face_resized = cv2.resize(face_image, (128, 128))
    face_normalized = face_resized / 255.0
    face_input = np.expand_dims(face_normalized, axis=0)
    embedding = model.predict(face_input, verbose=0)[0]
    # Normalize embedding
    embedding /= np.linalg.norm(embedding) + 1e-9
    return embedding


def crop_face_from_landmarks(frame: np.ndarray, landmarks) -> np.ndarray:
    h, w = frame.shape[:2]
    x_coords = [lm.x * w for lm in landmarks]
    y_coords = [lm.y * h for lm in landmarks]
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    # Add padding
    padding = int(0.1 * (x_max - x_min))
    x_min = max(0, x_min - padding)
    x_max = min(w, x_max + padding)
    y_min = max(0, y_min - padding)
    y_max = min(h, y_max + padding)
    face_crop = frame[y_min:y_max, x_min:x_max]
    return face_crop


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def send_message(host: str, port: int, payload: Dict[str, Any], timeout: float = 5.0) -> Dict[str, Any]:
    data = (json.dumps(payload) + "\n").encode("utf-8")
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.sendall(data)
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
    if not buf:
        return {"status": "error", "message": "No response"}
    return json.loads(buf.decode("utf-8").strip())


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a) + 1e-9
    b_norm = np.linalg.norm(b) + 1e-9
    return 1.0 - float(np.dot(a, b) / (a_norm * b_norm))


def _point(landmarks, idx: int, w: int, h: int) -> np.ndarray:
    p = landmarks[idx]
    return np.array([p.x * w, p.y * h, p.z * w], dtype=np.float32)




def _euclid(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


def compute_ear(landmarks, width: int, height: int, eye_idx: Iterable[int]) -> float:
    points = [_point(landmarks, idx, width, height) for idx in eye_idx]
    p1, p2, p3, p4, p5, p6 = points
    vertical = _euclid(p2, p6) + _euclid(p3, p5)
    horizontal = 2.0 * _euclid(p1, p4) + 1e-6
    return vertical / horizontal


def enhance_frame(frame: np.ndarray) -> np.ndarray:
    # Balance dynamic range for low-light and over-exposed scenes.
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l = clahe.apply(l)

    merged = cv2.merge((l, a, b))
    adjusted = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    mean_v = float(np.mean(gray))
    target = 125.0
    gamma = max(0.6, min(1.8, np.log(target / 255.0 + 1e-9) / np.log((mean_v / 255.0) + 1e-9)))

    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    gamma_fixed = cv2.LUT(adjusted, lut)

    hsv = cv2.cvtColor(gamma_fixed, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v.astype(np.int16) - np.maximum(v.astype(np.int16) - 240, 0) // 2, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

    return cv2.GaussianBlur(out, (3, 3), 0)


def blink_pattern_from_timestamps(timestamps: List[float], window_seconds: float = 10.0, buckets: int = 5) -> str:
    if not timestamps:
        return "0" * buckets
    t0 = timestamps[0]
    rel = [t - t0 for t in timestamps if 0.0 <= (t - t0) <= window_seconds]
    counts = [0] * buckets
    for t in rel:
        idx = min(int((t / window_seconds) * buckets), buckets - 1)
        counts[idx] += 1
    counts = [min(c, 9) for c in counts]
    return "".join(str(c) for c in counts)


def salted_hash(text: str, salt: bytes | None = None) -> Tuple[str, str]:
    s = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", text.encode("utf-8"), s, 120_000)
    return base64.b64encode(s).decode("utf-8"), base64.b64encode(digest).decode("utf-8")


def verify_salted_hash(text: str, salt_b64: str, hash_b64: str) -> bool:
    salt = base64.b64decode(salt_b64)
    _, candidate = salted_hash(text, salt=salt)
    return hmac.compare_digest(candidate, hash_b64)


def _derive_fernet_key(secret: str, salt: bytes) -> bytes:
    raw = hashlib.pbkdf2_hmac("sha256", secret.encode("utf-8"), salt, 200_000, dklen=32)
    return base64.urlsafe_b64encode(raw)


def lock_file_with_secret(path: Path, secret: str) -> Path:
    data = path.read_bytes()
    salt = os.urandom(16)
    key = _derive_fernet_key(secret, salt)
    token = Fernet(key).encrypt(data)

    out_path = path.with_suffix(path.suffix + ".lock")
    out_path.write_bytes(token)

    meta = {
        "salt": base64.b64encode(salt).decode("utf-8"),
        "original_name": path.name,
    }
    save_json(out_path.with_suffix(out_path.suffix + ".meta.json"), meta)
    return out_path


def unlock_file_with_secret(path: Path, secret: str) -> Path:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta = load_json(meta_path, None)
    if not meta:
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")

    salt = base64.b64decode(meta["salt"])
    key = _derive_fernet_key(secret, salt)
    token = path.read_bytes()
    data = Fernet(key).decrypt(token)

    original_name = meta.get("original_name") or path.stem
    out_path = path.parent / f"unlocked_{original_name}"
    out_path.write_bytes(data)
    return out_path