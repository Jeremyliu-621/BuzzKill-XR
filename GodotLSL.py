#!/usr/bin/env python3
"""Muse LSL -> Godot signal processing bridge.

- Reads EEG from an LSL stream (Muse).
- Computes band power features on a sliding window.
- Sends JSON packets via UDP to Godot.

Dependencies:
  pip install pylsl numpy
Optional (for better filtering):
  pip install scipy
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
import tempfile
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

print("running godotlsl")

try:
    from pylsl import StreamInlet
    try:
        from pylsl import resolve_byprop as _resolve_byprop
    except Exception:
        _resolve_byprop = None
    try:
        from pylsl import resolve_stream as _resolve_stream
    except Exception:
        _resolve_stream = None
    try:
        from pylsl import resolve_streams as _resolve_streams
    except Exception:
        _resolve_streams = None
except Exception as exc:  # pragma: no cover
    print("ERROR: pylsl is required. Install with `pip install pylsl`.")
    raise

try:
    from scipy.signal import butter, filtfilt

    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

try:
    import tkinter as tk
    from tkinter.scrolledtext import ScrolledText

    HAVE_TK = True
except Exception:
    HAVE_TK = False


BANDS_HZ = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


@dataclass
class Config:
    stream_name_contains: str
    stream_type: str
    max_resolve_seconds: float
    udp_host: str
    udp_port: int
    window_seconds: float
    step_seconds: float
    lsl_timeout: float
    detrend: bool
    use_filter: bool
    verbose: bool


class UDPJsonSender:
    def __init__(self, host: str, port: int) -> None:
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, payload: Dict) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.sock.sendto(data, self.addr)


class TextWindow:
    def __init__(self, title: str = "Muse Band Power") -> None:
        if not HAVE_TK:
            raise RuntimeError("tkinter is not available in this Python build.")
        self.root = tk.Tk()
        self.root.title(title)
        self.text = ScrolledText(self.root, width=64, height=20, state="disabled")
        self.text.pack(fill="both", expand=True)
        self.root.update_idletasks()

    def append_line(self, line: str) -> None:
        self.text.configure(state="normal")
        self.text.insert("end", line + "\n")
        self.text.see("end")
        self.text.configure(state="disabled")
        self.root.update()


class SlidingWindow:
    def __init__(self, maxlen: int, n_channels: int) -> None:
        self.n_channels = n_channels
        self.buffers: List[Deque[float]] = [deque(maxlen=maxlen) for _ in range(n_channels)]
        self.timestamps: Deque[float] = deque(maxlen=maxlen)

    def push(self, sample: List[float], ts: float) -> None:
        for i in range(self.n_channels):
            self.buffers[i].append(sample[i])
        self.timestamps.append(ts)

    def is_full(self) -> bool:
        return len(self.timestamps) == self.timestamps.maxlen

    def to_array(self) -> np.ndarray:
        return np.vstack([np.array(buf) for buf in self.buffers])

    def last_timestamp(self) -> float:
        return self.timestamps[-1] if self.timestamps else 0.0


def _resolve_by_type(stream_type: str, timeout: float):
    if _resolve_byprop is not None:
        return _resolve_byprop("type", stream_type, timeout=timeout)
    if _resolve_stream is not None:
        return _resolve_stream("type", stream_type, timeout=timeout)
    if _resolve_streams is not None:
        # resolve_streams returns all visible streams; filter by type below
        return _resolve_streams(timeout=timeout)
    raise RuntimeError("No LSL resolve function available in pylsl.")


def resolve_lsl_stream(cfg: Config):
    start = time.time()
    while time.time() - start < cfg.max_resolve_seconds:
        streams = _resolve_by_type(cfg.stream_type, timeout=1.0)
        for s in streams:
            if s.type().lower() != cfg.stream_type.lower():
                continue
            if cfg.stream_name_contains.lower() in s.name().lower():
                return s
    return None


def bandpower_fft(signal: np.ndarray, sf: float, band: Tuple[float, float]) -> float:
    # signal is 1D
    n = len(signal)
    if n == 0:
        return 0.0
    freqs = np.fft.rfftfreq(n, d=1.0 / sf)
    fft = np.fft.rfft(signal * np.hanning(n))
    psd = (np.abs(fft) ** 2) / n
    low, high = band
    idx = np.logical_and(freqs >= low, freqs <= high)
    if not np.any(idx):
        return 0.0
    return float(np.trapezoid(psd[idx], freqs[idx]))


def bandpass_filter(data: np.ndarray, sf: float, band: Tuple[float, float]) -> np.ndarray:
    if not HAVE_SCIPY:
        return data
    low, high = band
    nyq = 0.5 * sf
    lowc = low / nyq
    highc = min(high / nyq, 0.99)
    b, a = butter(4, [lowc, highc], btype="band")
    return filtfilt(b, a, data)


def compute_features(window: np.ndarray, sf: float, cfg: Config) -> Dict:
    # window shape: channels x samples
    if cfg.detrend:
        window = window - np.mean(window, axis=1, keepdims=True)

    features: Dict[str, List[float]] = {}
    for band_name, band in BANDS_HZ.items():
        values = []
        for ch in range(window.shape[0]):
            sig = window[ch]
            if cfg.use_filter:
                sig = bandpass_filter(sig, sf, band)
            values.append(bandpower_fft(sig, sf, band))
        features[band_name] = values

    return features


def estimate_sample_rate(timestamps: Deque[float]) -> float:
    if len(timestamps) < 2:
        return 0.0
    duration = timestamps[-1] - timestamps[0]
    if duration <= 0.0:
        return 0.0
    return (len(timestamps) - 1) / duration


def main() -> int:
    parser = argparse.ArgumentParser(description="Muse LSL to Godot signal processing bridge")
    parser.add_argument("--name-contains", default="Muse", help="LSL stream name substring")
    parser.add_argument("--type", default="EEG", dest="stream_type", help="LSL stream type")
    parser.add_argument("--resolve-seconds", type=float, default=15.0, help="time to resolve LSL stream")
    parser.add_argument("--udp-host", default="127.0.0.1", help="Godot UDP host")
    parser.add_argument("--udp-port", type=int, default=12000, help="Godot UDP port")
    parser.add_argument("--window", type=float, default=2.0, help="window size in seconds")
    parser.add_argument("--step", type=float, default=0.25, help="step size in seconds")
    parser.add_argument("--lsl-timeout", type=float, default=1.0, help="LSL pull timeout")
    parser.add_argument("--detrend", action="store_true", help="remove mean from window")
    parser.add_argument("--filter", action="store_true", help="apply bandpass filter before FFT")
    parser.add_argument("--verbose", action="store_true", help="verbose logging")
    parser.add_argument("--text", action="store_true", help="print band-power values periodically")
    parser.add_argument("--text-interval", type=float, default=10.0, help="seconds between text prints")
    parser.add_argument("--text-window", action="store_true", help="show a simple text window")
    parser.add_argument("--json", action="store_true", help="enable JSON file output")
    parser.add_argument("--json-file", default="muse_latest.json", help="path for JSON output file")
    parser.add_argument("--json-interval", type=float, default=5.0, help="seconds between JSON file writes")
    parser.add_argument("--raw-send", action="store_true", help="send individual feature values as plain text to 127.0.0.1:5005")
    args = parser.parse_args()

    cfg = Config(
        stream_name_contains=args.name_contains,
        stream_type=args.stream_type,
        max_resolve_seconds=args.resolve_seconds,
        udp_host=args.udp_host,
        udp_port=args.udp_port,
        window_seconds=args.window,
        step_seconds=args.step,
        lsl_timeout=args.lsl_timeout,
        detrend=args.detrend,
        use_filter=args.filter,
        verbose=args.verbose,
    )

    stream = resolve_lsl_stream(cfg)
    if stream is None:
        print("ERROR: No LSL stream found. Is Muse streaming to LSL?")
        return 1

    inlet = StreamInlet(stream)
    info = inlet.info()
    n_channels = info.channel_count()
    nominal_srate = float(info.nominal_srate())

    if cfg.verbose:
        print(f"Connected to LSL stream: {info.name()} ({n_channels} ch @ {nominal_srate} Hz)")

    # Compute buffer sizes based on nominal sample rate
    samples_per_window = max(8, int(round(cfg.window_seconds * nominal_srate)))
    samples_per_step = max(1, int(round(cfg.step_seconds * nominal_srate)))

    window = SlidingWindow(samples_per_window, n_channels)
    sender = UDPJsonSender(cfg.udp_host, cfg.udp_port)
    text_window = None
    if args.text_window:
        text_window = TextWindow()
    last_text_ts = 0.0
    last_json_ts = 0.0

    samples_since_last = 0
    last_send_ts = 0.0

    while True:
        sample, ts = inlet.pull_sample(timeout=cfg.lsl_timeout)
        if sample is None:
            continue

        window.push(sample, ts)
        samples_since_last += 1

        if not window.is_full():
            continue

        if samples_since_last < samples_per_step:
            continue

        samples_since_last = 0

        # Use estimated sample rate for robustness
        sf = estimate_sample_rate(window.timestamps) or nominal_srate
        data = window.to_array()

        features = compute_features(data, sf, cfg)

        payload = {
            "t": window.last_timestamp(),
            "sf": sf,
            "channels": n_channels,
            "features": features,
        }

        if args.raw_send:
            # send each feature value as plain text UDP to 127.0.0.1:5005
            for band_values in features.values():
                for v in band_values:
                    sender.sock.sendto(f"{v}".encode(), ("127.0.0.1", 5005))
            last_send_ts = time.time()
        else:
            sender.send(payload)
            last_send_ts = time.time()

        if args.json:
            now = time.time()
            if now - last_json_ts >= args.json_interval:
                tmp_dir = os.path.dirname(args.json_file) or "."
                with tempfile.NamedTemporaryFile("w", dir=tmp_dir, delete=False) as tmp:
                    json.dump(payload, tmp, separators=(",", ":"))
                    tmp.flush()
                    os.fsync(tmp.fileno())
                    tmp_name = tmp.name
                os.replace(tmp_name, args.json_file)
                last_json_ts = now

        if args.text or text_window is not None:
            now = time.time()
            if now - last_text_ts >= args.text_interval:
                parts = []
                for band, values in features.items():
                    if values:
                        avg = float(np.mean(values))
                        parts.append(f"{band}={avg:.4f}")
                line = " ".join(parts)
                if args.text:
                    print(line)
                    sys.stdout.flush()
                if text_window is not None:
                    text_window.append_line(line)
                last_text_ts = now

        if cfg.verbose:
            print(f"sent @ {last_send_ts:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
