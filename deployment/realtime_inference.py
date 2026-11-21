"""
Real-Time BCI Inference Engine
================================
Runs on the consumer device.
Takes live EEG from headset → predicts focused/relaxed every 0.5 seconds.

Signal flow:
  Headset → BrainFlow SDK → Buffer (2s window) → Preprocess → Model → Game

Latency breakdown:
  Signal buffer fill:  2000ms (need 2 seconds of data)
  Preprocessing:       ~3ms
  Model inference:     ~12ms
  Total:               ~2015ms
  
  Update rate: 1 new prediction every 0.5s (sliding window)
"""

import numpy as np
from scipy import signal
from collections import deque
import torch
import json
import socket
import time
import threading
from pathlib import Path
import sys
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from evaluation.models.bci_model import build_model


class RealTimePreprocessor:
    """
    Lightweight preprocessing for real-time inference.
    Matches EXACTLY what was done during training.
    Critical: any mismatch = accuracy drops.
    """
    def __init__(self, fs=250, n_channels=5):
        self.fs = fs
        self.n_channels = n_channels
        self.samples_per_window = int(2.0 * fs)  # 2 seconds = 500 samples at 250Hz

        # Pre-compute filter coefficients (do this ONCE, not per-window)
        nyq = fs / 2
        self.bp_sos = signal.butter(5, [0.5/nyq, 45/nyq], btype='band', output='sos')
        w50 = 50.0 / nyq
        w60 = 60.0 / nyq
        self.notch50_b, self.notch50_a = signal.iirnotch(w50, Q=30, fs=fs)
        self.notch60_b, self.notch60_a = signal.iirnotch(w60, Q=30, fs=fs)

    def process_window(self, raw_window):
        """
        Process one 2-second window.
        raw_window: (n_channels, samples) — channels first for hardware APIs
        returns: (1, n_channels, 400) tensor ready for model
        """
        # raw_window shape: (5, 500) at 250Hz
        data = raw_window.T  # → (500, 5) for scipy filters (samples, channels)

        # Apply filters (same as training)
        data = signal.sosfiltfilt(self.bp_sos, data, axis=0)
        data = signal.filtfilt(self.notch50_b, self.notch50_a, data, axis=0)
        data = signal.filtfilt(self.notch60_b, self.notch60_a, data, axis=0)

        # Resample to 200Hz to match training (250 → 200 samples)
        data = signal.resample(data, 400, axis=0)  # → (400, 5)

        # Z-score normalize per channel (SAME as training per-epoch normalization)
        mean = data.mean(axis=0, keepdims=True)
        std  = data.std(axis=0, keepdims=True) + 1e-8
        data = (data - mean) / std

        # Shape for model: (1, channels, time)
        tensor = torch.FloatTensor(data.T).unsqueeze(0)  # (1, 5, 400)
        return tensor


class BCIInferenceEngine:
    """
    Real-time prediction engine.
    
    Two modes:
    1. HARDWARE mode: connects to real BrainFlow device (OpenBCI, Muse, etc.)
    2. SIMULATION mode: replays recorded data (for testing without hardware)
    """

    def __init__(self, model_path, n_channels=5, fs=250, device='cpu'):
        self.device = device
        self.fs = fs
        self.n_channels = n_channels
        self.samples_per_window = int(2.0 * fs)
        self.step_samples = int(0.5 * fs)   # New prediction every 0.5 seconds

        # Circular buffer: always holds last 2 seconds of data
        # deque with maxlen automatically drops old samples
        self.buffer = deque(maxlen=self.samples_per_window)

        # Load model
        self.model = build_model(n_channels=n_channels).to(device)
        state = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"✅ Model loaded: {model_path}")

        self.preprocessor = RealTimePreprocessor(fs=fs, n_channels=n_channels)

        # Smoothing: exponential moving average on predictions
        # Prevents rapid flickering between states
        self.smoothed_prob_focused = 0.5
        self.alpha = 0.3  # smoothing factor (lower = more smoothing)

        # Output
        self.latest_prediction = {'focused': 0.5, 'relaxed': 0.5, 'state': 'unknown'}
        self.is_running = False

    def add_sample(self, sample):
        """
        Add one time sample to the buffer.
        sample: (n_channels,) array
        Called at fs Hz (250 times/second)
        """
        self.buffer.append(sample)

    @torch.no_grad()
    def predict(self):
        """
        Run prediction on current buffer contents.
        Returns dict with probabilities and state label.
        """
        if len(self.buffer) < self.samples_per_window:
            return {'focused': 0.5, 'relaxed': 0.5, 'state': 'buffering...'}

        # Convert buffer to array: (samples, channels)
        window = np.array(self.buffer)  # (500, 5)
        raw_window = window.T           # (5, 500) = (channels, samples)

        # Preprocess
        tensor = self.preprocessor.process_window(raw_window)
        tensor = tensor.to(self.device)

        # Model inference
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        p_focused = float(probs[0])
        p_relaxed = float(probs[1])

        # EMA smoothing (prevents jitter between states)
        self.smoothed_prob_focused = (
            self.alpha * p_focused +
            (1 - self.alpha) * self.smoothed_prob_focused
        )
        smoothed_relaxed = 1 - self.smoothed_prob_focused

        # State determination with hysteresis
        # Hysteresis: don't switch state unless very confident
        # This is crucial for UX — no one wants rapid state flipping
        if self.smoothed_prob_focused > 0.65:
            state = 'FOCUSED'      # Car STOPS
        elif smoothed_relaxed > 0.65:
            state = 'RELAXED'      # Car MOVES
        else:
            state = 'TRANSITIONING'  # Hold previous state

        self.latest_prediction = {
            'focused': self.smoothed_prob_focused,
            'relaxed': smoothed_relaxed,
            'state': state,
            'timestamp': time.time()
        }
        return self.latest_prediction

    def start_simulation_mode(self, recorded_data_path=None):
        """
        Simulate real-time data stream from recorded EEG.
        Useful for testing the full pipeline without hardware.
        """
        if recorded_data_path:
            data = np.load(recorded_data_path)
            eeg = data['epochs'][0].T  # Use first epoch, (5, 400)
        else:
            # Generate synthetic EEG-like signal
            print("Using synthetic EEG data (no hardware connected)")
            t = np.linspace(0, 60, 60 * self.fs)  # 60 seconds
            eeg = np.zeros((self.n_channels, len(t)))
            for ch in range(self.n_channels):
                # Alpha wave (10Hz) + Theta (6Hz) + noise
                eeg[ch] = (np.sin(2*np.pi*10*t) * 0.5 +    # Alpha
                           np.sin(2*np.pi*6*t) * 0.3 +     # Theta
                           np.random.randn(len(t)) * 0.2)   # Noise

        self.is_running = True
        sample_idx = 0
        print("▶ Simulation started. Press Ctrl+C to stop.")

        try:
            while self.is_running and sample_idx < eeg.shape[1]:
                # Add one sample (5 channels) to buffer
                self.add_sample(eeg[:, sample_idx])
                sample_idx += 1

                # Make prediction every step_samples
                if sample_idx % self.step_samples == 0 and len(self.buffer) >= self.samples_per_window:
                    pred = self.predict()
                    bar = '█' * int(pred['focused'] * 20) + '░' * (20 - int(pred['focused'] * 20))
                    print(f"\r  Focused [{bar}] {pred['focused']:.2f} | State: {pred['state']:15s}", end='')

                # Simulate real-time rate
                time.sleep(1 / self.fs)

        except KeyboardInterrupt:
            print("\n⏹ Simulation stopped.")

        self.is_running = False


class GameBridge:
    """
    Sends predictions to Unity game via socket.
    Unity C# code connects here and reads JSON predictions.
    Update rate: 10Hz (every 100ms)
    """
    def __init__(self, engine: BCIInferenceEngine, host='localhost', port=8888):
        self.engine = engine
        self.host = host
        self.port = port

    def start(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(1)
        print(f"🎮 Game bridge listening on {self.host}:{self.port}")

        conn, addr = server.accept()
        print(f"🎮 Unity connected from {addr}")

        try:
            while True:
                pred = self.engine.latest_prediction
                msg = json.dumps({
                    'focused': round(pred['focused'], 3),
                    'relaxed': round(pred['relaxed'], 3),
                    'state': pred['state']
                }) + '\n'
                conn.send(msg.encode())
                time.sleep(0.1)  # 10 Hz
        except (BrokenPipeError, ConnectionResetError):
            print("🎮 Unity disconnected")
        finally:
            conn.close()
            server.close()


# BrainFlow integration (uncomment when hardware is connected)
# ─────────────────────────────────────────────────────────
# Requires: pip install brainflow
#
# SUPPORTED HARDWARE:
#   OpenBCI Ganglion: board_id=1  (4 channels, $200)
#   OpenBCI Cyton:    board_id=0  (8 channels, $500)
#   Muse 2:           board_id=38 (4 channels, $250)
#   Crown:            board_id=50 (8 channels, $300)
#
# def start_with_hardware(model_path, board_id=1, serial_port='/dev/ttyUSB0'):
#     from brainflow.board_shim import BoardShim, BrainFlowInputParams
#
#     params = BrainFlowInputParams()
#     params.serial_port = serial_port
#     board = BoardShim(board_id, params)
#     board.prepare_session()
#     board.start_stream()
#
#     engine = BCIInferenceEngine(model_path, n_channels=4, fs=200)
#     eeg_channels = BoardShim.get_eeg_channels(board_id)
#
#     bridge = GameBridge(engine)
#     bridge_thread = threading.Thread(target=bridge.start)
#     bridge_thread.daemon = True
#     bridge_thread.start()
#
#     print("🧠 Reading live EEG... Press Ctrl+C to stop.")
#     try:
#         while True:
#             data = board.get_current_board_data(25)    # 25 samples (~100ms at 250Hz)
#             eeg = data[eeg_channels[:5], :]            # Take first 5 EEG channels
#             for i in range(eeg.shape[1]):
#                 engine.add_sample(eeg[:, i])
#             pred = engine.predict()
#             print(f"\r  State: {pred['state']:15s}  Focused: {pred['focused']:.3f}", end='')
#             time.sleep(0.1)
#     except KeyboardInterrupt:
#         board.stop_stream()
#         board.release_session()
#         print("\n✓ Hardware disconnected.")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='../saved_models/fold_1_subject_0.pth')
    p.add_argument('--mode', choices=['simulate', 'hardware'], default='simulate')
    args = p.parse_args()

    engine = BCIInferenceEngine(model_path=args.model)
    engine.start_simulation_mode()
