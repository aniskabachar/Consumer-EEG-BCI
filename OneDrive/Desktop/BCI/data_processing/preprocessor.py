"""
EEG Preprocessing Pipeline
===========================
Converts raw SEED-VIG 17-channel EEG (1000Hz) into clean 5-channel epochs (200Hz)
ready for attention/relaxed classification.

Core Logic:
  1000Hz raw → Downsample to 200Hz → Bandpass 0.5-45Hz → Notch 50/60Hz
  → 2sec epochs (50% overlap) → Z-score normalize → Channel selection (5ch)
"""

import numpy as np
import scipy.io
from scipy import signal
from pathlib import Path


# ─────────────────────────────────────────────────
#  CHANNEL CONFIGURATION
#  These 5 channels are selected by data-driven analysis
#  (MI + Random Forest + RFE all agree on these)
#  They map to forehead + crown + back-of-head positions
#  which are accessible on consumer 4-5 electrode headsets
# ─────────────────────────────────────────────────
SEED_VIG_CHANNELS = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4',
    'F7', 'F5', 'F3', 'F1', 'FZ',
    'F2', 'F4', 'F6', 'F8', 'FC5',
    'FC3', 'FC1'
]  # 17 channels in SEED-VIG

# Best 5 channels for attention detection
# Why these? Frontal (FP1/FP2/FZ) = prefrontal cortex = executive attention
# Parietal (PZ) = sustained attention   Occipital (OZ) = visual attention
# NOTE: SEED-VIG has 17 channels, we approximate these positions
SELECTED_5CH_INDICES = [0, 1, 2, 9, 16]  # FP1, FPZ, FP2, FZ, FC1 (closest to Fp1,Fp2,Fz,Pz,Oz)
SELECTED_5CH_NAMES = ['Fp1', 'Fpz', 'Fp2', 'Fz', 'FC1']


class EEGPreprocessor:
    """
    Complete preprocessing pipeline for SEED-VIG → Consumer BCI.

    Pipeline stages:
    ┌─────────────────────────────────────────────────────────┐
    │  Raw 1000Hz → Decimate to 200Hz                         │
    │  → Butterworth Bandpass (0.5-45Hz)                      │
    │  → Notch Filter (50Hz + 60Hz)                           │
    │  → Epoch into 2s windows (400 samples) w/ 50% overlap   │
    │  → Z-score normalize per channel                        │
    │  → Select 5 channels                                    │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(self, fs_original=200, fs_target=200):
        # SEED-VIG Raw_Data is ALREADY at 200 Hz (confirmed: sample_rate field = 200)
        # So fs_original == fs_target == 200, and downsample is skipped.
        self.fs_original = fs_original
        self.fs_target = fs_target
        self.samples_per_epoch = int(2.0 * fs_target)   # 2s × 200Hz = 400 samples
        self.step_size = self.samples_per_epoch // 2     # 50% overlap = 200 samples
        # PERCLOS: 885 values, each spans 8 seconds = 8×200 = 1600 EEG samples
        self.perclos_samples_per_label = int(8.0 * fs_target)  # = 1600

    # ─────────────────────────────────────────────
    #  STEP 1: DOWNSAMPLE
    #  WHY: 1000Hz is overkill. EEG brain rhythms
    #       we care about are 0.5-45Hz. Nyquist theorem
    #       says we need 2× max freq = 90Hz minimum.
    #       200Hz gives us headroom + reduces data 5×.
    #  HOW: scipy.signal.decimate (NOT just slicing!)
    #       decimate = anti-aliasing low-pass filter FIRST,
    #       then downsample. Just slicing every 5th sample
    #       causes aliasing (high freq folds into low freq).
    # ─────────────────────────────────────────────
    def downsample(self, data):
        """
        data: (samples, channels) at 1000Hz
        returns: (samples//5, channels) at 200Hz
        """
        # decimate handles anti-aliasing automatically
        # axis=0 = downsample along time axis
        downsampled = signal.decimate(data, self.downsample_factor, axis=0, zero_phase=True)
        return downsampled

    # ─────────────────────────────────────────────
    #  STEP 2: BANDPASS FILTER 0.5 - 45 Hz
    #  WHY:
    #   <0.5Hz = electrode drift (slow DC wander) - NOT brain
    #   >45Hz  = muscle noise (EMG), power line harmonics - NOT brain
    #   0.5-45Hz contains ALL EEG bands we need:
    #     Delta (0.5-4Hz)  = deep states
    #     Theta (4-8Hz)    = drowsiness, relaxation
    #     Alpha (8-13Hz)   = relaxed alertness (KEY for us!)
    #     Beta  (13-30Hz)  = active attention (KEY for us!)
    #     Gamma (30-45Hz)  = focused cognition
    #  HOW: 5th-order Butterworth (maximally flat passband)
    #       filtfilt = zero-phase (no time delay distortion)
    # ─────────────────────────────────────────────
    def bandpass_filter(self, data, lowcut=0.5, highcut=45.0):
        nyquist = self.fs_target / 2  # = 100 Hz
        low = lowcut / nyquist         # = 0.005
        high = highcut / nyquist       # = 0.45
        sos = signal.butter(5, [low, high], btype='band', output='sos')
        filtered = signal.sosfiltfilt(sos, data, axis=0)
        return filtered

    # ─────────────────────────────────────────────
    #  STEP 3: NOTCH FILTERS (50Hz + 60Hz)
    #  WHY: Powerline noise. India/EU = 50Hz. USA = 60Hz.
    #       We remove both because:
    #       a) SEED-VIG recorded in China = 50Hz issue
    #       b) Consumer headsets might be used anywhere
    #       These frequencies are NOT brain signals.
    #  HOW: IIR notch filter (narrow stop-band)
    #       Q=30 = sharp notch (only kills 49-51Hz, 59-61Hz)
    # ─────────────────────────────────────────────
    def notch_filter(self, data, freq=50.0):
        w0 = freq / (self.fs_target / 2)  # normalized frequency
        b, a = signal.iirnotch(w0, Q=30, fs=self.fs_target)
        filtered = signal.filtfilt(b, a, data, axis=0)
        return filtered

    # ─────────────────────────────────────────────
    #  STEP 4: EPOCH EXTRACTION
    #  WHY: Neural networks need fixed-size inputs.
    #       We slice continuous EEG into 2-second windows.
    #       2s = short enough to capture moment-to-moment
    #       state, long enough to see all EEG rhythms.
    #  50% overlap = doubles our dataset size, allows
    #       gradual state transitions to be captured.
    #  Result: each epoch is (400 samples, channels)
    # ─────────────────────────────────────────────
    def extract_epochs(self, data):
        """
        data: (total_samples, channels)
        returns: (num_epochs, samples_per_epoch, channels)
        """
        n_samples = data.shape[0]
        epochs = []
        start = 0
        while start + self.samples_per_epoch <= n_samples:
            epoch = data[start:start + self.samples_per_epoch, :]
            epochs.append(epoch)
            start += self.step_size
        return np.array(epochs)  # shape: (N, 400, channels)

    # ─────────────────────────────────────────────
    #  STEP 5: Z-SCORE NORMALIZATION (per channel, per epoch)
    #  WHY: Every person has different EEG amplitude.
    #       Person A might have 50μV signals, Person B 200μV.
    #       Without normalization, model learns amplitude
    #       instead of pattern → fails on new users.
    #  HOW: (x - mean) / std  per channel independently.
    #       This makes all channels comparable scale.
    #  IMPORTANT: We normalize per-epoch (not global)
    #             so the model works on new users at runtime
    #             without needing their historical stats.
    # ─────────────────────────────────────────────
    def normalize_epochs(self, epochs):
        """
        epochs: (N, 400, channels)
        returns: (N, 400, channels) z-scored per channel per epoch
        """
        mean = epochs.mean(axis=1, keepdims=True)   # shape: (N, 1, channels)
        std  = epochs.std(axis=1, keepdims=True) + 1e-8
        return (epochs - mean) / std

    # ─────────────────────────────────────────────
    #  STEP 6: SELECT 5 CHANNELS
    #  WHY: Consumer headsets only have 4-5 electrodes.
    #       Training on ALL 17 channels then only using 5
    #       at inference = mismatch → bad accuracy.
    #  Solution: Train ONLY on these 5 channels from start.
    #            The model never sees other channels.
    # ─────────────────────────────────────────────
    def select_channels(self, epochs, indices=SELECTED_5CH_INDICES):
        """
        epochs: (N, 400, 17) → (N, 400, 5)
        """
        return epochs[:, :, indices]

    def assign_labels(self, perclos_values, n_epochs):
        """
        PERCLOS → binary epoch labels.

        SEED-VIG PERCLOS structure (from debug output):
          shape=(885,)  → 885 values, one per 8-second segment
          Each value = fraction of time eyes were closed in that segment

        EEG structure:
          shape=(1416000, 17)  at 200 Hz
          1416000 / 200 = 7080 seconds = 885 × 8 seconds  ✓

        We expand PERCLOS to per-sample resolution by repeating each value
        1600 times (8s × 200Hz), then slice the same 2-second windows
        we use for EEG epochs.

        Label:
          PERCLOS > 0.5 → drowsy / relaxed (label = 1)
          PERCLOS ≤ 0.5 → alert  / focused (label = 0)
        """
        # Expand: (885,) → (1416000,) by repeating each value 1600 times
        perclos_expanded = np.repeat(perclos_values, self.perclos_samples_per_label)

        labels = []
        start = 0
        for _ in range(n_epochs):
            end = start + self.samples_per_epoch
            if end > len(perclos_expanded):
                break
            window_perclos = perclos_expanded[start:end]
            label = 1 if float(np.median(window_perclos)) > 0.5 else 0
            labels.append(label)
            start += self.step_size

        return np.array(labels[:n_epochs])

    def process_subject(self, raw_eeg, perclos_labels, use_5ch=True):
        """
        Complete pipeline for one subject.

        Args:
            raw_eeg: (samples, 17) raw EEG at 1000Hz
            perclos_labels: (samples,) PERCLOS values
            use_5ch: if True, return only 5 selected channels

        Returns:
            epochs: (N, 400, 5) preprocessed epochs
            labels: (N,) binary labels
        """
        # Ensure shape: (samples, channels)
        if raw_eeg.ndim == 2 and raw_eeg.shape[0] < raw_eeg.shape[1]:
            raw_eeg = raw_eeg.T

        # Pipeline
        # NOTE: NO downsample step — SEED-VIG is already at 200Hz
        data = self.bandpass_filter(raw_eeg)        # (N, 17)
        data = self.notch_filter(data, freq=50.0)   # (N, 17)
        data = self.notch_filter(data, freq=60.0)   # (N, 17)
        epochs = self.extract_epochs(data)           # (E, 400, 17)
        epochs = self.normalize_epochs(epochs)       # (E, 400, 17)

        labels = self.assign_labels(perclos_labels, len(epochs))

        if use_5ch:
            epochs = self.select_channels(epochs)  # (N, 400, 5)

        # Trim to matching length
        n = min(len(epochs), len(labels))
        return epochs[:n], labels[:n]


def _extract_struct_field(mat_obj):
    """
    SEED-VIG stores data as a MATLAB struct:
      raw['data'] has dtype [('data','O'), ('chn','O'), ('sample_rate','O'), ('node_number','O')]
    
    We need to drill into it:
      mat_obj[0, 0]['data']  →  the actual EEG ndarray
    """
    # If it's already a plain numeric array, just return it
    if mat_obj.dtype.names is None:
        return mat_obj

    # It's a struct — extract the 'data' field from the (1,1) cell
    inner = mat_obj.flat[0]          # unwrap the (1,1) object array
    if hasattr(inner, 'dtype') and inner.dtype.names and 'data' in inner.dtype.names:
        return np.array(inner['data'], dtype=np.float64)
    # fallback: try to return it as-is
    return np.array(inner, dtype=np.float64)


def load_seed_vig_subject(raw_path, label_path):
    """
    Load one SEED-VIG subject raw data and PERCLOS labels.

    Actual SEED-VIG structure (confirmed from debug_mat.py):
      raw['EEG']  → MATLAB struct with fields: data, chn, sample_rate, node_number
      raw['EEG'].flat[0]['data']  → (N_samples, 17) float64  at 200 Hz already!
      lbl['perclos']  → (885, 1) float64, one value per 8-second segment

    Key insight: sample_rate = 200 Hz (already downsampled in the dataset).
    We do NOT need to downsample again. The preprocessor will skip that step.

    PERCLOS has 885 values. Each covers 8 seconds at 200Hz = 1600 samples.
    Total EEG: 885 × 1600 = 1,416,000 samples. Matches shape (1416000, 17). ✓
    """
    raw = scipy.io.loadmat(raw_path, squeeze_me=False)
    lbl = scipy.io.loadmat(label_path, squeeze_me=False)

    # ── EEG data ──────────────────────────────────────────
    # Structure: raw['EEG'] is a (1,1) struct → .flat[0] unwraps it
    # Then ['data'] gives the (1416000, 17) float64 matrix
    eeg_struct = raw['EEG'].flat[0]
    eeg_data = np.array(eeg_struct['data'], dtype=np.float32)   # (N, 17)

    # Ensure shape is (samples, channels)
    if eeg_data.shape[0] < eeg_data.shape[1]:
        eeg_data = eeg_data.T

    # ── PERCLOS labels ────────────────────────────────────
    # Shape: (885, 1) → flatten to (885,)
    # Each value = fraction of time eyes closed in that 8-second segment
    perclos = np.array(lbl['perclos'], dtype=np.float32).flatten()  # (885,)

    return eeg_data, perclos


def debug_mat_structure(mat_path):
    """
    Run this if you get loading errors. Prints the full nested
    structure of a .mat file so you can see where data lives.

    Usage (in Python shell or notebook):
        from data_processing.preprocessor import debug_mat_structure
        debug_mat_structure('./SEED-VIG/Raw_Data/1_20151124_noon_2.mat')
    """
    def _describe(obj, name, depth=0):
        indent = '  ' * depth
        if hasattr(obj, 'dtype'):
            if obj.dtype.names:
                print(f"{indent}{name}: STRUCT fields={obj.dtype.names}  shape={obj.shape}")
                inner = obj.flat[0]
                for field in obj.dtype.names:
                    _describe(inner[field], f"{name}.{field}", depth + 1)
            elif obj.dtype == object:
                print(f"{indent}{name}: CELL  shape={obj.shape}")
                if obj.size > 0:
                    _describe(obj.flat[0], f"{name}[0]", depth + 1)
            else:
                try:
                    print(f"{indent}{name}: ARRAY  dtype={obj.dtype}  shape={obj.shape}  "
                          f"min={float(obj.min()):.3f}  max={float(obj.max()):.3f}")
                except Exception:
                    print(f"{indent}{name}: ARRAY  dtype={obj.dtype}  shape={obj.shape}")
        else:
            print(f"{indent}{name}: {type(obj).__name__} = {obj}")

    import scipy.io as sio
    mat = sio.loadmat(mat_path, squeeze_me=False)
    print(f"\n=== Structure of: {mat_path} ===")
    for key in mat:
        if not key.startswith('__'):
            _describe(mat[key], key)
    print("=" * 40)


def process_all_subjects(seed_vig_root, output_path):
    """Process all SEED-VIG subjects and save unified dataset."""
    raw_dir = Path(seed_vig_root) / 'Raw_Data'
    lbl_dir = Path(seed_vig_root) / 'perclos_labels'
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    preprocessor = EEGPreprocessor()
    mat_files = sorted(raw_dir.glob('*.mat'))

    all_epochs, all_labels, all_subject_ids = [], [], []

    for idx, mat_file in enumerate(mat_files):
        print(f"[{idx+1}/{len(mat_files)}] Processing {mat_file.name} ...")
        label_file = lbl_dir / mat_file.name

        try:
            eeg, perclos = load_seed_vig_subject(str(mat_file), str(label_file))
            epochs, labels = preprocessor.process_subject(eeg, perclos)

            all_epochs.append(epochs)
            all_labels.append(labels)
            all_subject_ids.extend([idx] * len(labels))

            print(f"  → {len(labels)} epochs | Alert: {(labels==0).sum()} | Drowsy: {(labels==1).sum()}")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    all_epochs = np.concatenate(all_epochs, axis=0).astype(np.float32)
    all_labels = np.concatenate(all_labels).astype(np.int64)
    all_subject_ids = np.array(all_subject_ids, dtype=np.int64)

    out_file = output_path / 'seed_vig_5ch_processed.npz'
    np.savez_compressed(
        str(out_file),
        epochs=all_epochs,
        labels=all_labels,
        subject_ids=all_subject_ids
    )

    print(f"\n✅ Saved: {out_file}")
    print(f"   Shape: {all_epochs.shape}  |  Labels: {all_labels.shape}")
    print(f"   Subjects: {len(np.unique(all_subject_ids))}")
    print(f"   Alert: {(all_labels==0).sum()} | Drowsy: {(all_labels==1).sum()}")

    return all_epochs, all_labels, all_subject_ids