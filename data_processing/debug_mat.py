"""
debug_mat.py  —  Run this FIRST to inspect your .mat file structure.

Usage:
    python debug_mat.py
"""

import scipy.io
import numpy as np


def describe(obj, name, depth=0):
    indent = '  ' * depth
    if hasattr(obj, 'dtype'):
        if obj.dtype.names:           # MATLAB struct
            print(f"{indent}[STRUCT] {name}  fields={obj.dtype.names}  shape={obj.shape}")
            inner = obj.flat[0]
            for field in obj.dtype.names:
                describe(inner[field], f"{name}.{field}", depth + 1)
        elif obj.dtype == object:     # cell array
            print(f"{indent}[CELL]   {name}  shape={obj.shape}")
            if obj.size > 0:
                describe(obj.flat[0], f"{name}[0]", depth + 1)
        else:                         # plain numeric array
            try:
                print(f"{indent}[ARRAY]  {name}  dtype={obj.dtype}  shape={obj.shape}  "
                      f"min={float(obj.min()):.4f}  max={float(obj.max()):.4f}")
            except Exception:
                print(f"{indent}[ARRAY]  {name}  dtype={obj.dtype}  shape={obj.shape}")
    else:
        print(f"{indent}[VALUE]  {name} = {obj}")


# ── Change these paths to your actual files ──────────────────────
RAW_FILE   = './SEED-VIG/Raw_Data/1_20151124_noon_2.mat'
LABEL_FILE = './SEED-VIG/perclos_labels/1_20151124_noon_2.mat'
# ─────────────────────────────────────────────────────────────────

print("=" * 60)
print("RAW EEG FILE")
print("=" * 60)
raw = scipy.io.loadmat(RAW_FILE, squeeze_me=False)
for key in raw:
    if not key.startswith('__'):
        describe(raw[key], key)

print("\n" + "=" * 60)
print("PERCLOS LABEL FILE")
print("=" * 60)
lbl = scipy.io.loadmat(LABEL_FILE, squeeze_me=False)
for key in lbl:
    if not key.startswith('__'):
        describe(lbl[key], key)

# ── Try to manually extract the EEG array ────────────────────────
print("\n" + "=" * 60)
print("ATTEMPTING MANUAL EXTRACTION")
print("=" * 60)

# SEED-VIG struct path: raw['EEG'].flat[0]['data']
try:
    eeg = raw['EEG'].flat[0]['data']
    eeg = np.array(eeg, dtype=np.float32)
    if eeg.shape[0] < eeg.shape[1]:
        eeg = eeg.T    # ensure (samples, channels)
    fs = int(raw['EEG'].flat[0]['sample_rate'].flat[0])
    print(f"✅ EEG extracted: shape={eeg.shape}  sample_rate={fs} Hz")
    print(f"   Duration: {eeg.shape[0]/fs/60:.1f} minutes")
except Exception as e:
    print(f"✗ Manual extraction failed: {e}")
    print("  Please paste the output above and share.")

try:
    perclos = np.array(lbl['perclos'], dtype=np.float32).flatten()
    print(f"✅ PERCLOS extracted: shape={perclos.shape}  "
          f"range=[{perclos.min():.3f}, {perclos.max():.3f}]")
except Exception as e:
    print(f"✗ PERCLOS extraction failed: {e}")
    print("  Available keys:", [k for k in lbl if not k.startswith('__')])