"""
Post-processing: causal temporal smoothing + decision threshold
================================================================
Both settings are tuned per LOSO fold on that fold's VALIDATION subjects
(the same 3 subjects the trainer used for epoch selection) and only then
applied to the held-out test subject. The test subject never influences
the choice, so the result stays leak-free.

Smoothing is causal (moving average over the current and previous k epochs),
so it can run in real time. Epoch step is 1 s (2 s windows, 50% overlap),
so k epochs ~= k seconds of history / reaction lag.

Usage:
  python -m evaluation.postprocess --data_path ./processed_data/seed_vig_5ch_processed.npz \
         --models_dir ./processed_data/saved_models_v2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from evaluation.models.bci_model import build_model

WINDOWS = (1, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120)   # smoothing window in SECONDS
STEP_SECONDS = 1.0   # epoch step; set from the dataset in main()
THRESHOLDS = np.round(np.arange(0.20, 0.801, 0.02), 2)


def val_subjects_per_fold(unique_subjects, n_val_subjects=3, seed=42):
    """Re-create exactly the validation split used in training/trainer.py."""
    rng = np.random.default_rng(seed)
    out = []
    for test_subject in unique_subjects:
        candidates = unique_subjects[unique_subjects != test_subject]
        out.append(np.sort(rng.choice(candidates, size=n_val_subjects, replace=False)))
    return out


@torch.no_grad()
def predict_proba(model, X, device, batch_size=512):
    probs = []
    for i in range(0, len(X), batch_size):
        # stored as (N, time, channels); the model expects (N, channels, time)
        xb = torch.from_numpy(X[i:i + batch_size]).float().transpose(1, 2).to(device)
        probs.append(torch.softmax(model(xb), dim=1)[:, 1].float().cpu().numpy())
    return np.concatenate(probs)


def causal_smooth(p, seconds):
    """Mean over the last `seconds` of epochs (shorter window at the start of a recording)."""
    k = max(1, int(round(seconds / STEP_SECONDS)))
    if k <= 1:
        return p
    c = np.cumsum(np.insert(p, 0, 0.0))
    t = np.arange(1, len(p) + 1)
    lo = np.maximum(0, t - k)
    return (c[t] - c[lo]) / (t - lo)


def compute_probs(data_path, models_dir, cache_path, device):
    data = np.load(data_path)
    epochs, labels, subject_ids = data['epochs'], data['labels'], data['subject_ids']
    unique_subjects = np.unique(subject_ids)
    val_sets = val_subjects_per_fold(unique_subjects)

    store = {'labels': labels, 'subject_ids': subject_ids}
    for fold, test_subj in enumerate(unique_subjects):
        path = Path(models_dir) / f'fold_{fold+1}_subject_{test_subj}.pth'
        model = build_model(n_channels=5).to(device)
        model.load_state_dict(torch.load(str(path), map_location=device))
        model.eval()
        for s in [test_subj, *val_sets[fold]]:
            store[f'f{fold}_s{s}'] = predict_proba(model, epochs[subject_ids == s], device)
        store[f'f{fold}_val'] = val_sets[fold]
        print(f"  fold {fold+1:2d}/{len(unique_subjects)} done (val {val_sets[fold].tolist()})", flush=True)
    np.savez(cache_path, **store)
    return dict(np.load(cache_path))


def pooled(store, fold, subjects, k, t):
    ys, ps = [], []
    for s in subjects:
        ys.append(store['labels'][store['subject_ids'] == s])
        ps.append(causal_smooth(store[f'f{fold}_s{s}'], k))
    y, p = np.concatenate(ys), np.concatenate(ps)
    return y, p, (p >= t).astype(int)


def tune(store, fold, metric, windows, thresholds):
    best = (-1, 1, 0.5)
    for k in windows:
        y, p, _ = pooled(store, fold, store[f'f{fold}_val'], k, 0.5)
        for t in thresholds:
            pred = (p >= t).astype(int)
            score = accuracy_score(y, pred) if metric == 'acc' else f1_score(y, pred, average='macro')
            if score > best[0]:
                best = (score, k, t)
    return best[1], best[2]


def run(store, mode, metric='acc'):
    unique_subjects = np.unique(store['subject_ids'])
    Y, P, PR, per_subj, chosen = [], [], [], [], []
    for fold, s in enumerate(unique_subjects):
        if mode == 'baseline':
            k, t = 1, 0.5
        elif mode == 'smooth':
            k, t = tune(store, fold, metric, WINDOWS, [0.5])
        elif mode == 'threshold':
            k, t = tune(store, fold, metric, [1], THRESHOLDS)
        else:
            k, t = tune(store, fold, metric, WINDOWS, THRESHOLDS)
        y, p, pred = pooled(store, fold, [s], k, t)
        Y.append(y); P.append(p); PR.append(pred)
        per_subj.append(accuracy_score(y, pred)); chosen.append((k, t))
    y, p, pred = np.concatenate(Y), np.concatenate(P), np.concatenate(PR)
    cm = confusion_matrix(y, pred)
    return {
        'acc': accuracy_score(y, pred), 'f1': f1_score(y, pred, average='macro'),
        'auc': roc_auc_score(y, p), 'mean': np.mean(per_subj), 'std': np.std(per_subj),
        'rec_alert': cm[0, 0] / cm[0].sum(), 'rec_drowsy': cm[1, 1] / cm[1].sum(),
        'per_subj': per_subj, 'chosen': chosen,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', default='./processed_data/seed_vig_5ch_processed.npz')
    ap.add_argument('--models_dir', default='./processed_data/saved_models_v2')
    ap.add_argument('--recompute', action='store_true')
    ap.add_argument('--fixed_window', type=float, default=45,
                    help='Also report one fixed smoothing window (seconds) for all folds')
    args = ap.parse_args()

    global STEP_SECONDS
    with np.load(args.data_path) as d:
        STEP_SECONDS = float(d['step_seconds']) if 'step_seconds' in d.files else 1.0
    print(f"Epoch step: {STEP_SECONDS:g} s")

    cache = Path(args.models_dir) / 'fold_probs.npz'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if cache.exists() and not args.recompute:
        store = dict(np.load(cache))
    else:
        print(f"Computing per-fold probabilities on {device} ...")
        store = compute_probs(args.data_path, args.models_dir, cache, device)

    rows = [('Baseline (no post-processing)', 'baseline', 'acc'),
            ('Smoothing only (tuned for acc)', 'smooth', 'acc'),
            ('Threshold only (tuned for acc)', 'threshold', 'acc'),
            ('Smoothing + threshold (acc)', 'both', 'acc'),
            ('Smoothing + threshold (macro F1)', 'both', 'f1')]
    print(f"\n{'Setting':34s} {'Acc':>6s} {'F1':>6s} {'AUC':>6s} {'Mean±Std':>12s} {'Alert':>6s} {'Drowsy':>6s}")
    print('-' * 84)
    results = {}
    for name, mode, metric in rows:
        r = run(store, mode, metric)
        results[name] = r
        print(f"{name:34s} {r['acc']:6.4f} {r['f1']:6.4f} {r['auc']:6.4f} "
              f"{r['mean']:.3f}±{r['std']:.3f} {r['rec_alert']:6.3f} {r['rec_drowsy']:6.3f}")
    global WINDOWS
    tuned_windows, WINDOWS = WINDOWS, (args.fixed_window,)
    name = f'Smoothing, fixed {args.fixed_window:g} s window'
    r = run(store, 'smooth', 'acc')
    results[name] = r
    WINDOWS = tuned_windows
    print(f"{name:34s} {r['acc']:6.4f} {r['f1']:6.4f} {r['auc']:6.4f} "
          f"{r['mean']:.3f}±{r['std']:.3f} {r['rec_alert']:6.3f} {r['rec_drowsy']:6.3f}")
    print("(Alert / Drowsy = per-class recall; settings tuned per fold on validation subjects only)")

    for name in ('Smoothing + threshold (acc)', 'Smoothing + threshold (macro F1)'):
        ks = [k for k, _ in results[name]['chosen']]
        ts = [t for _, t in results[name]['chosen']]
        print(f"\n{name}: window median {int(np.median(ks))} s (range {min(ks)}-{max(ks)}), "
              f"threshold median {np.median(ts):.2f} (range {min(ts):.2f}-{max(ts):.2f})")

    base, best = results['Baseline (no post-processing)'], results['Smoothing + threshold (acc)']
    print("\nPer-subject accuracy (baseline -> smoothing+threshold):")
    for i, (a, b) in enumerate(zip(base['per_subj'], best['per_subj'])):
        print(f"  Subject {i:2d}: {a:.3f} -> {b:.3f}  ({b - a:+.3f})")


if __name__ == '__main__':
    main()
