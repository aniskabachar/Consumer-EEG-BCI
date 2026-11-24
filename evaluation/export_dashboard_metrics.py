"""Export the final held-out metrics the dashboard (app.py) displays.

Reads the local per-fold probability caches (fold_probs.npz, not in git) and
writes processed_data/final_metrics.json with aggregate numbers only:
per-subject accuracy, confusion matrices, down-sampled ROC points and the
noise-robustness curves. No per-epoch labels or EEG are written.

Usage:  python -m evaluation.export_dashboard_metrics
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve

from evaluation import postprocess as pp
from evaluation.ensemble import fold_probs, load_sets

ROOT = Path('processed_data')
WINDOW = 45
# Noise-test results from results_loso_v2.txt / results_loso_v4.txt
NOISE = {
    'levels': [0, 5, 10, 20, 30, 50],
    'snr_db': [None, 26, 20, 14, 10, 6],
    'v2': [76.46, 76.45, 76.36, 76.04, 75.27, 73.61],
    'v4': [77.92, 77.91, 77.84, 77.39, 76.58, 74.34],
}


def collect(sets, ref, names, window):
    labels, sids = ref['labels'], ref['subject_ids']
    pp.STEP_SECONDS = 1.0
    Y, P, per = [], [], []
    for fold, s in enumerate(np.unique(sids)):
        m = sids == s
        p = pp.causal_smooth(fold_probs(sets, names, fold, s, m.sum()), window)
        Y.append(labels[m]); P.append(p)
        per.append(accuracy_score(labels[m], p >= 0.5))
    return np.concatenate(Y), np.concatenate(P), per


def summarize(y, p, per):
    pred = (p >= 0.5).astype(int)
    cm = confusion_matrix(y, pred)
    fpr, tpr, _ = roc_curve(y, p)
    keep = np.unique(np.linspace(0, len(fpr) - 1, 200).astype(int))
    return dict(
        acc=round(float(accuracy_score(y, pred)), 4),
        f1=round(float(f1_score(y, pred, average='macro')), 4),
        auc=round(float(roc_auc_score(y, p)), 4),
        std=round(float(np.std(per)), 4),
        alert_recall=round(float(cm[0, 0] / cm[0].sum()), 4),
        drowsy_recall=round(float(cm[1, 1] / cm[1].sum()), 4),
        n80=int((np.array(per) >= 0.8).sum()),
        per_subject=[round(float(a), 4) for a in per],
        confusion=cm.tolist(),
        roc=dict(fpr=np.round(fpr[keep], 4).tolist(), tpr=np.round(tpr[keep], 4).tolist()),
    )


def main():
    sets = load_sets(['v2:1', 'v3:1', 'v4:2'], ROOT)
    ref = sets['v2'][0]
    out = {
        'final': summarize(*collect(sets, ref, ('v2', 'v3', 'v4'), WINDOW)),
        'single_raw': summarize(*collect(sets, ref, ('v2',), 1)),
        'noise': NOISE,
        'window_seconds': WINDOW,
        'n_sessions': int(len(np.unique(ref['subject_ids']))),
    }
    path = ROOT / 'final_metrics.json'
    path.write_text(json.dumps(out, indent=1))
    f = out['final']
    print(f"wrote {path}: final acc {f['acc']:.3f}, F1 {f['f1']:.3f}, AUC {f['auc']:.3f}, "
          f"{f['n80']}/23 >= 80%")


if __name__ == '__main__':
    main()
