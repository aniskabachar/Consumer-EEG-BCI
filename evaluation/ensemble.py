"""
Per-fold ensemble of LOSO model sets + causal smoothing (no training)
=====================================================================
For every LOSO fold, averages the drowsy probabilities of that fold's models
from several model sets (e.g. v2, v3, v4). Every model in a fold was trained
without the fold's test subject, so the ensemble is still a held-out result.

All sets are mapped onto the 1 s grid of the 2 s-window datasets. A set with a
coarser step (v4: 4 s windows, 2 s step) contributes, at each 1 s epoch, its
latest window that has already ENDED by the end of that epoch (causal).

Needs the per-fold probability caches written by evaluation/postprocess.py
(<models_dir>/fold_probs.npz); run that once per model set first.

Usage:
  python -m evaluation.ensemble --sets v2:1 v3:1 v4:2 --window 45
     (name:step_seconds; models_dir = processed_data/saved_models_<name>)
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from evaluation.postprocess import causal_smooth
import evaluation.postprocess as pp


def to_1s_grid(p, step, n_target, window_target=2.0):
    """Map probabilities at `step` s (window = 2*step) onto the 1 s grid causally."""
    if step == 1:
        return p
    j = np.arange(n_target)
    # 1 s epoch j ends at j + window_target; coarse window i ends at i*step + 2*step.
    # Only the first epochs of a recording (before any coarse window has ended)
    # fall back to window 0.
    i = np.floor((j + window_target - 2 * step) / step).astype(int)
    return p[np.clip(i, 0, len(p) - 1)]


def load_sets(specs, root):
    sets = {}
    for spec in specs:
        name, step = spec.split(':') if ':' in spec else (spec, '1')
        path = Path(root) / f'saved_models_{name}' / 'fold_probs.npz'
        sets[name] = (dict(np.load(path)), float(step))
    return sets


def fold_probs(sets, names, fold, subj, n_target):
    ps = []
    for n in names:
        store, step = sets[n]
        ps.append(to_1s_grid(store[f'f{fold}_s{subj}'], step, n_target))
    return np.mean(ps, axis=0)


def evaluate(sets, ref, names, window, select_on_val=None):
    """select_on_val: list of candidate name-tuples; picks one per fold on val subjects."""
    labels, sids = ref['labels'], ref['subject_ids']
    subjects = np.unique(sids)
    pp.STEP_SECONDS = 1.0
    Y, P, per, chosen = [], [], [], []
    for fold, s in enumerate(subjects):
        use = names
        if select_on_val:
            best = -1
            for cand in select_on_val:
                yv, pv = [], []
                for v in ref[f'f{fold}_val']:
                    m = sids == v
                    yv.append(labels[m])
                    pv.append(causal_smooth(fold_probs(sets, cand, fold, v, m.sum()), window))
                acc = accuracy_score(np.concatenate(yv), np.concatenate(pv) >= 0.5)
                if acc > best:
                    best, use = acc, cand
        m = sids == s
        p = causal_smooth(fold_probs(sets, use, fold, s, m.sum()), window)
        Y.append(labels[m]); P.append(p); chosen.append(use)
        per.append(accuracy_score(labels[m], p >= 0.5))
    y, p = np.concatenate(Y), np.concatenate(P)
    pred = (p >= 0.5).astype(int)
    cm = confusion_matrix(y, pred)
    return dict(acc=accuracy_score(y, pred), f1=f1_score(y, pred, average='macro'),
                auc=roc_auc_score(y, p), mean=np.mean(per), std=np.std(per),
                alert=cm[0, 0] / cm[0].sum(), drowsy=cm[1, 1] / cm[1].sum(),
                n80=int((np.array(per) >= 0.8).sum()), per=per, chosen=chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', nargs='+', default=['v2:1', 'v3:1', 'v4:2'])
    ap.add_argument('--root', default='./processed_data')
    ap.add_argument('--window', type=float, default=45)
    args = ap.parse_args()

    sets = load_sets(args.sets, args.root)
    ref = sets[next(n for n, (_, st) in sets.items() if st == 1)][0]
    names = list(sets)
    combos = [c for r in range(1, len(names) + 1) for c in itertools.combinations(names, r)]

    print(f"Causal smoothing window: {args.window:g} s | threshold 0.5 | 1 s grid\n")
    print(f"{'Models':22s} {'Acc':>6s} {'F1':>6s} {'AUC':>6s} {'Mean±Std':>12s} "
          f"{'Alert':>6s} {'Drowsy':>6s} {'>=80%':>6s}")
    print('-' * 80)
    for raw in (True, False):
        w = 1 if raw else args.window
        print(f"[{'raw, no smoothing' if raw else f'{args.window:g} s smoothing'}]")
        for c in combos:
            r = evaluate(sets, ref, c, w)
            print(f"{'+'.join(c):22s} {r['acc']:6.4f} {r['f1']:6.4f} {r['auc']:6.4f} "
                  f"{r['mean']:.3f}±{r['std']:.3f} {r['alert']:6.3f} {r['drowsy']:6.3f} "
                  f"{r['n80']:3d}/23")
    r = evaluate(sets, ref, None, args.window, select_on_val=combos)
    print(f"{'val-selected per fold':22s} {r['acc']:6.4f} {r['f1']:6.4f} {r['auc']:6.4f} "
          f"{r['mean']:.3f}±{r['std']:.3f} {r['alert']:6.3f} {r['drowsy']:6.3f} {r['n80']:3d}/23")
    print("\nAll rows are held-out LOSO; 'val-selected' picks the model combination per fold "
          "using that fold's validation subjects only.")


if __name__ == '__main__':
    main()
