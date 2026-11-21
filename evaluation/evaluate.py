"""
Evaluation Pipeline
====================
Comprehensive model evaluation matching publication standards.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from pathlib import Path
import sys
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from evaluation.models.bci_model import build_model
from training.trainer import EEGDataset


def evaluate_loso(data_path, models_dir, device='cpu'):
    """
    Evaluate the LOSO-CV trained models.
    Loads each fold's model and evaluates on its held-out subject.
    """
    data = np.load(data_path)
    epochs      = data['epochs']
    labels      = data['labels']
    subject_ids = data['subject_ids']

    unique_subjects = np.unique(subject_ids)
    models_dir = Path(models_dir)

    all_preds, all_labels, all_probs = [], [], []
    per_subject_acc = {}

    for fold, test_subj in enumerate(unique_subjects):
        model_path = models_dir / f'fold_{fold+1}_subject_{test_subj}.pth'
        if not model_path.exists():
            print(f"Skip fold {fold+1} - no model file")
            continue

        model = build_model(n_channels=5).to(device)
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        model.eval()

        test_mask = subject_ids == test_subj
        X_test = epochs[test_mask]
        y_test = labels[test_mask]

        test_ds = EEGDataset(X_test, y_test)
        loader  = DataLoader(test_ds, batch_size=128, shuffle=False)

        preds, probs = [], []
        with torch.no_grad():
            for X, y in loader:
                X = X.to(device)
                logits = model(X)
                p = torch.softmax(logits, dim=1).cpu().numpy()
                probs.extend(p)
                preds.extend(p.argmax(1).tolist())

        acc = accuracy_score(y_test, preds)
        per_subject_acc[test_subj] = acc
        all_preds.extend(preds)
        all_labels.extend(y_test.tolist())
        all_probs.extend(probs)
        print(f"Subject {test_subj:2d}: acc={acc:.4f}")

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    print(f"\n{'='*50}")
    print(f"LOSO-CV FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Overall Accuracy:  {accuracy_score(all_labels, all_preds):.4f}")
    print(f"F1 Score (macro):  {f1_score(all_labels, all_preds, average='macro'):.4f}")
    print(f"AUC-ROC:           {roc_auc_score(all_labels, all_probs[:, 1]):.4f}")
    print(f"\nPer-subject accuracy:")
    for subj, acc in sorted(per_subject_acc.items()):
        bar = '█' * int(acc * 30)
        print(f"  Subject {subj:2d}: {bar:<30s} {acc:.3f}")
    print(f"\nMean ± Std: {np.mean(list(per_subject_acc.values())):.4f} ± {np.std(list(per_subject_acc.values())):.4f}")

    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(f"              Predicted")
    print(f"              Focused  Relaxed")
    print(f"Actual Focused  {cm[0,0]:5d}    {cm[0,1]:5d}")
    print(f"Actual Relaxed  {cm[1,0]:5d}    {cm[1,1]:5d}")

    # Saved next to the models so evaluating another model set never overwrites it
    np.savez(str(models_dir / 'roc_data.npz'), y_true=all_labels, y_prob=all_probs[:, 1])

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='macro'),
        'auc': roc_auc_score(all_labels, all_probs[:, 1]),
        'per_subject': per_subject_acc,
        'confusion_matrix': cm
    }


def noise_robustness_test(data_path, models_dir, device='cpu', seed=0,
                          noise_levels=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5)):
    """
    Test model robustness at different noise levels.
    Simulates increasingly noisy consumer hardware.

    Each subject is scored ONLY by its own LOSO fold model (the one that never
    saw that subject during training), so the numbers are held-out results
    directly comparable to evaluate_loso() at noise 0.
    """
    data = np.load(data_path)
    epochs      = data['epochs']
    labels      = data['labels']
    subject_ids = data['subject_ids']
    unique_subjects = np.unique(subject_ids)
    models_dir = Path(models_dir)

    # Noise is scaled by the per-channel std of the whole dataset
    # (≈1 after per-epoch z-scoring)
    signal_std = epochs.std(axis=(0, 1), keepdims=True)

    preds = {noise: [] for noise in noise_levels}
    all_y = []

    print("\nNoise Robustness Test (per-fold, held-out subjects only):")
    print("─" * 40)

    n_used = 0
    for fold, test_subj in enumerate(unique_subjects):
        model_path = models_dir / f'fold_{fold+1}_subject_{test_subj}.pth'
        if not model_path.exists():
            print(f"Skip fold {fold+1} - no model file")
            continue
        n_used += 1

        model = build_model(n_channels=5).to(device)
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        model.eval()

        mask = subject_ids == test_subj
        X_subj, y_subj = epochs[mask], labels[mask]
        all_y.extend(y_subj.tolist())

        for noise in noise_levels:
            # Same noise realisation per (subject, level) on every run
            rng = np.random.default_rng([seed, int(test_subj), int(noise * 1000)])
            X_noisy = X_subj
            if noise > 0:
                X_noisy = (X_subj + noise * signal_std *
                           rng.standard_normal(X_subj.shape)).astype(np.float32)

            loader = DataLoader(EEGDataset(X_noisy, y_subj), batch_size=256, shuffle=False)
            with torch.no_grad():
                for X, _ in loader:
                    preds[noise].extend(model(X.to(device)).argmax(1).cpu().numpy())

    if n_used == 0:
        raise FileNotFoundError(f"No fold models found in {models_dir}")

    results = {}
    for noise in noise_levels:
        acc = accuracy_score(all_y, preds[noise])
        results[noise] = acc
        snr_db = -20 * np.log10(noise) if noise > 0 else float('inf')
        print(f"  Noise {noise:.0%} (SNR≈{snr_db:4.0f}dB): {acc:.4f}")
    print(f"  ({n_used} fold models, {len(all_y):,} held-out epochs)")

    return results
