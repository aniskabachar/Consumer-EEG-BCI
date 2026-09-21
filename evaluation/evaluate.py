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

from models.bci_model import build_model
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

    np.savez('roc_data.npz', y_true=all_labels, y_prob=all_probs[:, 1])

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='macro'),
        'auc': roc_auc_score(all_labels, all_probs[:, 1]),
        'per_subject': per_subject_acc,
        'confusion_matrix': cm
    }


def noise_robustness_test(data_path, model_path, device='cpu'):
    """
    Test model robustness at different noise levels.
    Simulates increasingly noisy consumer hardware.
    """
    data = np.load(data_path)
    epochs = data['epochs']
    labels = data['labels']

    model = build_model(n_channels=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]  # fraction of signal std
    results = {}

    print("\nNoise Robustness Test:")
    print("─" * 40)

    for noise in noise_levels:
        noisy_epochs = epochs.copy()
        if noise > 0:
            # Add Gaussian noise scaled to signal standard deviation
            signal_std = epochs.std(axis=(0, 1), keepdims=True)
            noisy_epochs = epochs + noise * signal_std * np.random.randn(*epochs.shape)

        ds = EEGDataset(noisy_epochs, labels)
        loader = DataLoader(ds, batch_size=256, shuffle=False)

        preds, all_y = [], []
        with torch.no_grad():
            for X, y in loader:
                X = X.to(device)
                preds.extend(model(X).argmax(1).cpu().numpy())
                all_y.extend(y.numpy())

        acc = accuracy_score(all_y, preds)
        results[noise] = acc
        snr_db = -20 * np.log10(noise) if noise > 0 else float('inf')
        print(f"  Noise {noise:.0%} (SNR≈{snr_db:4.0f}dB): {acc:.4f}")

    return results
