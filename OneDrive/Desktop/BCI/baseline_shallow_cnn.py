"""
Baseline: Shallow CNN (standard approach, no SEN / no BiLSTM / no attention)
=============================================================================
Same data, same LOSO-CV protocol as evaluate.py, but a minimal architecture:
2 conv layers -> global pool -> 1 dense layer -> softmax.

Run:
    python baseline_shallow_cnn.py --data_path .\processed_data\seed_vig_5ch_processed.npz

Output:
    Per-subject accuracy, overall accuracy/F1/AUC, printed same format as evaluate.py
    Saves baseline_roc_data.npz for a real ROC comparison figure.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


class EEGDatasetSimple(Dataset):
    def __init__(self, X, y):
        # X: (N, 400, 5) -> model wants (N, 5, 400) for Conv1d
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ShallowCNN(nn.Module):
    """Standard/minimal baseline: no SEN, no BiLSTM, no attention."""
    def __init__(self, n_channels=5, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.squeeze(-1)
        return self.fc(x)


def run_loso_baseline(data_path, device='cuda' if torch.cuda.is_available() else 'cpu',
                       epochs=15, batch_size=256, lr=1e-3):
    print(f"Using device: {device}")
    data = np.load(data_path)
    X_all = data['epochs']
    y_all = data['labels']
    subject_ids = data['subject_ids']
    unique_subjects = np.unique(subject_ids)

    all_preds, all_labels, all_probs = [], [], []
    per_subject_acc = {}

    for test_subj in unique_subjects:
        train_mask = subject_ids != test_subj
        test_mask = subject_ids == test_subj

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        train_loader = DataLoader(EEGDatasetSimple(X_train, y_train),
                                   batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(EEGDatasetSimple(X_test, y_test),
                                  batch_size=batch_size, shuffle=False)

        model = ShallowCNN(n_channels=X_all.shape[2]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()

        model.train()
        for ep in range(epochs):
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                opt.zero_grad()
                loss = crit(model(X), y)
                loss.backward()
                opt.step()

        model.eval()
        preds, probs = [], []
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                logits = model(X)
                p = torch.softmax(logits, dim=1).cpu().numpy()
                probs.extend(p)
                preds.extend(p.argmax(1).tolist())

        acc = accuracy_score(y_test, preds)
        per_subject_acc[int(test_subj)] = acc
        all_preds.extend(preds)
        all_labels.extend(y_test.tolist())
        all_probs.extend(probs)
        print(f"Subject {int(test_subj):2d}: acc={acc:.4f}")

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    overall_acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    auc = roc_auc_score(all_labels, all_probs[:, 1])
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n{'='*50}")
    print("SHALLOW CNN BASELINE - LOSO-CV FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Overall Accuracy:  {overall_acc:.4f}")
    print(f"F1 Score (macro):  {f1:.4f}")
    print(f"AUC-ROC:           {auc:.4f}")
    print(f"\nMean ± Std per-subject: {np.mean(list(per_subject_acc.values())):.4f} "
          f"± {np.std(list(per_subject_acc.values())):.4f}")
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Focused  Relaxed")
    print(f"Actual Focused  {cm[0,0]:5d}    {cm[0,1]:5d}")
    print(f"Actual Relaxed  {cm[1,0]:5d}    {cm[1,1]:5d}")

    np.savez('baseline_roc_data.npz', y_true=all_labels, y_prob=all_probs[:, 1])
    print("\nSaved baseline_roc_data.npz")

    return {'accuracy': overall_acc, 'f1': f1, 'auc': auc,
            'per_subject': per_subject_acc, 'confusion_matrix': cm}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=15)
    args = parser.parse_args()
    run_loso_baseline(args.data_path, epochs=args.epochs)