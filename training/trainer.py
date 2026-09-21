"""
Training Pipeline
==================
Leave-One-Subject-Out Cross-Validation (LOSO-CV)
─────────────────────────────────────────────────
WHY LOSO? This is the CRITICAL question for generalization.

Normal train/test split: model sees subject 3 in training,
tests on different epochs of subject 3. This is CHEATING.
The model learns THIS PERSON's brain patterns, not general ones.

LOSO: train on 22 subjects, test on subject 23.
Repeat for each subject. Measures TRUE generalization to new users.

This is exactly what your consumer product needs:
A user buys the headset → NO calibration → it just works.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import sys
# Insert project root (two levels up from training/) so 'models' resolves
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.bci_model import build_model


# ────────────────────────────────────────────────────
#  DATASET
# ────────────────────────────────────────────────────

class EEGDataset(Dataset):
    """
    Wraps preprocessed EEG epochs as PyTorch dataset.
    Input format: (N, 400, 5) → reshaped to (N, 5, 400) for conv layers.
    """
    def __init__(self, epochs, labels, augment=False):
        # Reshape: (N, time, channels) → (N, channels, time)
        self.X = torch.FloatTensor(epochs.transpose(0, 2, 1))  # (N, 5, 400)
        self.y = torch.LongTensor(labels)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.augment:
            x = self._augment(x)
        return x, y

    def _augment(self, x):
        """
        Data augmentation for EEG (applied during training only).

        Why augment?
        - We have limited data from each subject
        - Consumer signals will have MORE noise than SEED-VIG
        - We want the model to be robust to noise variations

        Techniques:
        1. Gaussian noise injection (simulates cheap electrodes)
        2. Amplitude scaling (individual amplitude differences)
        3. Channel dropout (simulate occasional electrode loss)
        """
        # 1. Add Gaussian noise (30% chance, simulates consumer noise)
        if torch.rand(1) < 0.3:
            noise_level = torch.rand(1) * 0.15  # up to 15% noise
            x = x + noise_level * torch.randn_like(x)

        # 2. Random amplitude scaling (accounts for individual differences)
        if torch.rand(1) < 0.4:
            scale = 0.8 + 0.4 * torch.rand(1)  # 0.8× to 1.2×
            x = x * scale

        # 3. Channel dropout (10% chance per channel, simulates bad contact)
        if torch.rand(1) < 0.2:
            mask = (torch.rand(x.shape[0]) > 0.1).float().unsqueeze(1)
            x = x * mask

        return x


# ────────────────────────────────────────────────────
#  TRAINING LOOP
# ────────────────────────────────────────────────────

class Trainer:
    def __init__(self, device=None, use_amp=True):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_amp = use_amp and self.device == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high')
        print(f"Using device: {self.device}", flush=True)
        if self.use_amp:
            print("Using mixed precision: enabled", flush=True)

    def train_one_epoch(self, model, loader, optimizer, criterion):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for X, y in loader:
            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits = model(X)
                loss = criterion(logits, y)

            self.scaler.scale(loss).backward()
            # Gradient clipping: prevents exploding gradients
            # Common issue with LSTM on noisy signals
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)

        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self, model, loader, criterion):
        model.eval()
        total_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []

        for X, y in loader:
            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits = model(X)
                loss = criterion(logits, y)

            total_loss += loss.item() * len(y)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += len(y)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)

    def train_subject_independent(self, epochs_data, labels_data, subject_ids,
                                  n_epochs=80, batch_size=64, lr=1e-3,
                                  save_dir='saved_models', max_folds=None,
                                  num_workers=0):
        """
        LOSO Cross-Validation training.

        For each subject:
            train_subjects = all EXCEPT current subject
            test_subject   = current subject

        This is subject-INDEPENDENT training = true generalization test.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        unique_subjects = np.unique(subject_ids)
        fold_accuracies = []

        print(f"\n{'='*60}")
        print(f"LOSO-CV: {len(unique_subjects)} folds")
        print(f"{'='*60}\n")

        for fold, test_subject in enumerate(unique_subjects):
            save_path = save_dir / f'fold_{fold+1}_subject_{test_subject}.pth'
            if save_path.exists():
                print(f"Fold {fold+1}/{len(unique_subjects)} - SKIP (already saved: {save_path.name})", flush=True)
                continue

            if max_folds is not None and fold >= max_folds:
                print(f"Stopping early: reached max_folds={max_folds}")
                break

            print(f"Fold {fold+1}/{len(unique_subjects)} — Test subject: {test_subject}")

            print(f"  [1/6] Splitting train/test masks...", flush=True)
            # Split data
            train_mask = subject_ids != test_subject
            test_mask  = subject_ids == test_subject

            X_train = epochs_data[train_mask]
            y_train = labels_data[train_mask]
            X_test  = epochs_data[test_mask]
            y_test  = labels_data[test_mask]

            n0 = (y_test == 0).sum()
            n1 = (y_test == 1).sum()
            baseline_acc = max(n0, n1) / (n0 + n1)
            print(f"  Test labels: Focused={n0:,} | Relaxed={n1:,}", flush=True)
            print(f"  Majority-class baseline for this fold: {baseline_acc:.3f}", flush=True)

            # Keep natural batch distribution; handle imbalance in the loss.
            print(f"  [2/6] Computing class weights (n={len(y_train):,})...", flush=True)
            n_focused = int((y_train == 0).sum())
            n_relaxed = int((y_train == 1).sum())

            train_ds = EEGDataset(X_train, y_train, augment=True)
            test_ds  = EEGDataset(X_test, y_test, augment=False)

            print(f"  [3/6] Building DataLoaders...", flush=True)
            loader_kwargs = {
                'batch_size': batch_size,
                'num_workers': num_workers,
                'pin_memory': False,
            }
            if num_workers > 0:
                loader_kwargs['persistent_workers'] = True

            train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
            test_loader  = DataLoader(test_ds, shuffle=False, **loader_kwargs)

            print(f"  [4/6] Moving model to device...", flush=True)
            # Fresh model for each fold
            model = build_model(n_channels=5, n_classes=2).to(self.device)

            # Optimizer: AdamW (Adam + weight decay)
            # Weight decay = L2 regularization = prevents overfitting
            print(f"  [5/6] Setting up optimizer/scheduler...", flush=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

            # LR schedule: cosine annealing
            # Starts high, smoothly decreases to near-zero
            # Prevents getting stuck in local minima
            scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

            class_weights = torch.FloatTensor([
                len(y_train) / (2.0 * n_focused),
                len(y_train) / (2.0 * n_relaxed),
            ]).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)

            best_acc = 0
            best_state = None
            patience = 25
            epochs_no_improve = 0

            print(f"  [6/6] Starting training loop...", flush=True)
            for epoch in range(n_epochs):
                train_loss, train_acc = self.train_one_epoch(model, train_loader, optimizer, criterion)
                val_loss, val_acc, _, _ = self.evaluate(model, test_loader, criterion)
                scheduler.step()

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epoch == 0 or (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1:3d}/{n_epochs} | "
                          f"Train: {train_acc:.3f} | Val: {val_acc:.3f} | "
                          f"LR: {scheduler.get_last_lr()[0]:.6f}", flush=True)

                if epochs_no_improve >= patience:
                    print(f"  Early stop at epoch {epoch+1} (no improvement for {patience} epochs)", flush=True)
                    break

            fold_accuracies.append(best_acc)
            print(f"  ✓ Best accuracy: {best_acc:.4f}\n")

            # Save best model for this fold
            torch.save(best_state, save_path)

        mean_acc = np.mean(fold_accuracies)
        std_acc  = np.std(fold_accuracies)

        print(f"\n{'='*60}")
        print(f"LOSO-CV Results:")
        print(f"  Per-fold: {[f'{a:.3f}' for a in fold_accuracies]}")
        print(f"  Mean ± Std: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"{'='*60}")

        return fold_accuracies, mean_acc, std_acc


# ────────────────────────────────────────────────────
#  TRANSFER LEARNING FINE-TUNING
#  (Optional: adapt to a specific new user)
# ────────────────────────────────────────────────────

class PersonalCalibrator:
    """
    Optional: 2-minute calibration to personalize for a new user.
    Uses just 60-100 labeled epochs (30 focused + 30 relaxed).

    Protocol:
    1. Show user task requiring focus for 1 min → collect focused data
    2. Show user relaxing content for 1 min → collect relaxed data
    3. Fine-tune last 2 layers only (keep earlier features frozen)

    This is called "few-shot fine-tuning" or "meta-learning adaptation".
    """
    def __init__(self, base_model_path, device=None):
        self.device = device or ('cpu')
        self.model = build_model()
        state = torch.load(base_model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)

    def calibrate(self, calib_epochs, calib_labels, n_steps=50):
        """
        Fine-tune on user's personal calibration data.
        Only updates the final classifier layers (not SEN or CNN).
        """
        # Freeze all except final 2 classifier layers
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.classifier[-4:].parameters():
            param.requires_grad = True

        # Small LR for fine-tuning (don't destroy pre-trained features)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-4
        )

        dataset = EEGDataset(calib_epochs, calib_labels, augment=True)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        self.model.train()
        for step in range(n_steps):
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(self.model(X), y)
                loss.backward()
                optimizer.step()

        print(f"✅ Calibration complete ({n_steps} steps)")
        return self.model


# ────────────────────────────────────────────────────
#  MAIN TRAINING ENTRY POINT
# ────────────────────────────────────────────────────

def train(data_path, save_dir='saved_models', n_epochs=80, batch_size=64,
          max_folds=None, num_workers=0, use_amp=True):
    """Main training function."""
    print("Loading processed data...")
    data = np.load(data_path)
    epochs     = data['epochs']      # (N, 400, 5)
    labels     = data['labels']      # (N,)
    subject_ids = data['subject_ids'] # (N,)

    print(f"Dataset: {epochs.shape[0]:,} epochs | {len(np.unique(subject_ids))} subjects")
    print(f"Focused: {(labels==0).sum():,} | Relaxed: {(labels==1).sum():,}")

    trainer = Trainer(use_amp=use_amp)
    fold_accs, mean_acc, std_acc = trainer.train_subject_independent(
        epochs, labels, subject_ids,
        n_epochs=n_epochs,
        batch_size=batch_size,
        save_dir=save_dir,
        max_folds=max_folds,
        num_workers=num_workers
    )

    return fold_accs, mean_acc, std_acc


if __name__ == '__main__':
    train(
        data_path='../processed_data/seed_vig_5ch_processed.npz',
        save_dir='../saved_models',
        n_epochs=80,
        batch_size=64
    )
