"""
Generate all paper figures from evaluation results.
Run AFTER: python main.py evaluate ...
Then fill in the FILL_ values below from your terminal output.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os, sys

os.makedirs('paper_figures', exist_ok=True)

# ── Load ROC data (saved automatically by evaluate.py) ────────────
try:
    roc      = np.load('roc_data.npz')
    baseline = np.load('baseline_roc_data.npz')
    has_baseline = True
except FileNotFoundError:
    has_baseline = False
    print("WARNING: baseline_roc_data.npz not found, skipping baseline on ROC")

# ── FIGURE 1: ROC Curve ───────────────────────────────────────────
fpr1, tpr1, _ = roc_curve(roc['y_true'], roc['y_prob'])
auc1 = auc(fpr1, tpr1)

plt.figure(figsize=(7, 6))
plt.plot(fpr1, tpr1, 'b-', lw=2.5, label=f'SEN-CNN-LSTM  (AUC = {auc1:.3f})')
if has_baseline:
    fpr2, tpr2, _ = roc_curve(baseline['y_true'], baseline['y_prob'])
    auc2 = auc(fpr2, tpr2)
    plt.plot(fpr2, tpr2, 'r--', lw=2, label=f'Baseline ShallowCNN  (AUC = {auc2:.3f})')
plt.plot([0,1],[0,1],'k--', lw=1, alpha=0.4, label='Random (AUC = 0.500)')
plt.fill_between(fpr1, tpr1, alpha=0.08, color='blue')
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=13)
plt.title('ROC Curve — LOSO-CV (23 Subjects)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('paper_figures/fig1_roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: paper_figures/fig1_roc_curve.png")

# ── FIGURE 2: Confusion Matrix ────────────────────────────────────
# TODO: Fill these 4 values from your evaluate output
TN = 0   # Actual Alert,  Predicted Alert
FP = 0   # Actual Alert,  Predicted Drowsy
FN = 0   # Actual Drowsy, Predicted Alert
TP = 0   # Actual Drowsy, Predicted Drowsy

if TN + FP + FN + TP > 0:
    cm = np.array([[TN, FP], [FN, TP]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Alert (Pred)', 'Drowsy (Pred)'], fontsize=11)
    ax.set_yticklabels(['Alert (True)', 'Drowsy (True)'], fontsize=11)
    ax.set_title('Confusion Matrix (LOSO-CV)', fontsize=13, fontweight='bold')
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i,j] > cm.max()*0.5 else 'black'
            ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center', fontsize=15, color=color, fontweight='bold')
    plt.colorbar(im, shrink=0.8)
    plt.tight_layout()
    plt.savefig('paper_figures/fig2_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: paper_figures/fig2_confusion_matrix.png")
else:
    print("SKIP Fig2: Fill in TN/FP/FN/TP values from evaluate output first")

# ── FIGURE 3: Per-Subject Accuracy ───────────────────────────────
# TODO: Fill these 23 accuracy values from your evaluate output
per_subject_accs = {
    0: 0.0,  1: 0.0,  2: 0.0,  3: 0.0,  4: 0.0,
    5: 0.0,  6: 0.0,  7: 0.0,  8: 0.0,  9: 0.0,
   10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0,
   15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0,
   20: 0.0, 21: 0.0, 22: 0.0,
}

if any(v > 0 for v in per_subject_accs.values()):
    subjects = sorted(per_subject_accs.keys())
    accs_list = [per_subject_accs[s] for s in subjects]
    mean_acc  = np.mean(accs_list)
    colors = ['#e74c3c' if a < 0.70 else ('#f39c12' if a < 0.80 else '#27ae60') for a in accs_list]

    plt.figure(figsize=(14, 5))
    bars = plt.bar(subjects, accs_list, color=colors, edgecolor='black', linewidth=0.6, width=0.7)
    plt.axhline(y=mean_acc, color='navy', linestyle='--', lw=2, label=f'Mean = {mean_acc:.3f} ({mean_acc*100:.1f}%)')
    plt.axhline(y=0.70, color='orange', linestyle=':', lw=1.5, alpha=0.7, label='70% threshold')
    plt.xlabel('Subject ID', fontsize=13)
    plt.ylabel('Classification Accuracy', fontsize=13)
    plt.title('Per-Subject Accuracy — LOSO Cross-Validation', fontsize=14, fontweight='bold')
    plt.ylim(0.4, 1.02)
    plt.xticks(subjects, [f'S{i}' for i in subjects], fontsize=9)
    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ['50%','60%','70%','80%','90%','100%'])
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    for bar, acc in zip(bars, accs_list):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                 f'{acc:.2f}', ha='center', va='bottom', fontsize=7, rotation=90)
    plt.tight_layout()
    plt.savefig('paper_figures/fig3_per_subject_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: paper_figures/fig3_per_subject_accuracy.png")
else:
    print("SKIP Fig3: Fill in per_subject_accs dict from evaluate output first")

# ── FIGURE 4: Noise Robustness ────────────────────────────────────
# TODO: Fill these from noise test output
noise_levels  = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
noise_accs    = [0.0,  0.0, 0.0, 0.0, 0.0, 0.0]   # <-- fill these in

if any(v > 0 for v in noise_accs):
    plt.figure(figsize=(7, 5))
    plt.plot([n*100 for n in noise_levels], noise_accs,
             'bo-', lw=2.5, ms=9, markerfacecolor='white', markeredgewidth=2,
             label='SEN-CNN-LSTM')
    plt.axhline(0.5, color='gray', linestyle=':', lw=1.5, label='Chance level (50%)')
    plt.fill_between([n*100 for n in noise_levels], noise_accs, 0.5,
                     alpha=0.08, color='blue')
    plt.xlabel('Added Noise Level (%)', fontsize=13)
    plt.ylabel('Classification Accuracy', fontsize=13)
    plt.title('Noise Robustness — Signal Enhancement Network', fontsize=13, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.ylim(0.4, 1.05)
    plt.xticks([0, 5, 10, 20, 30, 50])
    plt.tight_layout()
    plt.savefig('paper_figures/fig4_noise_robustness.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: paper_figures/fig4_noise_robustness.png")
else:
    print("SKIP Fig4: Fill in noise_accs list from noise test output first")

print("\nDone! Check paper_figures/ folder")
