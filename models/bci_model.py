"""
EEG BCI Model Architecture
============================

The model has TWO parts working together:

Part 1: SEN (Signal Enhancement Network)
─────────────────────────────────────────
Consumer headsets produce NOISY signals (cheap electrodes,
no gel, motion artifacts). SEN learns to DENOISE them.

Think of it like: cheap camera → SEN → DSLR-quality image
The model learns what "real" EEG looks like vs noise.

Architecture: Encoder → Bottleneck → Decoder (U-Net style)
Input:  (batch, 5, 400)  ← 5 channels × 400 time samples
Output: (batch, 5, 400)  ← same shape, but ENHANCED

Part 2: AttentionCNNLSTM (Classifier)
──────────────────────────────────────
Takes enhanced signal → predicts Focused (0) or Relaxed (1)

CNN layers extract spatial+frequency features from each time window.
LSTM captures temporal patterns (how attention changes over time).
Attention mechanism focuses on the most informative time steps.

Full pipeline:
  Raw 5ch EEG → SEN → Enhanced EEG → CNN → LSTM → Attention → Label
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════
#  PART 1: SIGNAL ENHANCEMENT NETWORK (SEN)
#  Purpose: Denoise consumer-grade EEG signals
# ═══════════════════════════════════════════════════

class SpatialFilter(nn.Module):
    """
    Learn which channel combinations carry attention information.
    Like a learned re-weighting of the 5 electrode signals.

    Consumer headsets have poor electrode placement consistency.
    Spatial filtering compensates for this.
    """
    def __init__(self, n_channels=5):
        super().__init__()
        # Depthwise + pointwise convolution = spatial filter
        self.spatial_conv = nn.Conv1d(n_channels, n_channels, kernel_size=1, groups=1, bias=False)
        self.bn = nn.BatchNorm1d(n_channels)

    def forward(self, x):
        # x: (batch, channels, time)
        return F.elu(self.bn(self.spatial_conv(x)))


class TemporalEncoder(nn.Module):
    """
    Extract temporal patterns at multiple scales.
    Different kernel sizes capture different EEG rhythms:
    - kernel 4  → ~20ms patterns  (gamma-like)
    - kernel 16 → ~80ms patterns  (beta-like)
    - kernel 32 → ~160ms patterns (alpha-like)
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_small  = nn.Conv1d(in_ch, out_ch//3, kernel_size=4,  padding=2)
        self.conv_medium = nn.Conv1d(in_ch, out_ch//3, kernel_size=16, padding=8)
        self.conv_large  = nn.Conv1d(in_ch, out_ch - 2*(out_ch//3), kernel_size=32, padding=16)
        self.bn = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        s = self.conv_small(x)
        m = self.conv_medium(x)
        l = self.conv_large(x)
        # Align time dimensions (padding may differ by 1)
        min_len = min(s.shape[2], m.shape[2], l.shape[2])
        out = torch.cat([s[..., :min_len], m[..., :min_len], l[..., :min_len]], dim=1)
        out = F.elu(self.bn(out))
        return self.pool(out)


class SEN(nn.Module):
    """
    Signal Enhancement Network
    ────────────────────────────
    U-Net style encoder-decoder for EEG denoising.
    Skip connections preserve fine-grained temporal structure.

    Why U-Net? EEG signals have multi-scale structure:
    - Fast transients (gamma) live in high-freq detail
    - Slow rhythms (alpha/theta) are in low-freq content
    Skip connections let BOTH scales reach the output.

    Input/Output: (batch, 5, 400)
    Internal: expand to 32ch → 64ch → 32ch → 5ch
    """
    def __init__(self, n_channels=5, seq_len=400):
        super().__init__()

        # Encoder: compress temporal resolution, expand features
        self.spatial_filter = SpatialFilter(n_channels)

        self.enc1 = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ELU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ELU(),
        )
        self.pool1 = nn.MaxPool1d(2, return_indices=True)  # 400 → 200

        self.enc2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ELU(),
        )
        self.pool2 = nn.MaxPool1d(2, return_indices=True)  # 200 → 100

        # Bottleneck: deepest compression
        self.bottleneck = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ELU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ELU(),
        )

        # Decoder: expand back, skip connections from encoder
        self.unpool2 = nn.MaxUnpool1d(2)
        self.dec2 = nn.Sequential(
            nn.Conv1d(64 + 64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ELU(),
        )

        self.unpool1 = nn.MaxUnpool1d(2)
        self.dec1 = nn.Sequential(
            nn.Conv1d(32 + 32, n_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(n_channels), nn.ELU(),
            nn.Conv1d(n_channels, n_channels, kernel_size=1),
        )

        # Residual connection: enhanced = SEN(x) + x
        # This means: learn the NOISE to subtract, not the signal to produce
        # Same trick used in ResNet and denoising diffusion models

    def forward(self, x):
        # x: (batch, 5, 400)
        residual = x

        x = self.spatial_filter(x)

        # Encode
        e1 = self.enc1(x)                          # (B, 32, 400)
        e1_pooled, idx1 = self.pool1(e1)            # (B, 32, 200)

        e2 = self.enc2(e1_pooled)                   # (B, 64, 200)
        e2_pooled, idx2 = self.pool2(e2)            # (B, 64, 100)

        # Bottleneck
        b = self.bottleneck(e2_pooled)              # (B, 64, 100)

        # Decode with skip connections
        d2 = self.unpool2(b, idx2, output_size=e2.shape)   # (B, 64, 200)
        d2 = torch.cat([d2, e2], dim=1)                     # (B, 128, 200)
        d2 = self.dec2(d2)                                   # (B, 32, 200)

        d1 = self.unpool1(d2, idx1, output_size=e1.shape)   # (B, 32, 400)
        d1 = torch.cat([d1, e1], dim=1)                     # (B, 64, 400)
        d1 = self.dec1(d1)                                   # (B, 5, 400)

        # Residual: model learns noise pattern, we subtract it
        enhanced = residual + d1
        return enhanced


# ═══════════════════════════════════════════════════
#  PART 2: ATTENTION CLASSIFIER (CNN-LSTM + Attention)
#  Purpose: Focused vs Relaxed classification
# ═══════════════════════════════════════════════════

class EEGNetBlock(nn.Module):
    """
    EEGNet-inspired spatial-temporal convolution block.
    References the famous EEGNet architecture (Lawhern 2018)
    adapted for our 5-channel, 400-sample input.

    WHY EEGNet style?
    - Designed specifically for EEG (not repurposed from image CNNs)
    - Works well with small datasets (few parameters)
    - Separable convolutions = efficient
    """
    def __init__(self, n_channels=5, seq_len=400, F1=8, D=2, F2=16):
        super().__init__()
        # Temporal convolution (across time, per channel)
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
        )
        # Depthwise spatial convolution (across channels, per feature map)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(F1, F1*D, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.25),
        )
        # Separable convolution (pointwise)
        self.separable_conv = nn.Sequential(
            nn.Conv2d(F1*D, F2, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(0.25),
        )
        self.F2 = F2

    def forward(self, x):
        # x: (batch, channels, time) → add 2D dim
        x = x.unsqueeze(1)               # (B, 1, 5, 400)
        x = self.temporal_conv(x)        # (B, F1, 5, ~400)
        x = self.spatial_conv(x)         # (B, F2, 1, ~100)
        x = self.separable_conv(x)       # (B, F2, 1, ~12)
        x = x.squeeze(2)                 # (B, F2, ~12)
        return x                         # (B, 16, 12) = (batch, features, time_steps)


class TemporalAttention(nn.Module):
    """
    Self-attention over time steps.
    Lets the model focus on the MOST INFORMATIVE moments in the 2-sec window.

    For attention detection:
    - A brief alpha burst might be the key signal
    - Model learns to upweight that moment
    - Downweight irrelevant parts (blinks, transitions)
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        scores = self.attention(lstm_output)       # (B, T, 1)
        weights = F.softmax(scores, dim=1)         # (B, T, 1)
        context = (weights * lstm_output).sum(1)   # (B, hidden_size)
        return context, weights


class AttentionClassifier(nn.Module):
    """
    CNN-LSTM with attention for attention/relaxed classification.

    Flow: EEGNet features → LSTM (temporal dynamics) → Attention → FC → label

    Why LSTM after CNN?
    CNN sees local patterns. LSTM sees HOW patterns evolve over time.
    2-second window has ~200ms alpha cycles — LSTM tracks their rhythm.
    """
    def __init__(self, n_channels=5, n_classes=2, hidden_size=64):
        super().__init__()

        self.eegnet = EEGNetBlock(n_channels=n_channels)

        # Bidirectional LSTM: reads signal forward AND backward
        # Forward: normal time order
        # Backward: useful because EEG patterns are non-causal
        self.lstm = nn.LSTM(
            input_size=self.eegnet.F2,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.attention = TemporalAttention(hidden_size * 2)  # *2 for bidirectional

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x, return_attention=False):
        # x: (batch, 5, 400)

        # CNN feature extraction
        cnn_out = self.eegnet(x)          # (B, 16, 12)
        cnn_out = cnn_out.permute(0, 2, 1)  # (B, 12, 16) ← LSTM expects (B, T, features)

        # Temporal modeling
        lstm_out, _ = self.lstm(cnn_out)   # (B, 12, 128)

        # Attention pooling
        context, attn_weights = self.attention(lstm_out)  # (B, 128)

        # Classification
        logits = self.classifier(context)  # (B, 2)

        if return_attention:
            return logits, attn_weights
        return logits


# ═══════════════════════════════════════════════════
#  FULL MODEL: SEN + Classifier
# ═══════════════════════════════════════════════════

class BCIModel(nn.Module):
    """
    Complete BCI Model = SEN + AttentionClassifier

    At training time: both components train together end-to-end.
    The SEN learns to produce signals that HELP the classifier.

    At inference (real-time):
    raw_signal → SEN.forward() → AttentionClassifier.forward() → {focused, relaxed}
    """

    def __init__(self, n_channels=5, n_classes=2, hidden_size=64):
        super().__init__()
        self.sen = SEN(n_channels=n_channels)
        self.classifier = AttentionClassifier(n_channels=n_channels, n_classes=n_classes, hidden_size=hidden_size)

    def forward(self, x, return_attention=False):
        # x: (batch, 5, 400)
        enhanced = self.sen(x)              # Denoise signal
        out = self.classifier(enhanced, return_attention=return_attention)
        return out

    def get_enhanced_signal(self, x):
        """For visualization: show before/after SEN."""
        with torch.no_grad():
            return self.sen(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(n_channels=5, n_classes=2):
    """Factory function to create the model."""
    model = BCIModel(n_channels=n_channels, n_classes=n_classes)
    params = model.count_parameters()
    print(f"✅ Model created: {params:,} trainable parameters")
    print(f"   SEN params:        {sum(p.numel() for p in model.sen.parameters()):,}")
    print(f"   Classifier params: {sum(p.numel() for p in model.classifier.parameters()):,}")
    return model


if __name__ == '__main__':
    # Quick smoke test
    model = build_model()
    x = torch.randn(8, 5, 400)  # batch=8, channels=5, time=400
    out = model(x)
    print(f"\n✅ Forward pass OK")
    print(f"   Input:  {x.shape}")
    print(f"   Output: {out.shape}  (8 samples × 2 classes)")
    print(f"   Memory: ~{sum(p.numel()*4 for p in model.parameters())/1024/1024:.1f} MB")
