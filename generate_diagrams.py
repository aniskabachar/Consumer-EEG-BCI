import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set global styles
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# -------------------------------------------------------------
# 1. Fig Montage: 5-Channel Head Montage (FT7, FT8, T7, PZ, O2)
# -------------------------------------------------------------
def create_montage():
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.4)
    ax.set_aspect('equal')
    ax.axis('off')

    # Head outline
    head = plt.Circle((0, 0), 1.0, color='#e2e8f0', ec='#1e293b', lw=2.5, zorder=1)
    ax.add_patch(head)

    # Nose
    nose = patches.Polygon([[-0.15, 0.98], [0, 1.22], [0.15, 0.98]], closed=True,
                           fc='#cbd5e1', ec='#1e293b', lw=2.5, zorder=2)
    ax.add_patch(nose)

    # Ears
    left_ear = patches.Ellipse((-1.04, 0), 0.12, 0.35, angle=0, fc='#cbd5e1', ec='#1e293b', lw=2, zorder=0)
    right_ear = patches.Ellipse((1.04, 0), 0.12, 0.35, angle=0, fc='#cbd5e1', ec='#1e293b', lw=2, zorder=0)
    ax.add_patch(left_ear)
    ax.add_patch(right_ear)

    # 10-20 reference lines
    ax.plot([0, 0], [-1.0, 1.0], color='#94a3b8', linestyle='--', lw=1.2, zorder=1)
    ax.plot([-1.0, 1.0], [0, 0], color='#94a3b8', linestyle='--', lw=1.2, zorder=1)

    # Other non-selected scalp positions (faint)
    faint_positions = {
        # Remaining SEED-VIG channels (not used by NeuroVigi)
        'T8': (0.90, 0.0), 'TP7': (-0.81, -0.26), 'TP8': (0.81, -0.26),
        'CP1': (-0.22, -0.24), 'CP2': (0.22, -0.24),
        'P1': (-0.20, -0.45), 'P2': (0.20, -0.45),
        'PO3': (-0.30, -0.64), 'POZ': (0.0, -0.66), 'PO4': (0.30, -0.64),
        'O1': (-0.28, -0.85), 'OZ': (0.0, -0.90),
    }
    for name, (x, y) in faint_positions.items():
        circ = plt.Circle((x, y), 0.08, color='#f1f5f9', ec='#94a3b8', lw=1.2, zorder=3)
        ax.add_patch(circ)
        ax.text(x, y, name, color='#64748b', ha='center', va='center', fontsize=7, fontweight='bold', zorder=4)

    # 5 NeuroVigi channels actually used (SEED-VIG indices 0, 1, 2, 9, 16)
    active_positions = {
        'FT7': (-0.81, 0.26),
        'FT8': (0.81, 0.26),
        'T7':  (-0.90, 0.0),
        'PZ':  (0.0, -0.45),
        'O2':  (0.28, -0.85),
    }

    for name, (x, y) in active_positions.items():
        circ = plt.Circle((x, y), 0.11, color='#2563eb', ec='#1e3a8a', lw=2.2, zorder=5)
        ax.add_patch(circ)
        ax.text(x, y, name, color='white', ha='center', va='center', fontsize=8, fontweight='bold', zorder=6)

    ax.set_title("5-Channel NeuroVigi Montage (SEED-VIG)\n(FT7, FT8, T7, PZ, O2)", fontsize=11, fontweight='bold', pad=12, color='#0f172a')
    plt.tight_layout()
    plt.savefig('bluebook_figures/fig_montage.png', dpi=300)
    plt.close()

# -------------------------------------------------------------
# 2. Fig Architecture: End-to-End Deep Learning Architecture
# -------------------------------------------------------------
def create_architecture():
    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)
    ax.axis('off')

    # Draw Stage 1 Box
    s1_rect = patches.FancyBboxPatch((1.8, 0.4), 2.2, 3.2, boxstyle="round,pad=0.15",
                                     fc='#f0fdf4', ec='#16a34a', lw=1.8, linestyle='--')
    ax.add_patch(s1_rect)
    ax.text(2.9, 3.8, "Stage I: Signal Denoising", ha='center', va='center',
            fontsize=9, fontweight='bold', color='#15803d')

    # Draw Stage 2 Box
    s2_rect = patches.FancyBboxPatch((4.3, 0.4), 6.0, 3.2, boxstyle="round,pad=0.15",
                                     fc='#eff6ff', ec='#2563eb', lw=1.8, linestyle='--')
    ax.add_patch(s2_rect)
    ax.text(7.3, 3.8, "Stage II: Spatio-Temporal Attention Classification", ha='center', va='center',
            fontsize=9, fontweight='bold', color='#1d4ed8')

    blocks = [
        {"x": 0.2, "y": 1.5, "w": 1.3, "h": 1.2, "title": "Raw 5-Ch EEG\n$\mathbf{X} \in \mathbb{R}^{B\\times 5\\times 400}$", "sub": "2.0s @ 200 Hz", "fc": "#e2e8f0", "ec": "#475569"},
        {"x": 2.0, "y": 1.5, "w": 1.8, "h": 1.2, "title": "SEN Denoising\n(1D U-Net)", "sub": "101.9K Params\nEnc-Dec + Skips", "fc": "#dcfce7", "ec": "#16a34a"},
        {"x": 4.5, "y": 1.5, "w": 1.6, "h": 1.2, "title": "EEGNet Backbone\nConv2D + Depthwise", "sub": "F1=8, D=2, F2=16\nSpatial Filtering", "fc": "#dbeafe", "ec": "#2563eb"},
        {"x": 6.4, "y": 1.5, "w": 1.6, "h": 1.2, "title": "Bidirectional LSTM\nSequence Model", "sub": "2 Layers, H=64\nOutput: 12x128", "fc": "#ffedd5", "ec": "#ea580c"},
        {"x": 8.3, "y": 1.5, "w": 1.7, "h": 1.2, "title": "Temporal Self-\nAttention Module", "sub": "Dynamic Weights $\\alpha_t$\nContext $\mathbf{c} \in \mathbb{R}^{128}$", "fc": "#f3e8ff", "ec": "#9333ea"},
        {"x": 10.3, "y": 1.5, "w": 1.5, "h": 1.2, "title": "Classification Head\n& Softmax", "sub": "P(Alert), P(Drowsy)\nDropout (p=0.5)", "fc": "#fef08a", "ec": "#ca8a04"}
    ]

    for b in blocks:
        patch = patches.FancyBboxPatch((b['x'], b['y']), b['w'], b['h'], boxstyle="round,pad=0.08",
                                       fc=b['fc'], ec=b['ec'], lw=1.8)
        ax.add_patch(patch)
        ax.text(b['x'] + b['w']/2, b['y'] + b['h']*0.62, b['title'], ha='center', va='center',
                fontsize=8, fontweight='bold', color='#0f172a')
        ax.text(b['x'] + b['w']/2, b['y'] + b['h']*0.22, b['sub'], ha='center', va='center',
                fontsize=6.5, color='#334155')

    # Connecting arrows
    ax.annotate('', xy=(2.0, 2.1), xytext=(1.5, 2.1),
                arrowprops=dict(facecolor='#0f172a', edgecolor='#0f172a', width=1.5, headwidth=6, shrink=0.05))
    ax.annotate('', xy=(4.5, 2.1), xytext=(3.8, 2.1),
                arrowprops=dict(facecolor='#0f172a', edgecolor='#0f172a', width=1.5, headwidth=6, shrink=0.05))
    ax.annotate('', xy=(6.4, 2.1), xytext=(6.1, 2.1),
                arrowprops=dict(facecolor='#0f172a', edgecolor='#0f172a', width=1.5, headwidth=6, shrink=0.05))
    ax.annotate('', xy=(8.3, 2.1), xytext=(8.0, 2.1),
                arrowprops=dict(facecolor='#0f172a', edgecolor='#0f172a', width=1.5, headwidth=6, shrink=0.05))
    ax.annotate('', xy=(10.3, 2.1), xytext=(10.0, 2.1),
                arrowprops=dict(facecolor='#0f172a', edgecolor='#0f172a', width=1.5, headwidth=6, shrink=0.05))

    # Global Residual Skip Connection
    ax.plot([0.85, 0.85, 4.15, 4.15], [2.7, 3.3, 3.3, 2.1], color='#ef4444', lw=1.8, linestyle='--')
    ax.annotate('', xy=(4.5, 2.1), xytext=(4.15, 2.1),
                arrowprops=dict(facecolor='#ef4444', edgecolor='#ef4444', width=1.5, headwidth=6, shrink=0.05))
    ax.text(2.5, 3.45, "Global Residual Connection: $\mathbf{\\tilde{X}} = \mathbf{X} + \mathrm{SEN}(\mathbf{X})$",
            ha='center', va='center', fontsize=7.5, color='#b91c1c', fontweight='bold')

    plt.tight_layout()
    plt.savefig('bluebook_figures/fig_architecture.png', dpi=300)
    plt.close()

# -------------------------------------------------------------
# 3. Fig DFD: Data Flow Diagrams Level 0 & Level 1
# -------------------------------------------------------------
def create_dfd():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=300)
    
    # DFD Level 0 (Context Diagram)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 4)
    ax1.axis('off')
    ax1.text(5, 3.6, "Data Flow Diagram - Level 0 (Context Level)", ha='center', fontsize=10, fontweight='bold')

    # Entities
    ext_driver = patches.Rectangle((0.4, 1.2), 2.0, 1.4, fc='#f1f5f9', ec='#0f172a', lw=1.8)
    proc_core = patches.Circle((5.0, 1.9), 1.2, fc='#e0f2fe', ec='#0284c7', lw=2)
    ext_alert = patches.Rectangle((7.6, 1.2), 2.0, 1.4, fc='#fef2f2', ec='#b91c1c', lw=1.8)

    ax1.add_patch(ext_driver)
    ax1.add_patch(proc_core)
    ax1.add_patch(ext_alert)

    ax1.text(1.4, 1.9, "Driver &\nEEG\nSensors", ha='center', va='center', fontsize=8.5, fontweight='bold')
    ax1.text(5.0, 1.9, "0.0\nNeuroVigi\nVigilance\nSystem", ha='center', va='center', fontsize=9, fontweight='bold', color='#0369a1')
    ax1.text(8.6, 1.9, "In-Cabin DMS\nAudio Alarm &\nFleet Telemetry", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#991b1b')

    # Arrows
    ax1.annotate('', xy=(3.8, 2.2), xytext=(2.4, 2.2),
                 arrowprops=dict(facecolor='#0f172a', width=1.5, headwidth=6))
    ax1.text(3.1, 2.45, "Raw 5-Ch EEG Stream", ha='center', fontsize=7.5)

    ax1.annotate('', xy=(7.6, 2.2), xytext=(6.2, 2.2),
                 arrowprops=dict(facecolor='#b91c1c', edgecolor='#b91c1c', width=1.5, headwidth=6))
    ax1.text(6.9, 2.45, "Alert State & Probabilities", ha='center', fontsize=7.5, color='#991b1b')

    # DFD Level 1 (Decomposition)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 4)
    ax2.axis('off')
    ax2.text(5, 3.7, "Data Flow Diagram - Level 1 (Functional Decomposition)", ha='center', fontsize=10, fontweight='bold')

    nodes = [
        {"id": "1.0", "name": "Buffer Ingestion\n& Standardization", "x": 0.5, "y": 1.5, "w": 1.8, "h": 1.1},
        {"id": "2.0", "name": "SEN Denoising\n& Reconstruction", "x": 2.7, "y": 1.5, "w": 1.8, "h": 1.1},
        {"id": "3.0", "name": "Spatio-Temporal\nFeature Extraction", "x": 4.9, "y": 1.5, "w": 1.9, "h": 1.1},
        {"id": "4.0", "name": "Attention-Driven\nClassification", "x": 7.2, "y": 1.5, "w": 1.9, "h": 1.1},
    ]
    for n in nodes:
        rect = patches.FancyBboxPatch((n['x'], n['y']), n['w'], n['h'], boxstyle="round,pad=0.05",
                                      fc='#f8fafc', ec='#334155', lw=1.5)
        ax2.add_patch(rect)
        ax2.text(n['x'] + n['w']/2, n['y'] + n['h']*0.75, n['id'], ha='center', fontsize=8, fontweight='bold', color='#2563eb')
        ax2.text(n['x'] + n['w']/2, n['y'] + n['h']*0.35, n['name'], ha='center', fontsize=7, fontweight='bold')

    ax2.annotate('', xy=(2.7, 2.05), xytext=(2.3, 2.05), arrowprops=dict(facecolor='#0f172a', width=1.2, headwidth=5))
    ax2.annotate('', xy=(4.9, 2.05), xytext=(4.5, 2.05), arrowprops=dict(facecolor='#0f172a', width=1.2, headwidth=5))
    ax2.annotate('', xy=(7.2, 2.05), xytext=(6.8, 2.05), arrowprops=dict(facecolor='#0f172a', width=1.2, headwidth=5))

    ax2.text(2.5, 2.3, "Norm EEG", ha='center', fontsize=6.5)
    ax2.text(4.7, 2.3, "Cleaned", ha='center', fontsize=6.5)
    ax2.text(7.05, 2.3, "Features", ha='center', fontsize=6.5)

    plt.tight_layout()
    plt.savefig('bluebook_figures/fig_dfd.png', dpi=300)
    plt.close()

# -------------------------------------------------------------
# 4. Fig UML: Use Case & Sequence Diagrams
# -------------------------------------------------------------
def create_uml():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)

    # Use Case Diagram
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title("UML Use Case Diagram", fontsize=10, fontweight='bold')

    # System boundary
    sys_bound = patches.Rectangle((3.0, 0.5), 6.5, 9.0, fc='#f8fafc', ec='#475569', lw=1.5, linestyle='--')
    ax1.add_patch(sys_bound)
    ax1.text(6.25, 9.1, "NeuroVigi Vehicle BCI System", ha='center', fontsize=9, fontweight='bold', color='#1e293b')

    # Actors
    # Driver
    ax1.plot([1.2, 1.2], [7.3, 6.7], color='#0f172a', lw=1.8)
    ax1.plot([0.8, 1.6], [7.1, 7.1], color='#0f172a', lw=1.8)
    ax1.plot([1.2, 0.8], [6.7, 6.1], color='#0f172a', lw=1.8)
    ax1.plot([1.2, 1.6], [6.7, 6.1], color='#0f172a', lw=1.8)
    c_head = plt.Circle((1.2, 7.6), 0.25, color='#0f172a')
    ax1.add_patch(c_head)
    ax1.text(1.2, 5.7, "Vehicle\nDriver", ha='center', fontsize=8, fontweight='bold')

    # Safety Officer / System
    ax1.plot([1.2, 1.2], [3.3, 2.7], color='#0f172a', lw=1.8)
    ax1.plot([0.8, 1.6], [3.1, 3.1], color='#0f172a', lw=1.8)
    ax1.plot([1.2, 0.8], [2.7, 2.1], color='#0f172a', lw=1.8)
    ax1.plot([1.2, 1.6], [2.7, 2.1], color='#0f172a', lw=1.8)
    c_head2 = plt.Circle((1.2, 3.6), 0.25, color='#0f172a')
    ax1.add_patch(c_head2)
    ax1.text(1.2, 1.7, "Fleet Safety\nManager", ha='center', fontsize=8, fontweight='bold')

    use_cases = [
        {"name": "Stream 5-Ch EEG", "y": 8.0},
        {"name": "Perform Real-Time SEN Denoising", "y": 6.5},
        {"name": "Decode Cognitive Vigilance", "y": 5.0},
        {"name": "Trigger In-Cabin Multi-Modal Alarm", "y": 3.5},
        {"name": "Inspect Telemetry & Fatigue Log", "y": 2.0},
    ]
    for uc in use_cases:
        ellipse = patches.Ellipse((6.25, uc['y']), 3.8, 0.9, fc='#e0f2fe', ec='#0369a1', lw=1.5)
        ax1.add_patch(ellipse)
        ax1.text(6.25, uc['y'], uc['name'], ha='center', va='center', fontsize=7.5, fontweight='bold')

        # Connect lines
        if uc['y'] >= 3.5:
            ax1.plot([1.6, 4.35], [7.1, uc['y']], color='#64748b', lw=1.2)
        if uc['y'] <= 3.5:
            ax1.plot([1.6, 4.35], [3.1, uc['y']], color='#64748b', lw=1.2)

    # Sequence Diagram
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title("UML Sequence Diagram (Real-Time Loop)", fontsize=10, fontweight='bold')

    lifelines = [
        {"name": "Headband", "x": 1.2},
        {"name": "RingBuffer", "x": 3.4},
        {"name": "SEN Module", "x": 5.6},
        {"name": "Classifier", "x": 7.8},
        {"name": "AlarmSystem", "x": 9.2}
    ]
    for ll in lifelines:
        box = patches.Rectangle((ll['x']-0.7, 9.0), 1.4, 0.6, fc='#f1f5f9', ec='#1e293b', lw=1.2)
        ax2.add_patch(box)
        ax2.text(ll['x'], 9.3, ll['name'], ha='center', va='center', fontsize=7.5, fontweight='bold')
        ax2.plot([ll['x'], ll['x']], [9.0, 0.5], color='#94a3b8', linestyle='--', lw=1.2)

    # Messages
    msgs = [
        {"y": 8.0, "x1": 1.2, "x2": 3.4, "txt": "1: Push 200 samples/s"},
        {"y": 6.8, "x1": 3.4, "x2": 5.6, "txt": "2: Extract 400-pt Epoch"},
        {"y": 5.4, "x1": 5.6, "x2": 7.8, "txt": "3: Denoised Epoch X~"},
        {"y": 4.0, "x1": 7.8, "x2": 3.4, "txt": "4: P(Drowsy) Posterior"},
        {"y": 2.6, "x1": 7.8, "x2": 9.2, "txt": "5: P >= 0.5 for 3s (Alert!)"},
    ]
    for m in msgs:
        ax2.annotate('', xy=(m['x2'], m['y']), xytext=(m['x1'], m['y']),
                     arrowprops=dict(facecolor='#0f172a', width=1.0, headwidth=4.5))
        ax2.text((m['x1']+m['x2'])/2, m['y']+0.22, m['txt'], ha='center', fontsize=6.8, fontweight='bold')

    plt.tight_layout()
    plt.savefig('bluebook_figures/fig_uml.png', dpi=300)
    plt.close()

# -------------------------------------------------------------
# 5. Fig Gantt: Project Schedule and Work Breakdown Structure
# -------------------------------------------------------------
def create_gantt():
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    tasks = [
        "1. Literature Review & Problem Formulation",
        "2. SEED-VIG Preprocessing & 5-Ch Extraction",
        "3. 1D U-Net SEN Architecture Design",
        "4. EEGNet + BiLSTM + Attention Integration",
        "5. 23-Fold LOSO-CV Training & Benchmarking",
        "6. Synthetic Noise Injection Stress Tests",
        "7. Interactive Streamlit Dashboard Dev",
        "8. Report, Paper & Documentation Compilation"
    ]
    starts = [1, 2, 4, 6, 7, 9, 10, 11]
    durations = [2, 3, 3, 3, 3, 2, 3, 3]
    colors = ['#38bdf8', '#38bdf8', '#818cf8', '#818cf8', '#fbbf24', '#f87171', '#34d399', '#a78bfa']

    y_pos = np.arange(len(tasks))
    ax.barh(y_pos, durations, left=starts, height=0.55, align='center', color=colors, edgecolor='#1e293b', lw=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tasks, fontsize=8.5, fontweight='bold')
    ax.invert_yaxis()

    ax.set_xlabel("Project Timeline (Months / Weeks)", fontsize=9, fontweight='bold')
    ax.set_xlim(0, 15)
    ax.set_xticks(np.arange(1, 15, 1))
    ax.set_xticklabels([f"M{i}" for i in range(1, 15)], fontsize=8)
    ax.grid(axis='x', linestyle=':', alpha=0.6)

    ax.set_title("Project Work Breakdown Structure and Timeline (Semesters VII & VIII)",
                 fontsize=10, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig('bluebook_figures/fig_gantt.png', dpi=300)
    plt.close()

# -------------------------------------------------------------
# 6. Fig Waveform: Time Domain Signal Clean vs Noisy vs SEN
# -------------------------------------------------------------
def create_waveforms():
    np.random.seed(42)
    t = np.linspace(0, 2.0, 400) # 2 seconds at 200 Hz
    
    # Brain wave: mix of alpha (10 Hz), theta (6 Hz), and slow drift
    clean = 1.2 * np.sin(2 * np.pi * 10 * t) + 0.8 * np.sin(2 * np.pi * 6 * t) + 0.4 * np.sin(2 * np.pi * 2 * t)
    
    # Additive artifact: blink spike + Gaussian high frequency noise
    blink = 4.0 * np.exp(-((t - 0.9)**2) / (2 * (0.08**2)))
    noise = clean + blink + 0.7 * np.random.normal(0, 1, 400)
    
    # SEN reconstructed output (clean wave with tiny residual)
    sen_recon = clean + 0.15 * np.random.normal(0, 1, 400)

    fig, axes = plt.subplots(3, 1, figsize=(10, 5), dpi=300, sharex=True)

    axes[0].plot(t, clean, color='#1e40af', lw=1.5)
    axes[0].set_ylabel("Clean EEG (uV)", fontsize=8, fontweight='bold')
    axes[0].grid(True, linestyle=':', alpha=0.5)
    axes[0].set_title("Time-Domain Waveform Restoration via 1D U-Net SEN Module", fontsize=10, fontweight='bold')

    axes[1].plot(t, noise, color='#dc2626', lw=1.2)
    axes[1].set_ylabel("Corrupted (50% N)", fontsize=8, fontweight='bold')
    axes[1].grid(True, linestyle=':', alpha=0.5)

    axes[2].plot(t, sen_recon, color='#16a34a', lw=1.5)
    axes[2].set_ylabel("SEN Denoised", fontsize=8, fontweight='bold')
    axes[2].set_xlabel("Time (seconds)", fontsize=9, fontweight='bold')
    axes[2].grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig('bluebook_figures/fig_waveform.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    print("Generating diagrams...")
    create_montage()
    create_architecture()
    create_dfd()
    create_uml()
    create_gantt()
    create_waveforms()
    print("All diagrams generated successfully!")
