import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import shutil

# Configure Matplotlib fonts
plt.rcParams['font.family'] = 'DejaVu Sans'

# -------------------------------------------------------------------------
# 1. DFD Level 0 (Context Level Diagram) - Standalone with Large, Clear Text
# -------------------------------------------------------------------------
def render_dfd_level_0():
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.8)
    ax.axis('off')

    # Main Diagram Title
    ax.text(5.5, 4.4, "Data Flow Diagram – Level 0 (Context Level)",
            ha='center', va='center', fontsize=16, fontweight='bold', color='#0f172a')

    # External Entity: Driver & EEG Sensors
    entity_driver = patches.FancyBboxPatch((0.4, 1.2), 2.5, 2.3, boxstyle="round,pad=0.1",
                                          fc='#f8fafc', ec='#1e293b', lw=2.2)
    ax.add_patch(entity_driver)
    ax.text(1.65, 2.55, "Driver &\nEEG\nSensors",
            ha='center', va='center', fontsize=13, fontweight='bold', color='#0f172a', linespacing=1.2)
    ax.text(1.65, 1.6, "[FT7, FT8, T7, PZ, O2]\n(200 Hz Stream)",
            ha='center', va='center', fontsize=10, color='#475569', linespacing=1.1)

    # Core Process 0.0: NeuroVigi System
    proc_core = plt.Circle((5.5, 2.35), 1.45, fc='#e0f2fe', ec='#0284c7', lw=2.8)
    ax.add_patch(proc_core)
    ax.text(5.5, 2.9, "0.0", ha='center', va='center', fontsize=15, fontweight='bold', color='#0369a1')
    ax.text(5.5, 2.35, "NeuroVigi Core\nBCI System", ha='center', va='center',
            fontsize=14, fontweight='bold', color='#0f172a', linespacing=1.15)
    ax.text(5.5, 1.65, "(SEN Denoising &\nVigilance Inference)", ha='center', va='center',
            fontsize=10.5, fontstyle='italic', color='#0369a1', linespacing=1.1)

    # External Entity: Vehicle In-Cabin DMS & Fleet
    entity_alert = patches.FancyBboxPatch((8.1, 1.2), 2.5, 2.3, boxstyle="round,pad=0.1",
                                         fc='#fef2f2', ec='#b91c1c', lw=2.2)
    ax.add_patch(entity_alert)
    ax.text(9.35, 2.55, "In-Cabin DMS,\nAudio Alarm &\nFleet Telemetry",
            ha='center', va='center', fontsize=13, fontweight='bold', color='#991b1b', linespacing=1.2)
    ax.text(9.35, 1.6, "[Emergency Alerts &\nFatigue Trajectory]",
            ha='center', va='center', fontsize=10, color='#7f1d1d', linespacing=1.1)

    # Data Flow 1: Raw Stream
    ax.annotate('', xy=(4.05, 2.35), xytext=(2.9, 2.35),
                arrowprops=dict(facecolor='#0f172a', edgecolor='#0f172a', width=2.5, headwidth=8.5, shrink=0.04))
    ax.text(3.475, 2.65, "Raw 5-Ch EEG Stream", ha='center', va='bottom',
            fontsize=11.5, fontweight='bold', color='#0f172a')
    ax.text(3.475, 2.05, "400-pt Epochs (2.0s)", ha='center', va='top',
            fontsize=9.5, color='#475569')

    # Data Flow 2: Alert State
    ax.annotate('', xy=(8.1, 2.35), xytext=(6.95, 2.35),
                arrowprops=dict(facecolor='#dc2626', edgecolor='#dc2626', width=2.5, headwidth=8.5, shrink=0.04))
    ax.text(7.525, 2.65, "Alert State (Alarm/Normal)", ha='center', va='bottom',
            fontsize=11.5, fontweight='bold', color='#dc2626')
    ax.text(7.525, 2.05, "P(Drowsy) Posteriors", ha='center', va='top',
            fontsize=9.5, color='#991b1b')

    plt.tight_layout()
    os.makedirs('bluebook_figures', exist_ok=True)
    out_path = 'bluebook_figures/dfd_level_0.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

# -------------------------------------------------------------------------
# 2. DFD Level 1 (Functional Decomposition) - Standalone with Large, Clear Text
# -------------------------------------------------------------------------
def render_dfd_level_1():
    fig, ax = plt.subplots(figsize=(13.5, 6.2), dpi=300)
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 6.2)
    ax.axis('off')

    # Main Diagram Title
    ax.text(6.75, 5.8, "Data Flow Diagram – Level 1 (Functional Decomposition)",
            ha='center', va='center', fontsize=16, fontweight='bold', color='#0f172a')

    # 4 Main Functional Processes
    processes = [
        {
            "id": "1.0",
            "title": "Buffer Ingestion &\nStandardization",
            "desc": "Circular Ring Buffer\nChannel Z-Score Norm",
            "x": 0.4, "y": 2.2, "w": 2.6, "h": 2.3,
            "fc": "#f0fdf4", "ec": "#16a34a"
        },
        {
            "id": "2.0",
            "title": "SEN Denoising &\nReconstruction",
            "desc": "1D U-Net Enc-Dec\nGlobal Residual Skip",
            "x": 3.6, "y": 2.2, "w": 2.6, "h": 2.3,
            "fc": "#eff6ff", "ec": "#2563eb"
        },
        {
            "id": "3.0",
            "title": "Spatio-Temporal\nFeature Extraction",
            "desc": "EEGNet Depthwise Conv\n2-Layer BiLSTM Recurrence",
            "x": 6.8, "y": 2.2, "w": 2.8, "h": 2.3,
            "fc": "#fff7ed", "ec": "#ea580c"
        },
        {
            "id": "4.0",
            "title": "Attention Scoring &\nAlert Decision",
            "desc": "Temporal Self-Attention\nEMA Hysteresis & Alarm",
            "x": 10.2, "y": 2.2, "w": 2.9, "h": 2.3,
            "fc": "#faf5ff", "ec": "#9333ea"
        }
    ]

    for p in processes:
        box = patches.FancyBboxPatch((p['x'], p['y']), p['w'], p['h'], boxstyle="round,pad=0.1",
                                     fc=p['fc'], ec=p['ec'], lw=2.2)
        ax.add_patch(box)
        # ID badge
        badge = patches.FancyBboxPatch((p['x'] + 0.15, p['y'] + p['h'] - 0.55), 0.7, 0.4,
                                       boxstyle="round,pad=0.05", fc=p['ec'], ec=p['ec'])
        ax.add_patch(badge)
        ax.text(p['x'] + 0.5, p['y'] + p['h'] - 0.35, p['id'], ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')

        # Title
        ax.text(p['x'] + p['w']/2, p['y'] + 1.25, p['title'], ha='center', va='center',
                fontsize=12, fontweight='bold', color='#0f172a', linespacing=1.15)
        # Description
        ax.text(p['x'] + p['w']/2, p['y'] + 0.45, p['desc'], ha='center', va='center',
                fontsize=9.5, color='#475569', linespacing=1.1)

    # Arrows between processes
    # Process 1 -> Process 2
    ax.annotate('', xy=(3.6, 3.35), xytext=(3.0, 3.35),
                arrowprops=dict(facecolor='#0f172a', edgecolor='#0f172a', width=2.2, headwidth=7.5, shrink=0.04))
    ax.text(3.3, 3.75, "Standardized\nEpoch X", ha='center', va='center',
            fontsize=10, fontweight='bold', color='#0f172a', linespacing=1.05)

    # Process 2 -> Process 3
    ax.annotate('', xy=(6.8, 3.35), xytext=(6.2, 3.35),
                arrowprops=dict(facecolor='#0f172a', edgecolor='#0f172a', width=2.2, headwidth=7.5, shrink=0.04))
    ax.text(6.5, 3.75, "Denoised\nEpoch X~", ha='center', va='center',
            fontsize=10, fontweight='bold', color='#0f172a', linespacing=1.05)

    # Process 3 -> Process 4
    ax.annotate('', xy=(10.2, 3.35), xytext=(9.6, 3.35),
                arrowprops=dict(facecolor='#0f172a', edgecolor='#0f172a', width=2.2, headwidth=7.5, shrink=0.04))
    ax.text(9.9, 3.75, "Context\nMatrix H", ha='center', va='center',
            fontsize=10, fontweight='bold', color='#0f172a', linespacing=1.05)

    # Data Store D1: Model Parameters
    ds_box = patches.Rectangle((3.8, 0.4), 5.6, 1.0, fc='#f1f5f9', ec='#334155', lw=1.8)
    ax.add_patch(ds_box)
    # Open sides to represent standard DFD Data Store
    ax.plot([3.8, 3.8], [0.4, 1.4], color='#334155', lw=2.2)
    ax.plot([9.4, 9.4], [0.4, 1.4], color='#334155', lw=2.2)
    ax.text(6.6, 1.05, "D1: Trained NeuroVigi Weights (258,707 Parameters)",
            ha='center', va='center', fontsize=11, fontweight='bold', color='#1e293b')
    ax.text(6.6, 0.65, "[SEN: 101,968 params | Classifier: 156,739 params]",
            ha='center', va='center', fontsize=9.5, color='#475569')

    # Arrows from Data Store to Processes 2 & 3
    ax.annotate('', xy=(4.9, 2.2), xytext=(4.9, 1.4),
                arrowprops=dict(facecolor='#64748b', edgecolor='#64748b', width=1.5, headwidth=6, shrink=0.05))
    ax.annotate('', xy=(8.2, 2.2), xytext=(8.2, 1.4),
                arrowprops=dict(facecolor='#64748b', edgecolor='#64748b', width=1.5, headwidth=6, shrink=0.05))

    # Output Arrow from Process 4 to Outside (DMS)
    ax.annotate('', xy=(13.3, 3.35), xytext=(13.1, 3.35),
                arrowprops=dict(facecolor='#dc2626', edgecolor='#dc2626', width=2.5, headwidth=8.5))
    # Target Box
    alert_box = patches.FancyBboxPatch((11.0, 0.4), 2.1, 1.1, boxstyle="round,pad=0.08",
                                       fc='#fef2f2', ec='#b91c1c', lw=1.8)
    ax.add_patch(alert_box)
    ax.text(12.05, 1.08, "In-Cabin DMS", ha='center', va='center',
            fontsize=10.5, fontweight='bold', color='#991b1b')
    ax.text(12.05, 0.68, "Audio Alarm / HUD", ha='center', va='center',
            fontsize=9, color='#7f1d1d')

    # Arrow from P4 to Alert Box
    ax.plot([11.65, 11.65, 12.05, 12.05], [2.2, 1.8, 1.8, 1.5], color='#dc2626', lw=2)
    ax.annotate('', xy=(12.05, 1.5), xytext=(12.05, 1.6),
                arrowprops=dict(facecolor='#dc2626', edgecolor='#dc2626', width=2, headwidth=6))
    ax.text(12.45, 1.85, "Alert State", ha='left', va='center',
            fontsize=9.5, fontweight='bold', color='#dc2626')

    plt.tight_layout()
    out_path = 'bluebook_figures/dfd_level_1.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

# -------------------------------------------------------------------------
# 3. Combined DFD Figure (Updated with tight margins and larger text)
# -------------------------------------------------------------------------
def render_dfd_combined():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9.5), dpi=300,
                                   gridspec_kw={'height_ratios': [1, 1.25]})

    # --- TOP: Level 0 ---
    ax1.set_xlim(0, 11)
    ax1.set_ylim(0, 4.4)
    ax1.axis('off')
    ax1.text(5.5, 4.0, "Data Flow Diagram – Level 0 (Context Level)",
            ha='center', va='center', fontsize=15, fontweight='bold', color='#0f172a')

    # Driver Entity
    ed = patches.FancyBboxPatch((0.5, 0.9), 2.4, 2.3, boxstyle="round,pad=0.08",
                                fc='#f8fafc', ec='#1e293b', lw=2.2)
    ax1.add_patch(ed)
    ax1.text(1.7, 2.25, "Driver &\nEEG\nSensors",
            ha='center', va='center', fontsize=12.5, fontweight='bold', color='#0f172a', linespacing=1.2)
    ax1.text(1.7, 1.35, "5 Channels @ 200 Hz\n(FT7, FT8, T7, PZ, O2)",
            ha='center', va='center', fontsize=9.5, color='#475569', linespacing=1.1)

    # Core Process
    pc = plt.Circle((5.5, 2.05), 1.4, fc='#e0f2fe', ec='#0284c7', lw=2.5)
    ax1.add_patch(pc)
    ax1.text(5.5, 2.65, "0.0", ha='center', va='center', fontsize=15, fontweight='bold', color='#0369a1')
    ax1.text(5.5, 2.05, "NeuroVigi Core\nBCI System", ha='center', va='center',
            fontsize=13.5, fontweight='bold', color='#0f172a', linespacing=1.15)
    ax1.text(5.5, 1.45, "(Denoising & Vigilance)", ha='center', va='center',
            fontsize=10, fontstyle='italic', color='#0369a1')

    # DMS Entity
    ea = patches.FancyBboxPatch((8.1, 0.9), 2.4, 2.3, boxstyle="round,pad=0.08",
                                fc='#fef2f2', ec='#b91c1c', lw=2.2)
    ax1.add_patch(ea)
    ax1.text(9.3, 2.25, "In-Cabin DMS,\nAudio Alarm &\nFleet Telemetry",
            ha='center', va='center', fontsize=12.5, fontweight='bold', color='#991b1b', linespacing=1.2)
    ax1.text(9.3, 1.35, "Emergency Alerts &\nFatigue Trajectory",
            ha='center', va='center', fontsize=9.5, color='#7f1d1d', linespacing=1.1)

    # Arrows
    ax1.annotate('', xy=(4.1, 2.05), xytext=(2.9, 2.05),
                arrowprops=dict(facecolor='#0f172a', width=2.2, headwidth=8, shrink=0.04))
    ax1.text(3.5, 2.35, "Raw 5-Ch EEG", ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.annotate('', xy=(8.1, 2.05), xytext=(6.9, 2.05),
                arrowprops=dict(facecolor='#dc2626', edgecolor='#dc2626', width=2.2, headwidth=8, shrink=0.04))
    ax1.text(7.5, 2.35, "Alert State & Probabilities", ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='#dc2626')

    # --- BOTTOM: Level 1 ---
    ax2.set_xlim(0, 11)
    ax2.set_ylim(0, 5.0)
    ax2.axis('off')
    ax2.text(5.5, 4.6, "Data Flow Diagram – Level 1 (Functional Decomposition)",
            ha='center', va='center', fontsize=15, fontweight='bold', color='#0f172a')

    p1_nodes = [
        {"id": "1.0", "title": "Buffer Ingestion &\nStandardization", "x": 0.3, "y": 1.7, "w": 2.3, "h": 2.2, "fc": "#f0fdf4", "ec": "#16a34a"},
        {"id": "2.0", "title": "SEN Denoising &\nReconstruction", "x": 3.0, "y": 1.7, "w": 2.3, "h": 2.2, "fc": "#eff6ff", "ec": "#2563eb"},
        {"id": "3.0", "title": "Spatio-Temporal\nFeature Extraction", "x": 5.7, "y": 1.7, "w": 2.4, "h": 2.2, "fc": "#fff7ed", "ec": "#ea580c"},
        {"id": "4.0", "title": "Attention Scoring &\nAlert Decision", "x": 8.4, "y": 1.7, "w": 2.3, "h": 2.2, "fc": "#faf5ff", "ec": "#9333ea"}
    ]

    for p in p1_nodes:
        box = patches.FancyBboxPatch((p['x'], p['y']), p['w'], p['h'], boxstyle="round,pad=0.08",
                                     fc=p['fc'], ec=p['ec'], lw=2.0)
        ax2.add_patch(box)
        badge = patches.FancyBboxPatch((p['x'] + 0.1, p['y'] + p['h'] - 0.5), 0.6, 0.38,
                                       boxstyle="round,pad=0.04", fc=p['ec'], ec=p['ec'])
        ax2.add_patch(badge)
        ax2.text(p['x'] + 0.4, p['y'] + p['h'] - 0.31, p['id'], ha='center', va='center',
                fontsize=10.5, fontweight='bold', color='white')
        ax2.text(p['x'] + p['w']/2, p['y'] + 1.05, p['title'], ha='center', va='center',
                fontsize=11, fontweight='bold', color='#0f172a', linespacing=1.15)

    # Connections
    ax2.annotate('', xy=(3.0, 2.8), xytext=(2.6, 2.8),
                arrowprops=dict(facecolor='#0f172a', width=1.8, headwidth=6.5, shrink=0.05))
    ax2.text(2.8, 3.2, "Norm X", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    ax2.annotate('', xy=(5.7, 2.8), xytext=(5.3, 2.8),
                arrowprops=dict(facecolor='#0f172a', width=1.8, headwidth=6.5, shrink=0.05))
    ax2.text(5.5, 3.2, "Clean X~", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    ax2.annotate('', xy=(8.4, 2.8), xytext=(8.1, 2.8),
                arrowprops=dict(facecolor='#0f172a', width=1.8, headwidth=6.5, shrink=0.05))
    ax2.text(8.25, 3.2, "Context H", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    # Data Store D1
    ds = patches.Rectangle((3.2, 0.3), 4.6, 0.85, fc='#f1f5f9', ec='#334155', lw=1.6)
    ax2.add_patch(ds)
    ax2.plot([3.2, 3.2], [0.3, 1.15], color='#334155', lw=2.2)
    ax2.plot([7.8, 7.8], [0.3, 1.15], color='#334155', lw=2.2)
    ax2.text(5.5, 0.72, "D1: Trained NeuroVigi Weights (258,707 Parameters)",
            ha='center', va='center', fontsize=10, fontweight='bold', color='#1e293b')

    ax2.annotate('', xy=(4.15, 1.7), xytext=(4.15, 1.15),
                arrowprops=dict(facecolor='#64748b', width=1.2, headwidth=5, shrink=0.05))
    ax2.annotate('', xy=(6.9, 1.7), xytext=(6.9, 1.15),
                arrowprops=dict(facecolor='#64748b', width=1.2, headwidth=5, shrink=0.05))

    plt.tight_layout()
    out_path = 'bluebook_figures/fig_dfd.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    render_dfd_level_0()
    render_dfd_level_1()
    render_dfd_combined()
    
    # Copy to project root and artifact directory so user can directly view and download
    art_dir = r'C:\Users\anisk\.gemini\antigravity\brain\82d7710b-daf7-42dd-8257-b7815310846f'
    for f in ['dfd_level_0.png', 'dfd_level_1.png', 'fig_dfd.png']:
        src = os.path.join('bluebook_figures', f)
        dst_root = os.path.join(os.getcwd(), f)
        dst_art = os.path.join(art_dir, f)
        shutil.copyfile(src, dst_root)
        shutil.copyfile(src, dst_art)
        print(f"Copied {f} to root and artifacts.")
