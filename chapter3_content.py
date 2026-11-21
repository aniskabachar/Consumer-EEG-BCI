# -*- coding: utf-8 -*-
"""
chapter3_content.py
Requirement Gathering, Analysis, Planning, Feasibility, Tech Stack, and Gantt Chart.
"""

import os
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from bluebook_helpers import (
    set_cell_margins, set_cell_shading, set_table_borders,
    add_styled_para, add_heading_1, add_heading_2, add_heading_3, add_caption,
    format_paragraph
)

def add_chapter_3(doc):
    add_heading_1(doc, "Chapter 3: Requirement Gathering, Analysis and Planning")

    add_heading_2(doc, "3.1 Requirement Specification")
    p1 = (
        "Engineering a life-critical, in-cabin cognitive state monitoring system requires establishing exhaustive, verifiable "
        "engineering specifications. These requirements are divided into Functional Requirements (FR)—specifying what the software "
        "and algorithmic modules must execute—and Non-Functional Requirements (NFR)—specifying performance thresholds, reliability "
        "metrics, and hardware execution constraints."
    )
    add_styled_para(doc, p1)

    add_heading_3(doc, "3.1.1 Functional Requirements")
    fr_list = [
        "FR-1 (Multi-Channel Ingestion): The system shall continuously ingest continuous raw EEG time-series from exactly five channels (FT7, FT8, T7, PZ, O2) sampled at 200 Hz.",
        "FR-2 (Sliding Window Partitioning): The data ingestion engine shall maintain a circular ring buffer capable of extracting sliding epochs of duration Tw = 2.0 seconds (400 discrete samples per channel) with an update stride of 1.0 second.",
        "FR-3 (Z-Score Normalization): The preprocessing module shall perform online channel-wise standardization to transform microvolt-level potential fluctuations to zero-mean unit-variance representations.",
        "FR-4 (End-to-End Neural Denoising): The Signal Enhancement Network (SEN) shall process normalized 5-channel epochs to reconstruct clean temporal waveforms, attenuating high-frequency noise and baseline ocular drift without altering underlying phase characteristics.",
        "FR-5 (Spatio-Temporal Feature Extraction): The deep learning classifier shall apply depthwise separable convolutions across all five electrode locations to learn invariant spatial filter-bank projections.",
        "FR-6 (Temporal Micro-State Sequence Modeling): The Bidirectional LSTM module shall process temporal feature representations across both forward and reverse directions, capturing progressive fatigue transition kinetics.",
        "FR-7 (Dynamic Attention Weighting): The temporal self-attention mechanism shall compute dynamic relevance weights across sub-epoch intervals, focusing downstream decision logic on transient alpha/theta bursts.",
        "FR-8 (Hysteresis-Based Alarm Triggering): The post-processing engine shall apply an exponential moving average (EMA, beta=0.70) to posterior vigilance probabilities, asserting a high-priority audible/visual alert whenever smoothed drowsiness exceeds 0.50 for three consecutive update windows (3.0 seconds)."
    ]
    for fr in fr_list:
        add_styled_para(doc, fr, size_pt=11, space_after_pt=4)

    add_heading_3(doc, "3.1.2 Non-Functional Requirements")
    nfr_list = [
        "NFR-1 (Real-Time Latency Deadline): End-to-end inference latency—including buffer ingestion, standardization, SEN denoising, and classification—must execute in strictly less than 20 milliseconds on edge CPU and less than 5 milliseconds on edge GPU.",
        "NFR-2 (Lightweight Parameter Footprint): The total trainable parameter count of the combined neural pipeline must not exceed 300,000 parameters (< 2.0 MB model weight file size) to ensure compatibility with memory-constrained edge hardware.",
        "NFR-3 (Noise Resilience Threshold): The system must maintain at least 80% binary vigilance classification accuracy under severe additive Gaussian noise contamination up to 50% (SNR = 6 dB).",
        "NFR-4 (Zero-Calibration Cross-Subject Generalization): The model must achieve a cross-subject classification accuracy exceeding 75% on completely unseen drivers under Leave-One-Subject-Out Cross-Validation without any per-user recalibration.",
        "NFR-5 (High Alert Specificity): To prevent driver annoyance and alert fatigue, the false positive alarm rate (alert false alarms) must not exceed 15% (Specificity >= 85%).",
        "NFR-6 (Robustness & Fault Tolerance): The software pipeline must handle missing packets or corrupted sensor channels gracefully without causing memory leaks or daemon crashes.",
        "NFR-7 (Data Security & Privacy): All raw neural recordings and inferred drowsiness logs must remain strictly local to the vehicular edge hardware, safeguarding driver biometric privacy.",
        "NFR-8 (Operator Usability): The interactive dashboard interface must display real-time 5-channel oscilloscopes, responsive probability meters, and historical trend timelines with an interface refresh latency under 50 milliseconds."
    ]
    for nfr in nfr_list:
        add_styled_para(doc, nfr, size_pt=11, space_after_pt=4)

    # Table 3.1 Requirements Matrix
    add_caption(doc, "Table 3.1: Comprehensive Functional and Non-Functional Requirements Matrix", is_table=True)
    tbl_req = doc.add_table(rows=1, cols=4)
    tbl_req.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_req, color="94A3B8")

    req_headers = ["Req ID", "Requirement Description", "Category", "Target Specification / Benchmark"]
    for i, h in enumerate(req_headers):
        c = tbl_req.rows[0].cells[i]
        c.paragraphs[0].text = h
        set_cell_shading(c, "F1F5F9")
        set_cell_margins(c, top=80, bottom=80, left=80, right=80)
        format_paragraph(c.paragraphs[0], size_pt=9.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)

    req_rows = [
        ("FR-1", "5-Channel Ingestion", "Functional", "Channels: FT7, FT8, T7, PZ, O2 @ 200 Hz"),
        ("FR-2", "Sliding Ring Buffer", "Functional", "Window: 2.0 s (400 samples), Stride: 1.0 s"),
        ("FR-4", "1D U-Net SEN Denoising", "Functional", "Reconstruct waveform in raw voltage domain"),
        ("FR-8", "EMA Alert Post-Processing", "Functional", "Beta = 0.70, Alarm if P(Drowsy) >= 0.50 for 3 s"),
        ("NFR-1", "Inference Latency", "Non-Functional", "< 5 ms on Edge GPU, < 15 ms on Edge CPU"),
        ("NFR-2", "Parameter Economy", "Non-Functional", "< 300,000 parameters (< 2 MB memory footprint)"),
        ("NFR-3", "Noise Robustness", "Non-Functional", "Acc >= 80% under 50% additive noise (6 dB SNR)"),
        ("NFR-4", "Cross-Subject Generalization", "Non-Functional", "LOSO-CV Accuracy >= 78% (Zero Calibration)")
    ]
    for r_data in req_rows:
        row = tbl_req.add_row()
        for i, val in enumerate(r_data):
            c = row.cells[i]
            c.paragraphs[0].text = val
            set_cell_margins(c, top=60, bottom=60, left=60, right=60)
            format_paragraph(c.paragraphs[0], size_pt=9, bold=(i==0),
                             align=WD_ALIGN_PARAGRAPH.CENTER if i in [0, 2] else WD_ALIGN_PARAGRAPH.LEFT)

    add_heading_2(doc, "3.2 Feasibility Study")
    p_feas = (
        "A multi-dimensional feasibility study was conducted to evaluate the technical, economic, operational, and legal viability "
        "of deploying NeuroVigi within commercial vehicles:\n\n"
        "1. Technical Feasibility: The primary technical question was whether five electrodes provide sufficient neurodynamic information "
        "to distinguish vigilance states compared to 32 or 64 channels. Cognitive neuroscience literature confirms that frontal midline theta "
        "(Fm-theta, 4-8 Hz) and frontal alpha rhythms (8-13 Hz) are prime physiological indicators of mental fatigue. Modern deep learning architectures "
        "like EEGNet and 1D U-Net have demonstrated exceptional capability in extracting spatial filter weights and filtering non-linear artifacts. "
        "Given the availability of high-performance GPU tensor engines and PyTorch mixed-precision acceleration, edge execution is fully feasible.\n\n"
        "2. Economic Feasibility: High-density clinical EEG machines cost upwards of $20,000 to $50,000, rendering them economically unfeasible "
        "for automotive integration. Conversely, consumer-grade dry-electrode frontal headbands (utilizing conductive polymer or dry silver-chloride "
        "electrodes) cost between $150 and $400 at scale. When contrasted against the immense economic costs of a single commercial truck crash "
        "(averaging over $150,000 in property damage and liability), the economic justification for NeuroVigi is compelling.\n\n"
        "3. Operational Feasibility: Vehicle drivers cannot be expected to wear uncomfortable caps with conductive gels or undergo cumbersome "
        "daily calibration routines. NeuroVigi's few-channel montage (FT7, FT8, T7, PZ, O2) targets minimal skin preparation and can be integrated into standard baseball caps, "
        "hard-hats, or sun-visors. Its zero-calibration LOSO-CV architecture enables immediate plug-and-play onboarding.\n\n"
        "4. Schedule & Legal Feasibility: The development schedule was planned across two academic semesters (Semesters VII & VIII), incorporating "
        "preliminary dataset extraction, neural architecture implementation, 23-fold LOSO validation, and dashboard deployment. From a legal standpoint, "
        "the system operates non-invasively and adheres to biometric data privacy principles by conducting all inference strictly locally on edge hardware."
    )
    for seg in p_feas.split('\n\n'):
        add_styled_para(doc, seg)

    add_heading_2(doc, "3.3 Methodology")
    p_meth = (
        "The project follows an iterative Machine Learning Engineering lifecycle adhering to the CRISP-ML(Q) (Cross-Industry Standard "
        "Process for Data Mining with Quality Assurance) framework. The methodology comprises five disciplined phases:\n"
        "Phase 1 - Data Engineering: Curating 162,817 two-second epochs from the SEED-VIG repository, extracting the 5 selected channels (FT7, FT8, T7, PZ, O2), and "
        "synchronizing continuous PERCLOS eye-tracking labels.\n"
        "Phase 2 - Architectural Modeling: Implementing the 1D U-Net SEN denoiser, EEGNet spatial backbone, 2-layer BiLSTM, and self-attention "
        "module in PyTorch.\n"
        "Phase 3 - Strict Cross-Validation: Executing 23 complete training and evaluation cycles under Leave-One-Subject-Out protocol on NVIDIA "
        "RTX GPUs, logging fold-by-fold metrics.\n"
        "Phase 4 - Stress Testing & Robustness Analysis: Injecting synthetic Gaussian noise (0% to 50%) to evaluate noise suppression and executing "
        "systematic ablation studies.\n"
        "Phase 5 - Edge Deployment & Dashboard Engineering: Developing a reactive web dashboard with Streamlit and Plotly to demonstrate real-time "
        "inference and alarm generation."
    )
    add_styled_para(doc, p_meth)

    add_heading_2(doc, "3.4 Technology Stack")
    p_tech = "The computational infrastructure, hardware specifications, and deep learning software dependencies are detailed in Tables 3.2 and 3.3."
    add_styled_para(doc, p_tech)

    # Table 3.2 Hardware Environment
    add_caption(doc, "Table 3.2: Hardware Computing and Edge Workstation Specifications", is_table=True)
    tbl_hw = doc.add_table(rows=1, cols=3)
    tbl_hw.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_hw, color="94A3B8")
    for i, h in enumerate(["Hardware Subsystem", "Component Specification", "Operational Role"]):
        c = tbl_hw.rows[0].cells[i]
        c.paragraphs[0].text = h
        set_cell_shading(c, "F1F5F9")
        set_cell_margins(c, top=80, bottom=80, left=80, right=80)
        format_paragraph(c.paragraphs[0], size_pt=9.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)

    hw_rows = [
        ("Host Processor (CPU)", "AMD Ryzen 7 5700U (8 Cores, 16 Threads @ 1.8-4.3 GHz)", "Preprocessing, Buffer Ingestion, CPU Latency Benchmarking"),
        ("Dedicated GPU", "NVIDIA GeForce RTX 3050 Laptop GPU (6 GB GDDR6 VRAM)", "Mixed-Precision Neural Training & Real-Time Edge Inference"),
        ("System RAM", "16.0 GB DDR4 High-Speed Memory @ 3200 MHz", "Full 162,817 Epoch In-Memory Caching & Matrix Batching"),
        ("Storage Subsystem", "512 GB PCIe 4.0 NVMe Solid State Drive", "Rapid Epoch Access & Checkpoint Weight Storage")
    ]
    for r_data in hw_rows:
        row = tbl_hw.add_row()
        for i, val in enumerate(r_data):
            c = row.cells[i]
            c.paragraphs[0].text = val
            set_cell_margins(c, top=60, bottom=60, left=60, right=60)
            format_paragraph(c.paragraphs[0], size_pt=9, bold=(i==0))

    # Table 3.3 Software Stack
    add_caption(doc, "Table 3.3: Software Frameworks, Deep Learning Libraries, and Dependencies", is_table=True)
    tbl_sw = doc.add_table(rows=1, cols=3)
    tbl_sw.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_sw, color="94A3B8")
    for i, h in enumerate(["Software Layer", "Framework & Version", "Functional Purpose"]):
        c = tbl_sw.rows[0].cells[i]
        c.paragraphs[0].text = h
        set_cell_shading(c, "F1F5F9")
        set_cell_margins(c, top=80, bottom=80, left=80, right=80)
        format_paragraph(c.paragraphs[0], size_pt=9.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)

    sw_rows = [
        ("Deep Learning Framework", "PyTorch 2.0.1 + CUDA 11.8 + cuDNN 8.7", "Neural Network Modeling, Backpropagation & GPU Acceleration"),
        ("Mixed Precision Acceleration", "PyTorch AMP (torch.cuda.amp)", "FP16 Tensor Core Inference & Latency Minimization"),
        ("Numerical Computation", "NumPy 1.24.3, SciPy 1.10.1", "Multi-Channel Time-Series Manipulation & Signal Processing"),
        ("Machine Learning Metrics", "Scikit-Learn 1.2.2", "LOSO-CV Partitioning, ROC Curves, Confusion Matrices"),
        ("Interactive Dashboard", "Streamlit 1.22.0 (ASGI Web Server)", "Live Streaming Oscilloscope UI, Alerts, & Telemetry"),
        ("Real-Time Visualization", "Plotly 5.14.1, Matplotlib 3.7.1", "Dynamic Signal Plotting, Attention Heatmaps & Charts")
    ]
    for r_data in sw_rows:
        row = tbl_sw.add_row()
        for i, val in enumerate(r_data):
            c = row.cells[i]
            c.paragraphs[0].text = val
            set_cell_margins(c, top=60, bottom=60, left=60, right=60)
            format_paragraph(c.paragraphs[0], size_pt=9, bold=(i==0))

    add_heading_2(doc, "3.5 Gantt Chart and Process Model")
    p_gantt = (
        "The project execution schedule followed an iterative staged process model over a 14-month engineering roadmap "
        "spanning Semester VII (Project Part I) and Semester VIII (Project Part II). The schedule allocations, work breakdown "
        "structure (WBS), and milestones are visualized in the Gantt chart shown in Figure 3.1."
    )
    add_styled_para(doc, p_gantt)

    # Insert Figure 3.1 Gantt
    fig_path = os.path.join(os.getcwd(), 'bluebook_figures', 'fig_gantt.png')
    if os.path.exists(fig_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(4)
        run_img = p_img.add_run()
        run_img.add_picture(fig_path, width=Inches(5.5))
        add_caption(doc, "Figure 3.1: Project Work Breakdown Structure and Timeline Gantt Chart")

    add_heading_2(doc, "3.6 System Analysis")
    p_sys = (
        "System analysis was conducted through three complementary software engineering modeling perspectives:\n\n"
        "1. Functional Model: Decomposes the operational lifecycle into discrete functional transformations: signal ingestion, "
        "temporal windowing, z-score standardization, 1D U-Net neural enhancement, spatial depthwise filtering, recurrent sequence "
        "propagation, attention context pooling, and thresholded alarm assertion.\n\n"
        "2. Structural Model: Defines the hierarchical modular structure of the Python software package. The system is partitioned into: "
        "backend/simulator.py (responsible for asynchronous circular buffering and sample streaming), evaluation/models/bci_model.py "
        "(defining the PyTorch nn.Module graph of SEN, EEGNet, and BiLSTMAttention), backend/inference.py (managing model weight caching, "
        "GPU tensor mapping, and EMA probability smoothing), and app.py (implementing the reactive Streamlit web dashboard).\n\n"
        "3. Behavioral Model: Formulates the system as a Finite State Machine (FSM) transitioning between four primary states: "
        "[UNINITIALIZED] -> [STREAMING_ACQUISITION] -> [EVALUATING_VIGILANCE] -> [ALERT_ACTIVE]. When smoothed drowsy probability exceeds "
        "0.50 for three consecutive seconds, the state machine triggers an emergency in-cabin alarm transition, reverting to normal monitoring "
        "only when the operator demonstrates sustained alertness."
    )
    for seg in p_sys.split('\n\n'):
        add_styled_para(doc, seg)

    doc.add_page_break()

print("Chapter 3 module loaded.")
