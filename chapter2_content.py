# -*- coding: utf-8 -*-
"""
chapter2_content.py
Literature Survey, Gap Analysis, Taxonomy Table, and Proposed System description.
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

def add_chapter_2(doc):
    add_heading_1(doc, "Chapter 2: Literature Survey & Proposed System")

    add_heading_2(doc, "2.1 Literature Review of Existing System")
    p1 = (
        "The application of electroencephalography (EEG) to cognitive state monitoring and operator fatigue estimation has "
        "been investigated for over four decades. As neuroimaging and computational hardware evolved, methodologies progressed "
        "from basic spectral bandpower thresholds to hand-crafted machine learning classifiers, and ultimately to contemporary "
        "end-to-end deep neural network architectures. To establish the context and scientific lineage of NeuroVigi, this section "
        "provides a rigorous critical review of ten seminal research works in the domain of EEG-based vigilance decoding."
    )
    add_styled_para(doc, p1)

    studies = [
        ("Zheng and Lu (2017) - Multimodal Vigilance Estimation using EEG and Forehead EOG [IEEE JNE]",
         "Zheng and Lu established the landmark SEED-VIG benchmark dataset, investigating multimodal vigilance estimation across 23 "
         "subjects in a continuous driving simulator. They extracted Differential Entropy (DE) features across canonical frequency bands "
         "(Delta, Theta, Alpha, Beta, Gamma) from 17 EEG channels and forehead EOG electrodes. Using Support Vector Machines (SVM) and "
         "Continuous Graphical Bayesian Networks (GBN), they achieved an average accuracy of 87.5% using DE-GBN under a split-half within-subject "
         "evaluation protocol. Critical Limitation: Their evaluation scheme trained and tested on split sessions from the same subjects. When tested "
         "in cross-subject configurations without prior subject data, model accuracy degraded by over 18%, highlighting the severe impact of inter-subject variability."),

        ("Chai et al. (2016) - Driver Fatigue Classification with Independent Component Entropy [IEEE JBHI]",
         "Chai et al. investigated driver fatigue using 32-channel wet scalp EEG. They applied an Autoregressive-Continuous Wavelet Transform (AR-CWT) "
         "pipeline coupled with Sparse-Deep Belief Bayesian Networks (Sparse-Deep BBN). To eliminate ocular blinks and muscular movement artifacts, "
         "they executed offline Independent Component Analysis (ICA). Their framework achieved 82.3% cross-subject accuracy across a cohort of "
         "drivers. Critical Limitation: ICA requires decomposing high-density multi-channel recordings over multi-second temporal batches using iterative "
         "matrix inversions. This offline approach cannot be executed online within low-power edge microcontrollers with few electrodes."),

        ("Lawhern et al. (2018) - EEGNet: A Compact Convolutional Neural Network for EEG BCI [JNE]",
         "Lawhern et al. introduced EEGNet, a highly compact convolutional neural network architecture tailored for brain-computer interfaces. "
         "By employing 2D temporal convolutions followed by depthwise separable spatial convolutions and pointwise groupings, EEGNet achieved "
         "competitive accuracy across P300, visual evoked potentials, and motor imagery benchmarks while requiring only 18,410 trainable parameters. "
         "Critical Limitation: EEGNet was primarily evaluated in controlled, high-channel (16-64 channels) laboratory environments under within-subject "
         "cross-validation. It lacks built-in denoising capacity for raw, contaminated low-channel dry-sensor recordings."),

        ("Schirrmeister et al. (2017) - Deep Learning with Convolutional Neural Networks for EEG Decoding [Human Brain Mapping]",
         "Schirrmeister et al. designed ShallowConvNet and DeepConvNet, demonstrating that convolutional neural networks can learn filter-bank "
         "spatial-temporal representations directly from raw multi-channel time-series, outperforming traditional Filter Bank Common Spatial Patterns "
         "(FBCSP). Their architectures proved that end-to-end gradient backpropagation could capture both bandpass spectral filters and spatial projections. "
         "Critical Limitation: Both architectures were optimized for high-density 64-channel medical caps and suffer noticeable performance drops when "
         "restricted to sparse 3-5 channel frontal montages."),

        ("Zhang et al. (2021) - EEGDenoiseNet: Deep Learning Benchmark for EEG Denoising [JNE]",
         "Zhang et al. formulated the EEGDenoiseNet benchmark, proving that deep 1D convolutional autoencoders and U-Net architectures can effectively "
         "disentangle electrooculographic (EOG) eye-blinks and electromyographic (EMG) jaw-clench artifacts from pure underlying cerebral signals, achieving "
         "Signal-to-Noise Ratio (SNR) improvements exceeding +8.4 dB. Critical Limitation: EEGDenoiseNet operated exclusively as a detached, stand-alone "
         "preprocessing filter evaluated on synthetic linear signal mixtures; it did not couple the denoising stage to an integrated downstream classification head."),

        ("Song et al. (2022) - Spatial-Temporal Transformer for Driver Drowsiness Detection [IEEE TITS]",
         "Song et al. developed a spatial-temporal Transformer encoder that applies self-attention across spatial electrode graph nodes and temporal "
         "sub-windows simultaneously. Evaluated across multi-channel driving datasets, their model achieved 78.4% subject-independent accuracy. "
         "Critical Limitation: The Transformer architecture required over 850,000 parameters and heavy multi-head attention matrix multiplications, "
         "inducing inference latencies exceeding 45 milliseconds per epoch, which strains automotive embedded edge microprocessors."),

        ("Li et al. (2022) - Spatiotemporal Attention Networks for Multi-Channel EEG Decoding [IEEE TCDS]",
         "Li et al. combined 3D convolutional networks with Bidirectional LSTM sequence modeling and channel attention mechanisms to capture dynamic "
         "cortical changes, reaching 85.2% accuracy in cognitive state classification across 62 channels. Critical Limitation: High channel count dependence "
         "and an immense model size of 3.8 million parameters, entirely incompatible with consumer forehead headbands."),

        ("Albuquerque et al. (2020) - Generalizing Deep Models for Cognitive Workload across Sessions [IEEE TBME]",
         "Albuquerque et al. conducted extensive cross-session and cross-subject evaluations of cognitive workload, demonstrating that domain shifts "
         "in baseline resting alpha power significantly degrade cross-user generalization. They advocated for leave-one-subject-out validation and "
         "invariant feature representations as essential prerequisites for practical BCI deployment."),

        ("Craig et al. (2012) - Analysis of EEG Changes in the Development of Fatigue [Neuroscience Letters]",
         "Craig and colleagues conducted controlled driving simulator experiments to document the neurophysiological progression of fatigue. They proved "
         "that while posterior occipital alpha increases during passive relaxed wakefulness, frontal midline theta power (Fm-theta) and diffuse frontal alpha "
         "bursts represent the definitive neurophysiological biomarkers of involuntary cognitive vigilance loss."),

        ("Cui et al. (2022) - Compact and Interpretable CNN for Cross-Subject Drowsiness Detection [IEEE TNSRE]",
         "Cui et al. presented an interpretable convolutional architecture designed for cross-subject drowsiness classification. By analyzing learned spatial "
         "filter weights, they verified that frontal and central electrode locations contribute over 70% of discriminative vigilance information, providing "
         "strong empirical justification for lightweight frontal-only wearable montages.")
    ]

    for title, desc in studies:
        add_heading_3(doc, title)
        add_styled_para(doc, desc)

    add_heading_2(doc, "2.2 Limitations of Existing System & Gap Analysis")
    p_gap = (
        "A rigorous synthesis of prior literature reveals three systemic bottlenecks that have collectively hindered the commercial "
        "translation of EEG-based driver drowsiness detection systems—a phenomenon we formalize as the Consumer BCI Trilemma:\n"
        "1. High-Density Montage Bottleneck: State-of-the-art models in literature overwhelmingly rely on 16 to 64 electrode channels distributed "
        "across the entire scalp, including parietal, occipital, and temporal regions. In an automotive environment, user compliance demands a non-hair-bearing "
        "forehead montage (3 to 5 dry electrodes) that can be donned in seconds without conductive gels.\n"
        "2. Detached, Computationally Heavy Denoising: Prior works either ignore in-band environmental noise or rely on offline Independent Component "
        "Analysis (ICA) or Empirical Mode Decomposition (EMD). These multi-stage techniques require multi-second signal buffers and iterative matrix "
        "inversions that cannot execute in real-time on edge processors.\n"
        "3. Overly Optimistic Validation Schemes: Many published studies report high accuracy using within-subject or random epoch train/test splits. "
        "Such protocols suffer from catastrophic temporal autocorrelation and data leakage, failing entirely when applied to new, unseen drivers."
    )
    add_styled_para(doc, p_gap)

    # Add Table 2.1 Taxonomy Table
    add_caption(doc, "Table 2.1: Taxonomy and Comparative Analysis of Representative EEG Vigilance Estimators", is_table=True)
    tbl_tax = doc.add_table(rows=1, cols=7)
    tbl_tax.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_tax, color="94A3B8")

    headers = ["Study & Year", "Architecture", "Channels", "Montage Area", "Denoising", "Validation", "Accuracy / Metric"]
    for i, h in enumerate(headers):
        c = tbl_tax.rows[0].cells[i]
        c.paragraphs[0].text = h
        set_cell_shading(c, "F1F5F9")
        set_cell_margins(c, top=80, bottom=80, left=80, right=80)
        format_paragraph(c.paragraphs[0], size_pt=9.5, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)

    rows_tax = [
        ("Zheng & Lu (2017)", "DE + SVM / GBN", "17", "Whole Scalp + Forehead", "Offline EOG Filtering", "Split-Half (Within)", "87.5% (DE-GBN)"),
        ("Chai et al. (2016)", "AR-CWT + Sparse BBN", "32", "Whole Scalp", "Offline ICA", "Cross-Subject", "82.3% Acc"),
        ("Lawhern et al. (2018)", "EEGNet", "16-64", "Whole Scalp", "Standard Bandpass", "Within-Subject 4-Fold", "76.5% (ERP)"),
        ("Schirrmeister (2017)", "DeepConvNet", "64", "Whole Scalp", "Standard Bandpass", "Within-Subject", "75.1% Acc"),
        ("Zhang et al. (2021)", "1D CNN Autoencoder", "1", "Single Channel", "Deep Denoising", "Synthetic Benchmark", "+8.4 dB SNR"),
        ("Song et al. (2022)", "ST-Transformer", "17", "Whole Scalp", "Bandpass Filter", "Cross-Subject", "78.4% Acc"),
        ("Li et al. (2022)", "3D-CNN + BiLSTM", "62", "Whole Scalp", "Offline ICA", "Cross-Subject", "85.2% Acc"),
        ("NeuroVigi (Ours)", "SEN + EEGNet + BiLSTM + Attn", "5", "Temporal + Parieto-Occipital", "Integrated End-to-End SEN", "Strict 23-Fold LOSO-CV", "79.70% (0.817 AUC)")
    ]

    for r_data in rows_tax:
        row = tbl_tax.add_row()
        for i, val in enumerate(r_data):
            c = row.cells[i]
            c.paragraphs[0].text = val
            set_cell_margins(c, top=60, bottom=60, left=60, right=60)
            is_ours = ("NeuroVigi" in r_data[0])
            if is_ours:
                set_cell_shading(c, "E0F2FE")
            format_paragraph(c.paragraphs[0], size_pt=9, bold=is_ours,
                             align=WD_ALIGN_PARAGRAPH.CENTER if i in [2, 6] else WD_ALIGN_PARAGRAPH.LEFT)

    add_heading_2(doc, "2.3 Proposed System")
    p_prop = (
        "To decisively overcome the limitations identified in prior art, this project proposes NeuroVigi, a unified, two-stage deep "
        "learning framework engineered specifically for five-channel EEG vigilance decoding under real-world wearable constraints.\n\n"
        "Key Architectural Innovations:\n"
        "1. Few-Channel Montage: The system restricts input acquisition to five of the 17 SEED-VIG electrodes (the dataset has no frontal EEG channels): "
        "FT7 (Left Fronto-Temporal), FT8 (Right Fronto-Temporal), T7 (Left Temporal), PZ (Midline Parietal), and O2 (Right Occipital), as depicted "
        "in Figure 2.1. The temporal sites pick up lateral EOG and temporal activity, while PZ and O2 cover posterior alpha; PZ and O2 lie on hair-bearing scalp.\n"
        "2. Integrated Signal Enhancement Network (SEN): Rather than isolating artifact rejection into an offline step, NeuroVigi embeds a 1D U-Net "
        "encoder-decoder directly into the front of the neural pipeline. Equipped with a global residual skip connection, the SEN module dynamically "
        "reconstructs pristine neural waveforms in the temporal voltage domain, attenuating ocular and vibration artifacts end-to-end.\n"
        "3. Spatio-Temporal Convolutional Filtering: The denoised signal is processed through an EEGNet-inspired convolutional backbone with depthwise "
        "spatial filters that model cross-electrode relationships without mixing temporal frequency characteristics.\n"
        "4. Bidirectional Sequence & Temporal Self-Attention: A two-layer BiLSTM captures forward and backward temporal context across the 2-second "
        "epoch, followed by a temporal self-attention scoring head that dynamically weights the most diagnostically salient sub-intervals."
    )
    for seg in p_prop.split('\n\n'):
        add_styled_para(doc, seg)

    # Insert Figure 2.1 Montage
    fig_path = os.path.join(os.getcwd(), 'bluebook_figures', 'fig_montage.png')
    if os.path.exists(fig_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(4)
        run_img = p_img.add_run()
        run_img.add_picture(fig_path, width=Inches(3.8))
        add_caption(doc, "Figure 2.1: 5-Channel NeuroVigi Montage (FT7, FT8, T7, PZ, O2) on the SEED-VIG Layout")

    doc.add_page_break()

print("Chapter 2 module loaded.")
