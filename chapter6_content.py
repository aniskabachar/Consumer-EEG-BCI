# -*- coding: utf-8 -*-
"""
chapter6_content.py
Conclusion, Future Scope, References, Research Paper Summary, and Appendices A, B, C.
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

def add_chapter_6(doc):
    add_heading_1(doc, "Chapter 6: Conclusion & Future Scope")

    add_heading_2(doc, "6.1 Summary of Work Completed")
    p1 = (
        "This project report presented the comprehensive engineering and validation of NeuroVigi, an end-to-end deep learning "
        "framework designed specifically for robust, real-time driver vigilance estimation using only five EEG channels "
        "(FT7, FT8, T7, PZ, O2). By directly resolving the Consumer BCI Trilemma—balancing electrode economy, noise resilience, and "
        "cross-subject generalizability—the project establishes a viable pathway for translating clinical neuroimaging into life-saving "
        "automotive safety systems."
    )
    add_styled_para(doc, p1)

    p2 = (
        "Principal Technical Achievements of Project Part I:\n"
        "1. Curated 5-Channel Frontal Corpus: Successfully extracted and normalized 162,817 two-second epochs (over 90 hours of recording) "
        "from the 23-subject SEED-VIG driving benchmark, synchronized with continuous PERCLOS eye-tracking ground truth.\n"
        "2. Integrated 1D U-Net SEN Architecture: Embedded a 101,968-parameter convolutional denoising network with a global residual skip "
        "connection, enabling raw-domain temporal waveform reconstruction without detached offline matrix factorizations.\n"
        "3. Spatio-Temporal Attention Classification: Formulated an EEGNet convolutional backbone coupled with a two-layer Bidirectional "
        "LSTM and temporal self-attention scoring mechanism, totaling 156,739 parameters.\n"
        "4. Strict Zero-Calibration Validation: Executed complete 23-fold Leave-One-Subject-Out Cross-Validation, achieving a mean cross-subject "
        "accuracy of 79.70% +- 8.02%, an AUC-ROC of 0.8168, and a high alert specificity of 87.53%.\n"
        "5. Demonstrable Noise Resilience: Confirmed through synthetic stress tests that the SEN module retains 81.13% accuracy under 50% "
        "additive Gaussian noise (6 dB SNR), outperforming un-denoised baselines by +27.73%.\n"
        "6. Ultra-Low Edge Latency: Verified an end-to-end inference budget of only 4.18 ms on edge GPU and 12.30 ms on edge CPU, consuming less "
        "than 0.5% of the sliding window update budget."
    )
    add_styled_para(doc, p2)

    add_heading_2(doc, "6.2 Future Scope")
    p3 = (
        "Building upon the foundational achievements of Project Part I, the roadmap for Project Part II and future research encompasses "
        "four advanced directions:\n\n"
        "1. Self-Supervised Masked Autoencoder Pre-Training: To further reduce inter-subject variance, we plan to implement a self-supervised "
        "time-series masked autoencoder (MAE). By pre-training the SEN encoder on massive unlabelled public EEG repositories (e.g., TUH EEG, PhysioNet) "
        "to reconstruct masked temporal patches, the network will learn universal neurodynamic representations prior to supervised fine-tuning.\n\n"
        "2. Domain-Adversarial Neural Adaptation: We will integrate a gradient reversal layer (GRL) domain discriminator into the classifier head. "
        "By training the feature extractor to deceive the domain discriminator regarding subject identity, the model will learn strictly subject-invariant "
        "representations, lifting accuracy for challenging subjects.\n\n"
        "3. Continuous Vigilance Score Regression: Extending the terminal classification head from binary state predictions to continuous "
        "PERCLOS fatigue regression, providing granular vigilance indexes for progressive ADAS torque feedback.\n\n"
        "4. Physical Wearable Headband Hardware Prototyping: Fabricating a physical forehead headband embedding five dry Ag/AgCl polymer electrodes, "
        "an ADS1299 bio-potential analog front-end (AFE), and an ESP32-S3 microcontroller streaming real-time Bluetooth Low Energy (BLE) packets to "
        "the in-cabin vehicle gateway."
    )
    for seg in p3.split('\n\n'):
        add_styled_para(doc, seg)

    doc.add_page_break()

    # =========================================================================
    # REFERENCES
    # =========================================================================
    add_heading_1(doc, "References")
    
    refs = [
        "[1] World Health Organization, “Global status report on road safety 2023,” WHO Press, Geneva, Switzerland, Tech. Rep., 2023.",
        "[2] W.-L. Zheng and B.-L. Lu, “A multimodal approach to estimating vigilance using EEG and forehead EOG,” Journal of Neural Engineering, vol. 14, no. 2, art. 026017, pp. 1–15, 2017. doi: 10.1088/1741-2552/aa5a98.",
        "[3] Y. Dong, Z. Hu, K. Uchimura, and N. Murayama, “Driver inattention monitoring system for intelligent vehicles: A review,” IEEE Transactions on Intelligent Transportation Systems, vol. 12, no. 2, pp. 596–614, 2021. doi: 10.1109/TITS.2010.2092770.",
        "[4] R.-N. Duan, J.-Y. Zhu, and B.-L. Lu, “Differential entropy feature for EEG-based vigilance estimation,” in Proc. 34th Annu. Int. Conf. IEEE Eng. Med. Biol. Soc. (EMBC), San Diego, CA, USA, 2012, pp. 6620–6623. doi: 10.1109/EMBC.2012.6347512.",
        "[5] R. Chai, G. R. Naik, T. N. Nguyen, S. H. Ling, Y. Tran, A. Craig, and H. T. Nguyen, “Driver fatigue classification with independent component by entropy rate bound minimization analysis in an EEG-based system,” IEEE Journal of Biomedical and Health Informatics, vol. 21, no. 3, pp. 715–724, 2016. doi: 10.1109/JBHI.2016.2532354.",
        "[6] V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. J. Lance, “EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces,” Journal of Neural Engineering, vol. 15, no. 5, art. 056013, 2018. doi: 10.1088/1741-2552/aace8c.",
        "[7] R. T. Schirrmeister, J. T. Springenberg, L. D. J. Fiederer, M. Glasstetter, K. Eggensperger, M. Tangermann, F. Hutter, W. Burgard, and T. Ball, “Deep learning with convolutional neural networks for EEG decoding and visualization,” Human Brain Mapping, vol. 38, no. 11, pp. 5391–5420, 2017. doi: 10.1002/hbm.23730.",
        "[8] Y. Li, W. Zheng, Y. Zong, Z. Cui, T. Zhang, and X. Zhou, “A spatiotemporal attention network for multi-channel EEG-based emotion recognition,” IEEE Transactions on Cognitive and Developmental Systems, vol. 14, no. 3, pp. 987–998, 2022. doi: 10.1109/TCDS.2021.3090680.",
        "[9] H. Zhang, M. Wei, W. Zhao, and N. Gao, “EEGDenoiseNet: A benchmark dataset for deep learning solutions to EEG denoising,” Journal of Neural Engineering, vol. 18, no. 5, art. 056057, 2021. doi: 10.1088/1741-2552/ac2bf8.",
        "[10] Y. Song, D. Zheng, Q. Liu, and L. Gao, “Transformer-based spatial-temporal feature learning for EEG-based driver drowsiness detection,” IEEE Transactions on Intelligent Transportation Systems, vol. 23, no. 10, pp. 19201–19211, 2022. doi: 10.1109/TITS.2022.3168231.",
        "[11] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 30, Long Beach, CA, USA, 2017, pp. 5998–6008.",
        "[12] A. Craig, Y. Tran, N. Wijesuriya, and H. Nguyen, “Analysis of EEG changes in the development of fatigue during driving,” Neuroscience Letters, vol. 518, no. 2, pp. 77–81, 2012. doi: 10.1016/j.neulet.2012.04.053.",
        "[13] J. Cui, Z. Lan, Y. Liu, R. Li, F. Li, and O. Sourina, “A compact and interpretable convolutional neural network for cross-subject driver drowsiness detection,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 30, pp. 817–828, 2022. doi: 10.1109/TNSRE.2022.3160682.",
        "[14] I. Albuquerque, J. Monteiro, T. H. Falk, and T. Falk, “Generalizing deep learning models for EEG-based cognitive workload estimation across days and sessions,” IEEE Transactions on Biomedical Engineering, vol. 67, no. 7, pp. 1977–1986, 2020. doi: 10.1109/TBME.2019.2952445.",
        "[15] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” in Proc. Int. Conf. Med. Image Comput. Comput.-Assist. Interv. (MICCAI), Munich, Germany, 2015, pp. 234–241. doi: 10.1007/978-3-319-24574-4_28.",
        "[16] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997. doi: 10.1162/neco.1997.9.8.1735.",
        "[17] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in International Conference on Learning Representations (ICLR), New Orleans, LA, USA, 2019.",
        "[18] National Highway Traffic Safety Administration (NHTSA), “Drowsy driving research and countermeasures,” U.S. Department of Transportation, Washington, DC, USA, Tech. Rep. DOT HS 812 446, 2020."
    ]
    for r in refs:
        add_styled_para(doc, r, size_pt=10, line_spacing=1.3, space_after_pt=6)

    doc.add_page_break()

    # =========================================================================
    # RESEARCH PAPER SUMMARY
    # =========================================================================
    add_heading_1(doc, "Research Paper")
    add_styled_para(doc, "Title: NeuroVigi: Robust Cross-Subject Driver Vigilance Estimation via Frontal EEG Denoising and Spatio-Temporal Attention\n"
                         "Authors: Anis Kabachar, Nideesh Kayal, Dr. Faculty Advisor\n"
                         "Affiliation: Department of Computer Engineering, Thakur College of Engineering & Technology, Mumbai, India\n"
                         "Target Publication: IEEE Transactions on Intelligent Transportation Systems / IEEE EMBC Conference",
                    size_pt=11, bold=True, space_after_pt=14)

    p_pap = (
        "Paper Executive Abstract & Contributions:\n"
        "This research paper formulates NeuroVigi, addressing the Consumer BCI Trilemma in vehicular safety. The paper establishes three theoretical "
        "and empirical contributions:\n"
        "1. Mathematical Residual Denoising Formulation: Demonstrates that an integrated 1D U-Net SEN module with residual skip connection learns "
        "a zero-mean perturbation field F_theta(X) ~= -N, preserving pristine cerebral phase dynamics as N -> 0 while attenuating ocular artifacts.\n"
        "2. Strict 23-Fold Leave-One-Subject-Out Benchmark: Evaluates 162,817 epochs from SEED-VIG across 23 subjects, demonstrating 79.70% accuracy, "
        "0.8168 AUC-ROC, and 87.53% specificity on completely held-out subjects without calibration.\n"
        "3. Quantifiable Noise Robustness: Proves that the SEN module retains 81.13% accuracy at 50% additive Gaussian noise (6 dB SNR), whereas "
        "un-denoised models collapse to 53.40% (chance level).\n\n"
        "The full self-contained LaTeX manuscript for the paper has been compiled in standard IEEE conference two-column format with inline "
        "bibliographies, TikZ architectural flowcharts, and vectorized figure embeddings."
    )
    for seg in p_pap.split('\n\n'):
        add_styled_para(doc, seg)

    doc.add_page_break()

    # =========================================================================
    # APPENDIX A: ABBREVIATIONS AND SYMBOLS
    # =========================================================================
    add_heading_1(doc, "Appendix A: Abbreviation and Symbols")
    add_styled_para(doc, "This appendix compiles standard abbreviations, neurophysiological acronyms, and mathematical symbols utilized throughout the report.", size_pt=11, italic=True)

    tbl_app_a = doc.add_table(rows=1, cols=3)
    tbl_app_a.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_app_a, color="94A3B8")
    for i, h in enumerate(["Abbreviation / Symbol", "Full Technical Description", "Domain / Context"]):
        c = tbl_app_a.rows[0].cells[i]
        c.paragraphs[0].text = h
        set_cell_shading(c, "F1F5F9")
        set_cell_margins(c, top=60, bottom=60, left=40, right=40)
        format_paragraph(c.paragraphs[0], size_pt=9, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)

    app_a_rows = [
        ("BCI", "Brain-Computer Interface", "Neurotechnology & Biomedical Systems"),
        ("EEG", "Electroencephalography", "Electrophysiological Neural Recording"),
        ("PERCLOS", "Percentage of Eye Closure", "Optical Eye-Tracking Fatigue Metric"),
        ("SEN", "Signal Enhancement Network", "1D U-Net Convolutional Denoising Module"),
        ("LOSO-CV", "Leave-One-Subject-Out Cross-Validation", "Zero-Calibration Machine Learning Evaluation"),
        ("BiLSTM", "Bidirectional Long Short-Term Memory", "Recurrent Temporal Sequence Modeling"),
        ("SNR", "Signal-to-Noise Ratio (dB)", "Signal Quality & Noise Robustness"),
        ("AUC-ROC", "Area Under Receiver Operating Characteristic", "Binary Classifier Discrimination Metric"),
        ("DMS", "Driver Monitoring System", "Vehicular Active Safety Systems"),
        ("Fm-theta", "Frontal Midline Theta (4 - 8 Hz)", "Neurological Marker of Mental Fatigue"),
        ("X in R^{B x 5 x 400}", "Input EEG Epoch Batch Tensor", "Mathematical Data Matrix Representation"),
        ("alpha_t", "Dynamic Temporal Attention Weight", "Temporal Relevance Scoring Scalar")
    ]
    for r_data in app_a_rows:
        row = tbl_app_a.add_row()
        for i, val in enumerate(r_data):
            c = row.cells[i]
            c.paragraphs[0].text = val
            set_cell_margins(c, top=40, bottom=40, left=40, right=40)
            format_paragraph(c.paragraphs[0], size_pt=8.5, bold=(i==0),
                             align=WD_ALIGN_PARAGRAPH.CENTER if i==0 else WD_ALIGN_PARAGRAPH.LEFT)

    # Detailed Table A.1 Hyperparameters
    add_caption(doc, "Table A.1: Detailed Model Hyperparameters and Optimization Configurations", is_table=True)
    tbl_hyp = doc.add_table(rows=1, cols=2)
    tbl_hyp.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_hyp, color="94A3B8")
    for i, h in enumerate(["Hyperparameter", "Configured Training Value"]):
        c = tbl_hyp.rows[0].cells[i]
        c.paragraphs[0].text = h
        set_cell_shading(c, "F1F5F9")
        set_cell_margins(c, top=60, bottom=60, left=40, right=40)
        format_paragraph(c.paragraphs[0], size_pt=9, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)

    hyp_rows = [
        ("Epoch Duration (Tw)", "2.0 seconds"),
        ("Sampling Frequency (fs)", "200 Hz"),
        ("Samples per Channel (L)", "400 discrete time steps"),
        ("Electrode Channels (C)", "5 (FT7, FT8, T7, PZ, O2)"),
        ("Batch Size (B)", "128 epochs"),
        ("Training Epochs (E)", "80 epochs"),
        ("Optimizer", "AdamW (beta_1=0.9, beta_2=0.999)"),
        ("Initial Learning Rate (eta_0)", "1 x 10^-3"),
        ("Minimum Learning Rate (eta_min)", "1 x 10^-6"),
        ("Learning Rate Schedule", "Cosine Annealing"),
        ("Weight Decay (lambda)", "1 x 10^-4"),
        ("Gradient Clipping Max Norm", "1.0"),
        ("Dropout Rates", "SEN: None, EEGNet: 0.25, Head: 0.50"),
        ("BiLSTM Hidden Dimension (H)", "64 (128 Bidirectional)"),
        ("Attention Dimension", "64"),
        ("PERCLOS Alert Threshold", "<= 0.35"),
        ("PERCLOS Drowsy Threshold", ">= 0.50")
    ]
    for r_data in hyp_rows:
        row = tbl_hyp.add_row()
        for i, val in enumerate(r_data):
            c = row.cells[i]
            c.paragraphs[0].text = val
            set_cell_margins(c, top=40, bottom=40, left=40, right=40)
            format_paragraph(c.paragraphs[0], size_pt=8.5, bold=(i==0))

    doc.add_page_break()

    # =========================================================================
    # APPENDIX B: DEFINITIONS
    # =========================================================================
    add_heading_1(doc, "Appendix B: Definitions")
    defs = [
        ("1. Brain-Computer Interface (BCI)", "A hardware and software communications system that measures Central Nervous System (CNS) activity and converts it into artificial outputs that replace, restore, enhance, supplement, or improve natural CNS outputs."),
        ("2. Electroencephalography (EEG)", "An electrophysiological monitoring method to record electrical activity of the brain, measured via macroscopic ionic current flows within post-synaptic dendritic trees of cortical pyramidal neurons."),
        ("3. Percentage of Eye Closure (PERCLOS)", "The proportion of time that the eyes are closed at least 80% (or 70%) over a specified observation window, widely recognized by transportation safety agencies as the most validated optical measure of driver drowsiness."),
        ("4. Signal Enhancement Network (SEN)", "A specialized 1D U-Net convolutional encoder-decoder neural network designed to map noisy, contaminated multi-channel EEG time-series into clean, reconstructed temporal waveforms with minimal phase distortion."),
        ("5. Depthwise Separable Spatial Convolution", "A convolutional factorization that applies spatial filtering across electrode channels independently of temporal convolutions, dramatically reducing parameter counts while learning cortical spatial topologies."),
        ("6. Leave-One-Subject-Out Cross-Validation (LOSO-CV)", "A cross-validation protocol wherein a model is trained on N-1 subjects and evaluated exclusively on the completely held-out Nth subject, repeated across all N folds to ensure unbiased zero-calibration evaluation."),
        ("7. Temporal Self-Attention Mechanism", "A learnable mathematical function that maps temporal hidden state vectors into dynamic scalar relevance weights, enabling the network to focus computational capacity on informative micro-states."),
        ("8. Exponential Moving Average (EMA) Hysteresis", "A recursive digital smoothing filter applied to raw vigilance probabilities to suppress transient spurious misclassifications and enforce sustained condition criteria before asserting in-cabin alarms.")
    ]
    for term, d_txt in defs:
        add_styled_para(doc, f"{term}:\n{d_txt}", size_pt=11, line_spacing=1.3, space_after_pt=10)

    doc.add_page_break()

    # =========================================================================
    # APPENDIX C: LIST OF PUBLICATIONS
    # =========================================================================
    add_heading_1(doc, "Appendix C: List of Publications")
    add_styled_para(doc, "Publications Resulting from this Research Work (IEEE Format):", size_pt=12, bold=True, space_after_pt=12)

    pub_txt = (
        "[1] Anis Kabachar, Nideesh Kayal, and Dr. Faculty Advisor, “NeuroVigi: Robust Cross-Subject Driver Vigilance "
        "Estimation via Frontal EEG Denoising and Spatio-Temporal Attention,” in Manuscript Prepared for Submission to "
        "IEEE Transactions on Intelligent Transportation Systems / IEEE Engineering in Medicine and Biology Society (EMBC), "
        "Thakur College of Engineering & Technology, Mumbai, India, Academic Year 2026–2027.\n\n"
        "Status: Full manuscript compiled, benchmarked, and formatted under standard IEEE double-column conference guidelines. "
        "Code, pre-trained weights, and real-time Streamlit demonstration dashboard archived in project repository."
    )
    add_styled_para(doc, pub_txt, size_pt=11, line_spacing=1.4)

print("Chapter 6 module loaded.")
