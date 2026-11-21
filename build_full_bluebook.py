# -*- coding: utf-8 -*-
"""
build_full_bluebook.py
Master script to assemble the complete 40-50 page TCET Bluebook Project Report (Part I).
"""

import os
import sys
import docx

from make_bluebook_docx import create_bluebook
from chapter1_content import add_chapter_1
from chapter2_content import add_chapter_2
from chapter3_content import add_chapter_3
from chapter4_content import add_chapter_4
from chapter5_content import add_chapter_5
from chapter6_content import add_chapter_6

def main():
    print("=" * 60)
    print("Building TCET Bluebook Project Report (Part I)...")
    print("=" * 60)

    # 1. Front Matter (Title, Guide, Certificate, Approval, Acknowledgement, Plagiarism, Index, Lists, Abstract)
    print("Step 1: Adding Front Matter...")
    doc = create_bluebook()

    # 2. Chapter 1: Introduction
    print("Step 2: Adding Chapter 1 (Introduction)...")
    add_chapter_1(doc)

    # 3. Chapter 2: Literature Survey & Proposed System
    print("Step 3: Adding Chapter 2 (Literature Survey & Proposed System)...")
    add_chapter_2(doc)

    # 4. Chapter 3: Requirement Gathering, Analysis and Planning
    print("Step 4: Adding Chapter 3 (Requirements, Feasibility, Tech Stack, Gantt)...")
    add_chapter_3(doc)

    # 5. Chapter 4: System Design and Experimental Setup
    print("Step 5: Adding Chapter 4 (Design, Architecture, DFD, UML, Algorithms, Testing)...")
    add_chapter_4(doc)

    # 6. Chapter 5: Results & Discussion
    print("Step 6: Adding Chapter 5 (Results, 23 Folds, Noise Stress, Ablation, Latency)...")
    add_chapter_5(doc)

    # 7. Chapter 6: Conclusion, References, Paper, Appendices A, B, C
    print("Step 7: Adding Chapter 6, References, Paper, and Appendices...")
    add_chapter_6(doc)

    # 8. Save the Document
    output_docx = os.path.join(os.getcwd(), "NeuroVigi_TCET_Bluebook_Project_Report.docx")
    doc.save(output_docx)
    print(f"SUCCESS! Master Bluebook saved at: {output_docx}")

    # Also save a .doc extension copy for user convenience
    output_doc = os.path.join(os.getcwd(), "NeuroVigi_TCET_Bluebook_Project_Report.doc")
    try:
        import shutil
        shutil.copyfile(output_docx, output_doc)
        print(f"Convenience copy saved at: {output_doc}")
    except Exception as e:
        print(f"Note on .doc copy: {e}")

if __name__ == '__main__':
    main()
