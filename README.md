# SHINRA: Explainable AI for Diabetic Retinopathy Screening in Rural India

[![Platform](https://img.shields.io/badge/Platform-MATLAB%20%7C%20MATLAB%20Online-0076A8.svg)](https://matlab.mathworks.com)
[![Category](https://img.shields.io/badge/Competition-Smart%20India%20Hackathon-FF9933.svg)]()
[![Status](https://img.shields.io/badge/Status-Complete%20MATLAB--Native%20Release-brightgreen.svg)]()

> **SHINRA** is a quality-aware, evidence-based, explainable, and reliability-aware Diabetic Retinopathy (DR) screening system natively implemented in pure MATLAB for direct execution in **MATLAB Online** and standard MATLAB desktop environments.

---

## 1. Executive Summary & Clinical Context

Diabetic Retinopathy is a principal cause of preventable vision loss worldwide, particularly rampant in rural and low-resource regions of India where specialist ophthalmologists are severely scarce. Standard deep learning "black-box" classifiers face high failure rates in primary healthcare centers (PHCs) due to:
1. **Uncalibrated Overconfidence on Degraded Images**: Handheld cameras frequently yield blurry, underexposed, or occluded images that black-box models misclassify with high false confidence.
2. **Lack of Interpretable Biomarkers**: Doctors and health workers cannot verify whether an AI referral is grounded in actual lesions (microaneurysms, hemorrhages, exudates) or irrelevant background artifacts.
3. **Absence of Reliability-Aware Triage**: Systems lack mechanisms to distinguish routine autonomous decisions from uncertain cases requiring tele-ophthalmologist review.

**SHINRA addresses these challenges through a closed-loop, multi-modal screening pipeline:**

```
RAW FUNDUS IMAGE
       ↓
IMAGE QUALITY GATE (Sharpness, Illumination, Exposure, FOV)
       ↓
 [UNGRADABLE] ──────────────────────────→ ACTIONABLE RECAPTURE GUIDANCE
       ↓
 [BORDERLINE] ──────────────────────────→ ADAPTIVE RESTORATIVE ENHANCEMENT → RE-ASSESS
       ↓
   [GOOD]
       ↓
RETINAL PREPROCESSING & FOV ISOLATION (Illumination Normalization + CLAHE)
       ↓
ANATOMY LOCALIZATION (Optic Disc & Fovea / FAZ Center)
       ↓
VESSEL SEGMENTATION & MORPHOMETRY (U-Net / Matched Filters, Tortuosity, Density)
       ↓
MULTI-SCALE LESION EXTRACTION (Microaneurysms, Hemorrhages, Exudates, DME Risk, NV)
       ↓
5-LEVEL DR SEVERITY (No DR, Mild, Moderate, Severe, PDR + Referable DR P(DR >= 2))
       ↓
MULTI-MODAL EVIDENCE FUSION (Deep CNN + Quantified Biomarkers + Quality)
       ↓
RELIABILITY ESTIMATION & CALIBRATION (Calibrated Confidence × Quality × Agreement)
       ↓
GRAD-CAM VISUAL EXPLAINABILITY (Attention Heatmaps + Lesion Alignment + 6-Panel Panel)
       ↓
FINAL CLINICAL DISPOSITION (NOT REFERABLE / REFERABLE DR / CLINICAL TRIAGE)
       ↓
AUTOMATED PATIENT DIAGNOSTIC REPORT (TXT, CSV, MAT, Annotated PNG)
```

---

## 2. Dataset Inventory & Technical Audit

SHINRA dynamically scans and validates local datasets upon initialization:
- **IDRiD (Indian Diabetic Retinopathy Image Dataset)**:
  - `413` Training Images (`.jpg`) + `103` Testing Images (`.jpg`)
  - Ground-truth 5-class DR grades (0 to 4) and 3-class Macular Edema risk (0 to 2).
  - Used for supervised DR severity training, validation threshold calibration, and evidence fusion.
- **DRIVE (Digital Retinal Images for Vessel Extraction)**:
  - `20` Training Images (`.tif`) with manual vessel segmentation masks (`1st_manual`) and FOV masks.
  - `20` Testing Images (`.tif`) with FOV masks.
  - Used for supervised U-Net vascular segmentation and morphometry extraction.
- **APTOS 2019 Blind Screening Cohort**:
  - `1,928` Retinal fundus images (`.png`).
  - Evaluated as an unlabelled rural screening cohort for high-throughput batch screening and quality distribution testing (**no fabricated accuracy claims**).

---

## 3. Project Structure

```
SHINRA/
├── main.m                                # Master orchestration pipeline
├── demo.m                                # Interactive single-image demonstration
├── setup_SHINRA.m                        # Environment setup, path configuration & toolbox check
├── run_single_image.m                    # Single-image diagnostic entry point
├── run_batch_screening.m                 # High-throughput batch directory screener
├── README.md                             # System documentation
├── requirements.md                       # Toolbox dependencies & hardware requirements
│
├── config/
│   └── shinra_config.m                   # Central configuration struct, thresholds & seeds
│
├── src/
│   ├── data/                             # scan_datasets, load_idrid, load_drive, load_aptos, split_dataset
│   ├── preprocessing/                    # detectRetinalFOV, normalizeIllumination, enhanceRetina, preprocessRetina
│   ├── quality/                          # qualityFeatures, qualityScore, qualityClassifier, adaptiveEnhance, recaptureGuidance
│   ├── anatomy/                          # detectOpticDisc, locateFovea, visualizeAnatomy
│   ├── vessels/                          # buildVesselUNet, segmentVessels, vesselMorphometry, trainVesselSegmentation
│   ├── lesions/                          # detectMicroaneurysms, detectHemorrhages, detectExudates, analyzeNeovascularization
│   ├── severity/                         # buildSeverityCNN, predictDRSeverity, trainSeverityModel
│   ├── fusion/                           # extractEvidenceVector, fuseEvidence, trainFusionModel
│   ├── reliability/                      # calibrateConfidence, computeReliability, makeScreeningDecision
│   ├── explainability/                   # generateGradCAM, explanationAgreement, generateExplanationPanel
│   ├── evaluation/                       # evaluateClassifier, runAblationStudy, runRobustnessTest, runCrossDatasetEvaluation
│   ├── reporting/                        # generateClinicalReport, exportScreeningResults
│   └── utilities/                        # get_project_root, save_figure_safe
│
├── models/                               # Checkpoints (.mat) for vessels, severity, fusion
├── results/                              # Generated figures, clinical reports, predictions, metrics
├── tests/                                # Automated test suite (smoke_test.m + unit tests)
└── simulink/                             # SHINRA_Screening_System.slx + throughput simulation
```

---

## 4. Quick Start & Execution Guide

### Running on MATLAB Online or MATLAB Desktop

1. **Initialize the Environment**:
   ```matlab
   setup_SHINRA
   ```
   *Dynamically resolves root paths, adds all source folders to path, verifies toolboxes, and scans datasets.*

2. **Run Interactive Single-Image Demo**:
   ```matlab
   demo
   ```
   *Screens a sample clinical image, prints the patient diagnostic report, and generates the 6-panel explainability figure.*

3. **Screen Any Custom Fundus Image**:
   ```matlab
   result = run_single_image('SHINRAdataset/IDRid/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_001.jpg');
   ```

4. **Run High-Throughput Batch Screening**:
   ```matlab
   [summary_table, results] = run_batch_screening('SHINRAdataset/APTOS 2019', 50);
   ```

5. **Run Master Benchmark Pipeline**:
   ```matlab
   main
   ```
   *Runs dataset audit, ablation benchmark (Exp A through F), robustness stress-test, cross-dataset audit, demo, and Simulink capacity modeling.*

6. **Execute System Integration Test Suite**:
   ```matlab
   tests/smoke_test
   ```

---

## 5. Methodological Highlights

### A. Closed-Loop Quality Safety Gate
Extracts quantitative physical features: Laplacian blur variance, Tenengrad sharpness, mean illumination, underexposure/overexposure ratios, and retinal field-of-view coverage. Classifies images into **`GOOD`**, **`BORDERLINE`** (triggers targeted adaptive enhancement), or **`UNGRADABLE`** (generates plain-language feedback for ASHA workers e.g., *"Stabilize camera against patient forehead"*).

### B. Multi-Scale Biomarker Extraction
- **Microaneurysms**: Multi-scale circular Top-Hat / Black-Hat morphology ($r \in [2, 5]$ px) with vessel and optic disc exclusion.
- **Hemorrhages**: Multi-scale dark lesion segmentation ($r \in [6, 22]$ px) measuring total hemorrhagic area fraction.
- **Hard & Soft Exudates**: Multi-scale bright lesion segmentation with strict anatomical Optic Disc suppression and Foveal proximity DME risk grading.
- **Neovascularization**: Exploratory biomarker quantifying fine vessel branching clustering and vascular tortuosity.

### C. Multi-Modal Evidence Fusion & Reliability
Combines deep CNN probabilities with physical lesion burdens, vessel morphometry, and quality scores into a 16-dimensional evidence vector. Evaluates system reliability:
$$\text{Reliability} = \text{Confidence}_{\text{calibrated}} \times \left(\frac{\text{Quality}}{100}\right) \times \text{Agreement}$$
Triages output into **`NOT REFERABLE`**, **`REFERABLE DR`**, or **`UNCERTAIN / CLINICAL TRIAGE`**.

### D. Simulink Rural Screening Capacity Modeling
The Simulink model (`simulink/SHINRA_Screening_System.slx`) models a network of 25 rural vision kiosks operating 8 hours/day across 250 clinic days/year. Simulates queuing, quality-gate recaptures, and autonomous throughput (82% autonomous disposition / 18% specialist review), demonstrating system capacity supporting **100,000+ patient screenings annually**.

---

## 6. Formal Disclosure of Scope & Limitations

| Component | Status | Clinical Methodology |
| :--- | :--- | :--- |
| **5-Class DR Severity** | `IMPLEMENTED & TRAINED` | Trained on IDRiD clinical ground truth; 5-class ICDR scale + Referable DR ($DR \ge 2$). |
| **Retinal Vessel Segmentation** | `IMPLEMENTED & TRAINED` | U-Net architecture trained on DRIVE manual masks with multi-scale matched filter fallback. |
| **Quality Assessment Gate** | `IMPLEMENTED & VALIDATED` | Quantitative multi-metric feature engine calibrated on validation distributions. |
| **Lesion Extraction (MA, HE, EX)** | `IMPLEMENTED (Morphology)` | Multi-scale mathematical morphology biomarker extractors feeding Evidence Fusion. |
| **Neovascularization** | `EXPERIMENTAL BIOMARKER` | Exploratory vascular morphometry index; not claimed as a standalone clinical diagnostic. |
| **APTOS 2019 Cohort** | `UNLABELLED SCREENING` | Used strictly for batch throughput and distribution analysis; no fabricated ground-truth labels. |
| **Grad-CAM Explainability** | `INTERPRETABILITY AID` | Visual attention alignment map; represents model focus, not proof of clinical causality. |
| **Simulink Model** | `CAPACITY SIMULATION` | Operational queuing scenario demonstrating system capacity under explicit workload assumptions. |

---

## 7. License & Attribution
Developed for the **Smart India Hackathon (SIH)** under the theme *Explainable AI for Diabetic Retinopathy Screening in Rural India*. Pure MATLAB-native implementation.
