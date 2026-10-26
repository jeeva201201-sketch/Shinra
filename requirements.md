# SHINRA System Requirements & Specifications

## 1. MATLAB Platform & Version Requirements
- **Target Platform**: MATLAB Desktop (Windows, macOS, Linux) & **MATLAB Online**
- **Recommended MATLAB Release**: MATLAB R2022b, R2023a/b, R2024a/b, or latest MATLAB Online
- **Minimum MATLAB Release**: MATLAB R2020a

## 2. Required MATLAB Toolboxes & Functional Dependencies
| MATLAB Toolbox | Component Dependencies in SHINRA | Fallback Behavior if Missing |
| :--- | :--- | :--- |
| **Image Processing Toolbox** | `detectRetinalFOV`, `enhanceRetina` (CLAHE), `normalizeIllumination`, `qualityFeatures`, mathematical morphology | Required for core preprocessing & biomarker extraction |
| **Deep Learning Toolbox** | U-Net vessel segmentation (`trainVesselSegmentation`), 5-Class DR CNN (`trainSeverityModel`), Grad-CAM (`generateGradCAM`) | Automatic fallback to multi-scale matched filtering and morphological biomarker estimation |
| **Computer Vision Toolbox** | Candidate feature matching and spatial morphology | Standard matrix operations |
| **Statistics and Machine Learning Toolbox** | Stratified splitting (`split_dataset`), Evidence Fusion GLM (`trainFusionModel`), Platt/Temperature calibration (`calibrateConfidence`), evaluation metrics (`evaluateClassifier`) | Analytical ridge regression & empirical binning |
| **Simulink** | Rural clinic screening throughput & queue capacity simulation (`simulink/SHINRA_Screening_System.slx`) | Discrete-event mathematical simulation in `run_simulink_simulation.m` |

## 3. Hardware & Compute Recommendations
- **Processor**: Intel Core i5/i7/i9, AMD Ryzen 5/7/9, Apple Silicon, or MATLAB Online Cloud Compute
- **RAM**: Minimum 8 GB (16 GB recommended for full batch screening)
- **Disk Space**: ~2.5 GB for dataset storage, model checkpoints, and generated figures
- **GPU Acceleration**: Optional (NVIDIA CUDA-compatible GPU). Automatically detected; transparently falls back to CPU.

## 4. Local Dataset Organization
The project expects datasets to be located under `SHINRAdataset/`:
```
SHINRAdataset/
├── IDRid/
│   └── B. Disease Grading/
│       ├── 1. Original Images/
│       │   ├── a. Training Set/    (413 .jpg fundus images)
│       │   └── b. Testing Set/     (103 .jpg fundus images)
│       └── 2. Groundtruths/
│           ├── a. IDRiD_Disease Grading_Training Labels.csv
│           └── b. IDRiD_Disease Grading_Testing Labels.csv
├── DRIVE/
│   ├── training/
│   │   ├── images/                 (20 .tif fundus images)
│   │   ├── 1st_manual/             (20 .gif manual vessel segmentation masks)
│   │   └── mask/                   (20 .gif retinal FOV masks)
│   └── test/
│       ├── images/                 (20 .tif fundus images)
│       └── mask/                   (20 .gif retinal FOV masks)
└── APTOS 2019/
    └── *.png                       (1,928 fundus images for unlabelled throughput evaluation)
```
