# Edge-AI-Based Defect Classification for Semiconductor Images

## Hackathon
IESA – NXP DeepTech Hackathon 2026  
Phase: Phase-1 (Baseline Model)

## Overview
This repository contains the Phase-1 implementation of an Edge-AI system for
automatic classification of semiconductor wafer/die defects.
The solution is designed to balance accuracy, latency, and model size,
targeting deployment on low-power edge devices.

## Problem Statement
Manual and centralized inspection of semiconductor images introduces latency
and scalability issues. This project proposes a lightweight deep learning-based
defect classification system that can be deployed on edge hardware for
real-time inspection.

## Dataset
- Total images: 792
- Number of classes: 8
- Classes:
  - Scratch
  - Bridge
  - LER
  - Open
  - Vias
  - Crack
  - Clean
  - Other
- Image type: Grayscale SEM images replicated to 3 channels
- Input size: 160 × 160
- Train / Val / Test split: 70 / 15 / 15

Dataset ZIP: [ZIP File](https://drive.google.com/drive/folders/1UKVNA51bFTDLGS0up5qFB-jTLoyAklP9?usp=sharing)

Dataset Description:

The current dataset consists of 792 SEM images organized into 8 classes
for multi-class defect classification.

At Phase 1, the dataset size is limited due to the difficulty of
collecting labeled SEM defect images. Images have been gathered from
multiple sources including GitHub repositories,
public datasets (e.g., Kaggle), web searches(Google), and AI-generated samples(ChatGPT and Gemini).

The dataset is planned to be expanded to at least 1200+ images in
subsequent phases to improve class balance and model generalization.

The available images are split into training, validation, and test sets
and are provided as a ZIP file linked externally in this repository.


## Model Details
- Architecture: MobileNetV3-Small
- Training approach: Transfer Learning
- Framework: PyTorch
- Loss function: Cross-Entropy with label smoothing
- Class imbalance handling: WeightedRandomSampler
- Optimizer: Adam
- Scheduler: ReduceLROnPlateau

## Results (Internal Test Set)
- Accuracy: 83 %
- Precision: 84
- Recall: 82
- Average inference latency: 72 ms per image (CPU)

Confusion matrix is available in the `Validation\` folder.

## ONNX & Edge Deployment
The trained PyTorch model is exported to ONNX format for edge deployment.
The ONNX model is imported into the NXP eIQ Toolkit for compatibility validation,
quantization, and deployment analysis targeting i.MX RT series devices.


ONNX Link: [Link](https://github.com/adityverma11/silicon_forge_PS01_defect_detection/tree/main/ONNX_Material)



# Phase: Phase-2
## (Testing the model with Hackathon Dataset)

## Hackathon test dataset
- Total images: 296
- Number of classes: 9
- Classes:
  - Scratch
  - Bridge
  - LER
  - Open
  - Vias
  - Crack
  - Clean
  - Other
  - Particle
- Input size: 160 × 160(Preprocessing)

Change:-
We mapped Particle to Others to match the no. of classes as our dataset had only 8 classes.

Hackathon Dataset ZIP - [Hackathon test dataset](https://www.dropbox.com/scl/fi/gq1hwzqtd0gcpz9mpwvfv/hackathon_test_dataset_final.zip?rlkey=woq536ib0sd9cj198mf7vxzc6&st=kd6pkz0h&dl=0)

Phase 2 Demo Video- [Video](https://www.dropbox.com/scl/fi/sabxdadfzw6mp1txb9gy3/PHASE2-DEMO.mp4?rlkey=abenh4mtvedjncnnotylhibxh&st=wxn2pso9&dl=0)

## Results(Hackathon Dataset)
- Accuracy: 39 %
- Precision: 45
- Recall: 41
- Average inference latency: 1.24 ms per image (CPU)

### [Inference Log](https://github.com/adityverma11/silicon_forge_PS01_defect_detection/blob/main/phase2_inference_log.txt)

Confusion matrix is available in the `Phase2_test\` folder.




