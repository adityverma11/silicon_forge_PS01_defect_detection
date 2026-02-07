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

Dataset ZIP: https://drive.google.com/file/d/1_XCpBISF7uaA3ktySa_VkIgcwFbjSI9O/view?usp=sharing


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

ONNX Model:


## How to Train
```bash
pip install -r training/requirements.txt
python training/train.py
