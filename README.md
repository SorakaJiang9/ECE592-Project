
# Dual-Domain Underwater Enhancement Network (DD-UIE)

Welcome! This repository contains the implementation of our **ECE 592 course project** on **underwater image enhancement** based on **dual-domain learning**.

The project adapts and extends the idea of spatial–spectral dual-domain learning for underwater image enhancement, and re-implements it with our customized architecture **Dual-Domain Underwater Enhancement Network (DD-UIE)**.

---

## :art: Abstract

We propose a Dual-Domain Underwater Enhancement Network (**DD-UIE**) for learning-based underwater image enhancement. The key idea is to jointly model **spatial degradation patterns** and **spectral/frequency characteristics** in a unified dual-domain framework with efficient components.

Concretely, we first introduce a **Pyramidal Selective Scan Module (PSSM)** for spatial modeling and a **Frequency-Domain Self-Attention (FDSA)** module for spectral/frequency modeling. Both modules are designed to scale linearly with image size. We combine PSSM and FDSA in parallel inside a **Dual-Domain Fusion Block (DDFB)** and stack them to form a higher-level **Multi-Stage Spatial–Spectral Block (MSSB)**. Benefiting from the global receptive field of PSSM and FDSA, MSSB can effectively capture degradation levels across different spatial regions and spectral bands.

On top of these building blocks, DD-UIE is organized into:

- A **Low-Level Encoder (LLE)** that extracts shallow features from the input underwater image.
- A **Deep Feature Aggregation Module (DFAM)** that densely connects multiple MSSBs and fuses their outputs via an **Adaptive Feature Fusion Block (AFFB)**.
- A **Decoder Reconstruction Head (DRH)** that progressively upsamples and fuses shallow and deep features to reconstruct high-quality images.

To further emphasize frequency-sensitive degradations, we introduce an **Adaptive Spectral Loss (ASL)** that constrains the reconstruction in the frequency domain and encourages the network to recover high-frequency details and fine structures without adding additional parameters during inference.

Extensive experiments on standard underwater benchmark datasets show that DD-UIE achieves competitive or improved quantitative performance and visual quality compared to previous state-of-the-art methods, while maintaining efficient memory and computation.


### Main Contributions

1. **Dual-domain fusion backbone.** We design a **Dual-Domain Fusion Block (DDFB)** that couples **PSSM** (for spatial degradation modeling) and **FDSA** (for spectral/frequency modeling) in parallel, enabling the network to capture spatial- and spectral-wise degradation patterns jointly with linear complexity.

2. **Multi-stage spatial–spectral representation.** By stacking DDFBs into a **Multi-Stage Spatial–Spectral Block (MSSB)** and organizing them inside the **Deep Feature Aggregation Module (DFAM)**, DD-UIE can aggregate multi-level dual-domain features and adaptively focus on severely degraded regions and bands.

3. **Adaptive Spectral Loss (ASL).** We introduce an ASL term that operates in the frequency domain to narrow the discrepancy between reconstructed and ground-truth spectra, encouraging recovery of high-frequency details without extra test-time computation.

4. **Practical ECE 592 implementation.** We provide a complete PyTorch implementation with **`train.py`** and **`test.py`**, along with scripts and configuration for reproducing our course project experiments on LSUI-style data splits.


---

## 💾 Repository Structure (ECE 592 Project)

A simplified view of the key files used in the course project:

```text
Adaptive-Dual-domain-Learning-for-Underwater-Image-Enhancement/
├─ net/
│  ├─ model.py        # DD-UIE (Dual-Domain Underwater Enhancement Network)
│  └─ blocks.py       # DDFB, PSSM, FDSA, MSSB, LLE/DFAM/DRH components
├─ utils/
│  ├─ LAB.py          # LAB color-space loss
│  ├─ LCH.py          # LCH color-space loss
│  ├─ FDL.py          # Frequency-domain loss implementation (used in ASL)
│  └─ utils.py        # PSNR, logging helpers, etc.
├─ data/
│  ├─ Train/
│  │  ├─ input/       # Training input underwater images
│  │  └─ GT/          # Training ground-truth images
│  ├─ Test/
│  │  ├─ input/       # Validation / test input images (optional)
│  │  └─ GT/          # Validation / test ground-truth images (optional)
│  └─ Test_400/
│     ├─ input/       # 400 test input images (LSUI-style split)
│     └─ output/      # Model outputs written by test.py
├─ train.py           # Training script for DD-UIE
├─ test.py            # Testing / inference script for DD-UIE
├─ saved_models/      # Folder for checkpoints (DD_UIE_*.pth)
└─ README.md          # This file
