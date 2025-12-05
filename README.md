
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


## ✈️ Get Started

### 1. Clone the repository:

   ```bash
   git clone https://github.com/SorakaJiang9/ECE592-Project.git
   cd ECE592-Project
   ```

To set up the environment for this project, follow the steps below:

### 2. Create and Activate Conda Environment

```bash
conda create -n your_env_name python=3.10
conda activate your_env_name
```

### 3. Install PyTorch with CUDA Support

```bash
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 3. Install CUDA Compiler (nvcc)

```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
```

### 4. Install Additional Dependencies

```bash
conda install packaging
pip install timm
pip install scikit-image
pip install opencv-python
pip install causal-conv1d==1.1.1
pip install mamba-ssm==1.1.1
```

If you cannot install causal-conv1d and mamba-ssm, you can download the whl file we provide and install it directly using the local whl file. The download link is [causal-conv1d](https://drive.google.com/file/d/1Os8ibqmPF6ldN1EBBruY-90R8XERzG1K/view?usp=sharing) and [mamba-ssm](https://drive.google.com/file/d/1qj7VwDPMpCo0bpJLm4KLGwoCZhpAQenx/view?usp=sharing). Then run,

```bash
pip install causal_conv1d-1.2.0.post2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64
pip install mamba_ssm-1.1.1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64
```


### 🚀 Training

If you need to train DD-UIE from scratch, you need to download the LSUI dataset, and then randomly select 3879 picture pairs as the training set to replace the data folder, and the remaining 400 as the test set to replace the test folder.

Then, run the `train.py`, and the trained model weight file will be automatically saved in saved_ Models folder. 



## 📊 Testing
For your convenience, we provide some example datasets in `./data` folder. 

After downloading, extract the pretrained model into the `./saved_models` folder, and then run `test.py`. The code will use the pretrained model to automatically process all the images in the `./data/Test_400/input` folder and output the results to the `./data/Test_400/output` folder. 

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
