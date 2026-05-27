# STDN-MD：In-Vehicle Low-Light Face Image Enhancement with Physical–Semantic Constrained Diffusion and Gated Selective State-Space Scanning
Official PyTorch implementation of STDN-MD, accepted by Neural Networks.
**In-Vehicle Low-Light Face Image Enhancement with Physical–Semantic Constrained Diffusion and Gated Selective State-Space Scanning**  
*Neural Networks, 2026*

> Repository: https://github.com/sgsggahggajshsjg/STDN-MD

---

## Overview

STDN-MD is a two-stage low-light face image enhancement framework designed for in-vehicle driver monitoring scenarios.

The framework consists of:

1. **Stage 1: Physical Illumination Inference**  
   STDN and L-Mamba are jointly trained to estimate a physically guided illumination map.

2. **Stage 2: Semantic-Constrained Diffusion Restoration**  
   A HybridMambaUNet-based diffusion model restores low-light face images under semantic and illumination guidance.

---

## Repository Structure

```text
STDN-MD/
├── data/
│   ├── dataset_stage1_guided.py
│   └── dataset_stage2.py
├── models/
│   ├── model_stdn.py
│   ├── model_l_mamba.py
│   └── model_r_diffusion.py
├── losses/
│   ├── loss_stdn.py
│   ├── loss_stage1_guided.py
│   └── loss_r_diffusion_guided.py
├── scripts/
│   ├── train_stage1_guided.py
│   └── train_stage2_guided.py
├── pretrained/
│   └── README.md
├── assets/
│   └── framework.png
├── requirements.txt
├── LICENSE
└── README.md
dataset/
├── train/
│   ├── low/
│   ├── high/
│   ├── mask/
│   └── l_high_gen_blur_3/
└── test/
    ├── low/
    ├── high/
    ├── mask/
    └── l_high_gen_blur_3/
