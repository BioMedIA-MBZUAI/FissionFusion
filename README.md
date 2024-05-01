# FissionFusion: Fast Geometric Generation and Hierarchical Souping for Medical Image Analysis

**Authors**: Santosh Sanjeev, Nuren Zhaksylyk, Ibrahim Almakky, Anees Ur Rehman Hashmi, Mohammad Areeb Qazi, Mohammad Yaqub

---


---
## Overview

Welcome to the repository for "FissionFusion: Fast Geometric Generation and Hierarchical Souping for Medical Image Analysis". This paper introduces an innovative approach to enhance the robustness of volumetric medical image segmentation models against adversarial attacks. We propose a novel frequency domain adversarial training method and demonstrate its effectiveness compared to traditional input or voxel domain attacks.

## Abstract

The scarcity of well-annotated medical datasets requires leveraging transfer learning from broader datasets like ImageNet or pre-trained models like CLIP. Model soups averages multiple fine-tuned models aiming to improve performance on In-Domain (ID) tasks and enhance robustness against Out-of-Distribution (OOD) datasets. However, applying these methods to the medical imaging domain faces challenges and results in suboptimal performance. This is primarily due to differences in error surface characteristics that stem from data complexities such as heterogeneity, domain shift, class imbalance, and distributional shifts between training and testing phases. To address this issue, we propose a hierarchical merging approach that involves local and global aggregation of models at various levels based on models' hyperparameter configurations. Furthermore, to alleviate the need for training a large number of models in the hyperparameter search, we introduce a computationally efficient method using a cyclical learning rate scheduler to produce multiple models for aggregation in the weight space. Our method demonstrates significant improvements over the model souping approach across multiple datasets (around 6\% gain in HAM10000 and CheXpert datasets) while maintaining low computational costs for model generation and selection. Moreover, we achieve better results on OOD datasets than model soups.

![Fast Geometric Generation and Hierarchical Souping](image_url)

---
## Installation 🔧

1. Create a conda environment:

    ```bash
    conda create --name vafa python=3.8
    conda activate vafa
    ```

2. Install PyTorch and other dependencies:

    ```bash
    pip install -r requirements.txt
    ```

---


[![Paper](https://img.shields.io/badge/Paper-Link-blue)](https://arxiv.org/abs/2403.13341)
[![Slides](https://img.shields.io/badge/Slides-Link-green)](link_to_slides)

---

## Contact

For any inquiries or questions, please create an issue on this repository or contact Santosh Sanjeev at santosh.sanjeev@mbzuai.ac.ae.

---

