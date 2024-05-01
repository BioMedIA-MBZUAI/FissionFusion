# FissionFusion: Fast Geometric Generation and Hierarchical Souping for Medical Image Analysis

**Authors**: Santosh Sanjeev, Nuren Zhaksylyk, Ibrahim Almakky, Anees Ur Rehman Hashmi, Mohammad Areeb Qazi, Mohammad Yaqub

[![Paper](https://img.shields.io/badge/Paper-Link-blue)](https://arxiv.org/abs/2403.13341)
[![Slides](https://img.shields.io/badge/Slides-Link-green)](https://mbzuaiac-my.sharepoint.com/:p:/g/personal/santosh_sanjeev_mbzuai_ac_ae/EecfRMTQnE9Kl1GfBnkpNPEBRK3nTGtSh8_egySSlt2Eug?e=3ogVQu)

---
## Overview

Welcome to the repository for "FissionFusion: Fast Geometric Generation and Hierarchical Souping for Medical Image Analysis". This paper introduces the limitations of model soups and introduces an innovative approach towards generation and merging of models. 

## Abstract

The scarcity of well-annotated medical datasets requires leveraging transfer learning from broader datasets like ImageNet or pre-trained models like CLIP. Model soups averages multiple fine-tuned models aiming to improve performance on In-Domain (ID) tasks and enhance robustness against Out-of-Distribution (OOD) datasets. However, applying these methods to the medical imaging domain faces challenges and results in suboptimal performance. This is primarily due to differences in error surface characteristics that stem from data complexities such as heterogeneity, domain shift, class imbalance, and distributional shifts between training and testing phases. To address this issue, we propose a hierarchical merging approach that involves local and global aggregation of models at various levels based on models' hyperparameter configurations. Furthermore, to alleviate the need for training a large number of models in the hyperparameter search, we introduce a computationally efficient method using a cyclical learning rate scheduler to produce multiple models for aggregation in the weight space. Our method demonstrates significant improvements over the model souping approach across multiple datasets (around 6\% gain in HAM10000 and CheXpert datasets) while maintaining low computational costs for model generation and selection. Moreover, we achieve better results on OOD datasets than model soups.

![Fast Geometric Generation and Hierarchical Souping](image_url)

---
## Installation 🔧

1. Create a conda environment:

    ```bash
    conda create --name fissionfusion python=3.8
    conda activate fissionfusion
    ```

2. Install PyTorch and other dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Training Pipeline 

1. Grid Search Experiments
    a) To run the grid search experiments, we need to run the linear probing first as a warmup to get the linear-probed model (theta superscript lp). Please change the DATASETS paths, and the implementation section in the corresponding config file (lp.yaml) as per the dataset and model. 
        ```bash
        python train.py --config './configs/lp.yaml'
        ```
    b) To run the finetuning stage (which returns 48 models for the hyperparameter settings)
        ```bash
        python finetune.py --config './configs/full_finetuning.yaml'
        ```
2. Fast Geometric Generation
    a) For the fast geometric generation experiments, we first get the models for different learning rates fixing the seed = 1 and augmentation = Heavy. We get a total of 8 models.
        ```bash
        python finetune.py --config './configs/pre_fgg_finetuning.yaml'
        ```




---


---

## Contact

For any inquiries or questions, please create an issue on this repository or contact Santosh Sanjeev at santosh.sanjeev@mbzuai.ac.ae.

---

