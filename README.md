# Mirror Transformer: Sequence Reversal Task

​This repository contains a Decoder-Only Transformer implementation designed to perform string reversal (mirroring). The project focuses on character-level autoregressive sequence learning using a custom dataset and the PyTorch Ignite framework.

 Author: Leelakrishna Rajasimha Yadav Doddakula

Date: 2 February 2026

## ​Project Overview
* ​Objective: Given an input sequence (e.g., abc), the model learns to generate its mirror image (cba).
* ​Dataset: Synthetic samples of random lowercase strings.
​* Sequence Format: [SOS] + original_string + [EOS] + reversed_string.
​*Training Style: The model is trained to predict the next token at each position (GPT-style/Teacher Forcing).

## Architecture & Technical Specifications
* Model Type: Decoder-Only Transformer
* Embedding Dimension: 256
* ​Attention Heads: 8
* ​Transformer Layers: 4 Decoder Layers
​* Positional Encoding: Fixed Sinusoidal Encoding
* ​Attention Masking: Causal Masking (prevents looking at future tokens)
* ​Normalization: norm_first=True for improved training stability

## ​Optimization Strategy
* ​Optimizer: Adam (base LR = 0.001)
* ​Scheduler: OneCycleLR (peak LR = 0.005)
* ​Loss Function: CrossEntropyLoss (padding tokens ignored)
* ​Scheduler Behavior: The OneCycleLR schedule initially helped escape a loss plateau but later caused optimization instability, demonstrating the sensitivity of small Transformer models to aggressive learning rate schedules.

## ​Training Observations
​The model exhibited two distinct phases during training on the synthetic dataset.
### ​Phase 1 — Structure Learning
Rapid improvement was observed as the model learned the structural relationship between input and output segments:
* Peak accuracy observed before LR spike : ~0.60
### ​Phase 2 — LR-Induced Instability
As the learning rate reached its peak, a divergence occurred, followed by recovery behavior:
* ​Divergence: ~0.60 → ~0.03
* ​Recovery: ~0.03 → ~0.37
* This provides valuable insight into training dynamics and the recovery capabilities of autoregressive Transformer setups.

## ​Example Output
* ​Input: HELLO
* Output: xuh]]
* ​Analysis: Indicates partial learning of character dependencies and sequence structure (EOS/Padding) without full squence reversal convergence.

## ​Hardware & Environment
* ​GPU: NVIDIA RTX 4050 (Laptop)
* Frameworks: PyTorch, PyTorch Ignite, tqdm

## Trained Weights
Model weights ('.pth') were saved locally after training but are not included in the repository due to file size limits.

## ​Summary
​This project focuses on Transformer training mechanics, optimization behavior, and debugging learning dynamics rather than solely on final accuracy.
