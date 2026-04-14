# Tixcraft Captcha Model

Chinese version: [README_zh.md](./README_zh.md)

## Overview

This project is a PyTorch-based captcha recognition model for 4-character lowercase English captchas.

It includes:

- synthetic captcha generation
- dataset loading for both synthetic and real images
- a CNN + Transformer based recognition model
- pretraining on synthetic data
- finetuning on real captcha images

The project treats captcha recognition as a fixed-length sequence prediction task rather than explicit character segmentation.

## Features

- Supports 4-character lowercase English captchas
- Uses synthetic data generation for pretraining
- Supports finetuning on real captcha images
- Uses a CNN + Transformer Encoder/Decoder architecture
- Uses fixed learnable queries for position-wise character prediction

## Project Structure

```text
.
├── create_img.py
├── dataset.py
├── model.py
├── train.py
├── finetune.py
├── SpicyRice-Regular.ttf
├── README.md
└── README_zh.md