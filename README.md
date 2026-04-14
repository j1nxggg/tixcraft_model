# Tixcraft Captcha Model

A captcha recognition project based on PyTorch, designed for 4-character lowercase English captcha images.

This repository includes:

- synthetic captcha image generation
- dataset loading for both synthetic and real images
- a CNN + Transformer model for fixed-length captcha recognition
- pretraining on synthetic data
- finetuning on real captcha images

## Project Overview

This project treats captcha recognition as a fixed-length sequence prediction task.

The target captcha format is:

- 4 characters
- lowercase English letters only (`a-z`)

The training pipeline is divided into two stages:

1. Pretrain the model on synthetic captcha images
2. Finetune the pretrained model on real captcha images

## Model Architecture

The model is implemented in `model.py` and is named `CaptchaTransformer`.

Architecture summary:

1. Input image is converted to grayscale and resized to `32 x 128`
2. A CNN backbone extracts visual features
3. The feature map is converted into a sequence
4. Positional encoding is added
5. A Transformer encoder models global context
6. Four learnable queries are passed into a Transformer decoder
7. Each query predicts one character position
8. Final output shape is `(B, 4, 26)`

This design avoids explicit character segmentation and directly predicts the full 4-character captcha.

## Repository Structure

```text
.
├── create_img.py         # Generate synthetic captcha images
├── dataset.py            # Dataset definition for synthetic and real images
├── model.py              # CNN + Transformer model
├── train.py              # Pretrain on synthetic images
├── finetune.py           # Finetune on real images
├── SpicyRice-Regular.ttf # Font used for synthetic data generation
└── README.md