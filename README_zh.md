
---

## README_zh.md

```markdown
# Tixcraft 驗證碼辨識模型

English version: [README.md](./README.md)

## 專案簡介

使用 PyTorch 實作的驗證碼辨識模型，目標是辨識 4 碼小寫英文字母驗證碼。

此專案包含：

- 合成驗證碼圖片生成
- 合成資料與真實資料的資料集讀取
- CNN + Transformer 驗證碼辨識模型
- 使用合成資料進行預訓練
- 使用真實資料進行微調

本專案將驗證碼辨識視為固定長度序列預測問題，不是先進行明確的字元切割。

## 功能

- 支援 4 碼小寫英文字母驗證碼
- 支援使用合成資料進行預訓練
- 支援使用真實圖片進行微調
- 使用 CNN + Transformer Encoder/Decoder 架構
- 使用固定 learnable queries 進行逐位置字元預測

## 專案結構

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