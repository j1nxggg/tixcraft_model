"""
資料集模組
兩種標籤來源
    合成圖 從 labels.txt 讀取
    真實圖 從檔名提取
"""
import os
import re
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# 字元集 a-z 共 26 類
CHARS = 'abcdefghijklmnopqrstuvwxyz'
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
IDX2CHAR = {i: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS)  # 26


def extract_label(filepath):
    """
    從檔名提取標籤
    abcd.png:abcd
    abcd_v2.png:abcd
    abcd_001.png:abcd
    """
    name = os.path.splitext(os.path.basename(filepath))[0].lower()
    label = re.split(r'_', name)[0]
    return label


def is_valid_label(label):
    """檢查標籤是否為 4 個小寫字母"""
    return len(label) == 4 and label.isalpha() and label.islower()


class CaptchaDataset(Dataset):
    """
    驗證碼資料集
    """
    def __init__(self, data_dir, is_train=True):
        self.image_paths = []
        self.labels = []

        label_file = os.path.join(data_dir, "labels.txt")

        if os.path.exists(label_file):
            # 合成圖
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        filename, text = parts
                        text = text.lower()
                        if is_valid_label(text):
                            self.image_paths.append(os.path.join(data_dir, filename))
                            self.labels.append(text)
        else:
            # 真實圖
            for p in sorted(glob.glob(os.path.join(data_dir, "*.png"))):
                label = extract_label(p)
                if is_valid_label(label):
                    self.image_paths.append(p)
                    self.labels.append(label)

        # 數據增強
        if is_train:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((32, 128)),
                transforms.RandomAffine(
                    degrees=3,
                    translate=(0.03, 0.03),
                    scale=(0.95, 1.05),
                ),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                ], p=0.5),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
                ], p=0.3),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((32, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        label = self.labels[idx]

        # 標籤轉為 (4,) 的 long tensor
        target = torch.tensor([CHAR2IDX[c] for c in label], dtype=torch.long)

        return img, target, label


if __name__ == "__main__":
    # 測試
    for test_dir in ["real_img", "synthetic/train"]:
        if os.path.exists(test_dir):
            ds = CaptchaDataset(test_dir, is_train=False)
            print(f"{test_dir}: {len(ds)} 張")
            if len(ds) > 0:
                img, target, label = ds[0]
                print(f"  圖片: {img.shape}, 標籤: {label}, target: {target}")
