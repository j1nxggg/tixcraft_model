"""
Fine-tune 用真實圖微調 CNN + Transformer Decoder
從 best_pretrain.pth 訓練
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
import random

from model import CaptchaTransformer
from dataset import CaptchaDataset, NUM_CLASSES, IDX2CHAR


def decode_output(logits):
    preds = logits.argmax(dim=2)
    texts = []
    for i in range(preds.size(0)):
        text = "".join([IDX2CHAR[p.item()] for p in preds[i]])
        texts.append(text)
    return texts

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"裝置: {device}")

    # 設定
    batch_size = 64
    epochs = 60
    learning_rate = 5e-5
    patience = 15
    model_path = "best_pretrain.pth"
    real_dir = "real_img"

    # 資料集
    full_dataset = CaptchaDataset(real_dir, is_train=True)
    val_dataset_raw = CaptchaDataset(real_dir, is_train=False)

    print(f"真實圖: {len(full_dataset)} 張")

    # 拆分 80/20
    indices = list(range(len(full_dataset)))
    random.seed(42)
    random.shuffle(indices)
    split = int(0.8 * len(indices))

    train_loader = DataLoader(
        Subset(full_dataset, indices[:split]),
        batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(val_dataset_raw, indices[split:]),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
    )

    print(f"Train: {split} 張, Val: {len(indices) - split} 張")

    # 載入模型
    model = CaptchaTransformer(
        num_classes=NUM_CLASSES,
        d_model=256,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"已載入: {model_path}\n")

    # Loss + Optimizer + Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_acc = 0.0
    no_improve = 0

    # 訓練
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for images, targets, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(images)

            loss = sum(
                criterion(logits[:, i, :], targets[:, i])
                for i in range(4)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # 驗證
        model.eval()
        correct = 0
        total = 0
        char_correct = 0
        char_total = 0

        with torch.no_grad():
            for images, targets, texts in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                logits = model(images)
                preds = logits.argmax(dim=2)

                correct += (preds == targets).all(dim=1).sum().item()
                total += images.size(0)
                char_correct += (preds == targets).sum().item()
                char_total += targets.numel()

        acc = correct / total * 100
        char_acc = char_correct / char_total * 100

        print(
            f"Epoch [{epoch:02d}/{epochs}] "
            f"Loss: {total_loss / len(train_loader):.4f} "
            f"LR: {current_lr:.2e} "
            f"Val: {acc:.2f}% "
            f"Char: {char_acc:.2f}%"
        )

        if acc > best_acc:
            best_acc = acc
            no_improve = 0
            torch.save(model.state_dict(), "best_finetune.pth")
            print(f"  -> 儲存最佳模型 (Acc: {best_acc:.2f}%)")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"\n連續 {patience} epoch 沒改善 提前停止")
            break

    print(f"\n{'='*60}")
    print(f"Fine-tune 完成 最佳: {best_acc:.2f}%")
    print(f"模型: best_finetune.pth")


if __name__ == "__main__":
    main()
