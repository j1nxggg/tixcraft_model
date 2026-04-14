"""
Pre-train 用合成圖訓練 CNN + Transformer Decoder
Loss = 4 個位置的 CrossEntropy 之和
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

from model import CaptchaTransformer
from dataset import CaptchaDataset, NUM_CLASSES, IDX2CHAR


def decode_output(logits):
    """
    logits: (B, 4, 26)
    回傳: list of str
    """
    preds = logits.argmax(dim=2)  # (B, 4)
    texts = []
    for i in range(preds.size(0)):
        text = "".join([IDX2CHAR[p.item()] for p in preds[i]])
        texts.append(text)
    return texts


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"裝置: {device}")

    # 超參數
    batch_size = 256
    epochs = 50
    learning_rate = 3e-4
    train_dir = "synthetic/train"
    val_dir = "synthetic/val"

    # 資料集
    train_dataset = CaptchaDataset(train_dir, is_train=True)
    val_dataset = CaptchaDataset(val_dir, is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )

    print(f"Train: {len(train_dataset)} 張")
    print(f"Val: {len(val_dataset)} 張")

    # 模型
    model = CaptchaTransformer(
        num_classes=NUM_CLASSES,
        d_model=256,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型參數量: {total_params:,}\n")

    # Loss + Optimizer + Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
    )

    best_acc = 0.0

    # 訓練
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, targets, _ in train_loader:
            images = images.to(device)      # (B, 1, 32, 128)
            targets = targets.to(device)    # (B, 4)

            optimizer.zero_grad()
            logits = model(images)  # (B, 4, 26)

            # 4 個位置的 CrossEntropy 之和
            loss = sum(
                criterion(logits[:, i, :], targets[:, i])
                for i in range(4)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            # 訓練準確率
            preds = logits.argmax(dim=2)  # (B, 4)
            train_correct += (preds == targets).all(dim=1).sum().item()
            train_total += images.size(0)

        avg_loss = total_loss / len(train_loader)
        train_acc = train_correct / train_total * 100

        # 驗證
        model.eval()
        val_correct = 0
        val_total = 0
        char_correct = 0
        char_total = 0

        with torch.no_grad():
            for images, targets, texts in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                logits = model(images)

                preds = logits.argmax(dim=2)  # (B, 4)

                # 完全正確率
                val_correct += (preds == targets).all(dim=1).sum().item()
                val_total += images.size(0)

                # 字元正確率
                char_correct += (preds == targets).sum().item()
                char_total += targets.numel()

        val_acc = val_correct / val_total * 100
        char_acc = char_correct / char_total * 100
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch [{epoch:02d}/{epochs}] "
            f"Loss: {avg_loss:.4f} "
            f"LR: {current_lr:.2e} "
            f"Train: {train_acc:.1f}% "
            f"Val: {val_acc:.2f}% "
            f"Char: {char_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_pretrain.pth")
            print(f"  -> 儲存最佳模型 (Acc: {best_acc:.2f}%)")

    print(f"\n{'='*60}")
    print(f"Pre-train 完成 最佳驗證準確率: {best_acc:.2f}%")
    print(f"模型已儲存至: best_pretrain.pth")


if __name__ == "__main__":
    main()
