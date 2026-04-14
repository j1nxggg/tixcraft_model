"""
CNN + Transformer Decoder (4 learnable queries)
固定輸出 4 個字元

架構
  Input (1, 32, 128)
    -> CNN Backbone -> (256, 1, 16) -> flatten -> (16, 256) 序列
    -> Positional Encoding
    -> Transformer Encoder 全局特徵交互
    -> 4 Learnable Queries X Transformer Decoder (cross-attention)
    -> 4 個獨立分類頭 -> 各輸出 26 類
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """固定正弦位置編碼"""
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class CaptchaTransformer(nn.Module):
    def __init__(
        self,
        num_classes=26,
        d_model=256,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model


        # CNN Backbone
        # 輸入: (B, 1, 32, 128)
        self.cnn = nn.Sequential(
            # Block 1: (B, 1, 32, 128) -> (B, 64, 16, 64)
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: -> (B, 128, 8, 32)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: -> (B, 256, 4, 16)
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: -> (B, 256, 2, 16)
            nn.Conv2d(256, d_model, 3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),

            # 壓縮高度: -> (B, 256, 1, 16)
            nn.AdaptiveAvgPool2d((1, 16)),
        )

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=16)

        #  Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Learnable Queries (4 個 對應 4 個字元)
        self.query_embed = nn.Embedding(4, d_model)

        # 5. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # 分類頭 (4 個共享同一個 FC)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        """
        x: (B, 1, 32, 128)
        returns: (B, 4, 26) — 4 個位置各自的 26 類 logits
        """
        B = x.size(0)

        # CNN 特徵提取
        features = self.cnn(x)              # (B, 256, 1, 16)
        features = features.squeeze(2)      # (B, 256, 16)
        features = features.permute(0, 2, 1)  # (B, 16, 256)

        # 加入位置編碼
        features = self.pos_encoder(features)  # (B, 16, 256)

        # Transformer Encoder
        memory = self.transformer_encoder(features)  # (B, 16, 256)

        # 4 個 learnable queries
        query_pos = torch.arange(4, device=x.device)
        queries = self.query_embed(query_pos)  # (4, 256)
        queries = queries.unsqueeze(0).expand(B, -1, -1)  # (B, 4, 256)

        # Transformer Decoder: queries attend to encoder memory
        decoded = self.transformer_decoder(queries, memory)  # (B, 4, 256)

        # 分類 每個 query 獨立分類
        logits = self.classifier(decoded)  # (B, 4, 26)

        return logits


if __name__ == "__main__":
    model = CaptchaTransformer()
    dummy = torch.randn(2, 1, 32, 128)
    out = model(dummy)
    print(f"模型建立成功")
    print(f"輸入: {dummy.shape}")
    print(f"輸出: {out.shape} -> (Batch, 4 字元, 26 類)")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數量: {total_params:,}")
    print(f"可訓練參數: {trainable_params:,}")
