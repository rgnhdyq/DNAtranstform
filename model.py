import torch.nn as nn
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SequenceEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, layers)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos(x)
        x = self.encoder(x)
        return x.mean(dim=1)  # global average pooling

class EPITransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.enhancer_enc = SequenceEncoder(vocab_size)
        self.promoter_enc = SequenceEncoder(vocab_size)

        self.classifier = nn.Sequential(
            nn.Linear(128*4 + 1, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, enh, pro, dist):
        e = self.enhancer_enc(enh)
        p = self.promoter_enc(pro)

        pair = torch.cat([
            e,
            p,
            torch.abs(e - p),
            e * p,
            dist.unsqueeze(1)
        ], dim=1)

        return self.classifier(pair).squeeze(1)
