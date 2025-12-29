import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from dataset import KMerTokenizer, EPIDataset
from model import EPITransformer
from sklearn.metrics import roc_auc_score

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in loader:
        enh = batch['enhancer'].to(device)
        pro = batch['promoter'].to(device)
        dist = batch['distance'].to(device)
        label = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(enh, pro, dist)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def eval_epoch(model, loader, device):
    model.eval()
    ys, preds = [], []

    with torch.no_grad():
        for batch in loader:
            enh = batch['enhancer'].to(device)
            pro = batch['promoter'].to(device)
            dist = batch['distance'].to(device)

            y = batch['label'].numpy()
            p = torch.sigmoid(model(enh, pro, dist)).cpu().numpy()

            ys.extend(y)
            preds.extend(p)

    return roc_auc_score(ys, preds)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = KMerTokenizer(k=3)
    dataset = EPIDataset("train.csv", tokenizer, max_len=200)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = EPITransformer(vocab_size=len(tokenizer.vocab)+1).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(10):
        loss = train_epoch(model, loader, optimizer, criterion, device)
        auc = eval_epoch(model, loader, device)
        print(f"Epoch {epoch+1} | Loss {loss:.4f} | AUC {auc:.4f}")
