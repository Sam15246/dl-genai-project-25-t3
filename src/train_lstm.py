#!/usr/bin/env python3
"""
train_lstm.py

Trains a BiLSTM model for multi-label emotion classification.

Uses:
- Custom vocabulary
- Bi-directional LSTM
- BCEWithLogitsLoss
- Threshold tuning
"""

import argparse
import json
import gc
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# project imports
from config import (
    DEVICE,
    LABELS,
    EMBED_DIM,
    MAX_LEN_SEQ,
    LSTM_HIDDEN,
    LSTM_DROPOUT,
    BATCH_SIZE,
    EPOCHS,
)
from preprocessing import load_and_clean
from dataset import SimpleEmbedDataset
from utils import sweep_thresholds, macro_f1


# -------------------------
# BiLSTM Model
# -------------------------
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, dropout, num_labels):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids):
        x = self.emb(input_ids)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # final hidden state
        out = self.dropout(out)
        return self.fc(out)


# -------------------------
# Training Function
# -------------------------
def train_lstm(
    train_df,
    test_df,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    save_dir="saved_models/lstm",
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build vocabulary
    print("Building vocabulary...")
    text_col = "text_clean"
    toks = [t.lower().split() for t in train_df[text_col].tolist()]
    cnt = Counter()
    for row in toks:
        cnt.update(row)

    itos = ["<pad>", "<unk>"] + [w for w, _ in cnt.most_common(20000)]
    stoi = {w: i for i, w in enumerate(itos)}
    vocab_size = len(itos)

    X = train_df[text_col].tolist()
    y = train_df[LABELS].values

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    tr_ds = SimpleEmbedDataset(X_tr, y_tr, stoi, max_len=MAX_LEN_SEQ)
    val_ds = SimpleEmbedDataset(X_val, y_val, stoi, max_len=MAX_LEN_SEQ)

    dl_tr = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    dl_va = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = LSTMModel(
        vocab_size=vocab_size,
        emb_dim=EMBED_DIM,
        hidden_dim=LSTM_HIDDEN,
        dropout=LSTM_DROPOUT,
        num_labels=len(LABELS),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = -1.0
    best_state = None
    best_thr = np.array([0.5] * len(LABELS))

    print("Training LSTM...")
    for ep in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        pbar = tqdm(dl_tr, desc=f"Epoch {ep}/{epochs}")

        for batch in pbar:
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * input_ids.size(0)

        avg_train_loss = run_loss / len(tr_ds)

        # Validation
        model.eval()
        val_probs = []
        val_trues = []

        with torch.no_grad():
            for batch in dl_va:
                input_ids = batch["input_ids"].to(DEVICE)
                labels = batch["labels"].cpu().numpy()

                logits = model(input_ids).cpu().numpy()
                probs = 1 / (1 + np.exp(-logits))

                val_probs.append(probs)
                val_trues.append(labels)

        val_probs = np.vstack(val_probs)
        val_trues = np.vstack(val_trues)

        thr = sweep_thresholds(val_trues, val_probs)
        preds = (val_probs >= thr).astype(int)
        val_f1 = macro_f1(val_trues, preds)

        print(f"Epoch {ep}: TrainLoss={avg_train_loss:.4f}  ValF1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_thr = thr

        gc.collect()
        torch.cuda.empty_cache()

    # Save best model + metadata
    model_path = save_dir / "lstm_best.pth"
    torch.save(best_state, model_path)

    meta = {
        "vocab": itos,
        "best_thresholds": best_thr.tolist(),
        "best_val_f1": float(best_f1),
    }
    meta_path = save_dir / "lstm_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model -> {model_path}")
    print(f"Saved metadata -> {meta_path}")

    # Test inference
    test_ds = SimpleEmbedDataset(test_df[text_col].tolist(), labels=None, stoi=stoi)
    dl_test = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_probs = []
    with torch.no_grad():
        for batch in dl_test:
            input_ids = batch["input_ids"].to(DEVICE)
            logits = model(input_ids)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.vstack(all_probs)
    preds = (all_probs >= best_thr).astype(int)

    submission = pd.DataFrame({"id": test_df["id"]})
    for i, lbl in enumerate(LABELS):
        submission[lbl] = preds[:, i].astype(int)

    sub_path = save_dir / "submission_lstm.csv"
    submission.to_csv(sub_path, index=False)

    print(f"Saved submission -> {sub_path}")


def main():
    args = parse_args()
    print("Loading & preprocessing data...")
    train_df, test_df = load_and_clean()

    train_lstm(train_df, test_df)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--save_dir", type=str, default="saved_models/lstm")
    return p.parse_args()


if __name__ == "__main__":
    main()
