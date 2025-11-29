#!/usr/bin/env python3
"""
train_simple_embed.py

Trains the custom SimpleEmbed model for multi-label emotion classification.

Usage:
    python src/train_simple_embed.py --epochs 5 --batch_size 32

Dependencies:
- config.py (paths, hyperparameters)
- preprocessing.py (load_and_clean)
- dataset.py (SimpleEmbedDataset)
- utils.py (thresholds, metrics)
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

# Project imports
from config import (
    DEVICE,
    LABELS,
    EMBED_DIM,
    MAX_LEN_SEQ,
    BATCH_SIZE,
    EPOCHS,
)
from preprocessing import load_and_clean
from dataset import SimpleEmbedDataset
from utils import sweep_thresholds, macro_f1

# Optional W&B
try:
    import wandb
    WANDB_AVAILABLE = True
except:
    WANDB_AVAILABLE = False


# -------------------------
# Model definition
# -------------------------

class SimpleEmbedModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_labels):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.fc = nn.Linear(emb_dim, num_labels)

    def forward(self, input_ids):
        x = self.emb(input_ids)
        mask = (input_ids != 0).unsqueeze(-1).float()
        pooled = (x * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.fc(pooled)


# -------------------------
# Training function
# -------------------------

def train_simple_embed(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    save_dir: str = "saved_models/simple_embed"
):
    """
    Full training + inference workflow for SimpleEmbed model.
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------
    # Vocabulary creation
    # -----------------------------------

    print("Building vocabulary...")

    text_col = "text_clean"
    toks = [t.lower().split() for t in train_df[text_col].tolist()]
    cnt = Counter()
    for row in toks:
        cnt.update(row)

    # <pad> = 0, <unk> = 1
    itos = ["<pad>", "<unk>"] + [w for w, _ in cnt.most_common(20000)]
    stoi = {w: i for i, w in enumerate(itos)}

    vocab_size = len(itos)
    print(f"Vocabulary size = {vocab_size}")

    # -----------------------------------
    # Train/Val Split
    # -----------------------------------
    X = train_df[text_col].tolist()
    y = train_df[LABELS].values

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    tr_dataset = SimpleEmbedDataset(X_tr, y_tr, stoi, max_len=MAX_LEN_SEQ)
    va_dataset = SimpleEmbedDataset(X_va, y_va, stoi, max_len=MAX_LEN_SEQ)

    dl_tr = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
    dl_va = DataLoader(va_dataset, batch_size=batch_size, shuffle=False)

    # -----------------------------------
    # Model, Optimizer, Loss
    # -----------------------------------
    model = SimpleEmbedModel(vocab_size, EMBED_DIM, len(LABELS)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = -1.0
    best_state = None
    best_thresholds = np.array([0.5] * len(LABELS))

    print("Training started...")

    # W&B logging
    if WANDB_AVAILABLE:
        wandb.init(project="simple_embed", name="simple_embed_run", reinit=True)
        wandb.config.update({"epochs": epochs, "batch_size": batch_size})

    # -----------------------------------
    # Training Loop
    # -----------------------------------

    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(dl_tr, desc=f"Epoch {ep}/{epochs}")

        for batch in pbar:
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * input_ids.size(0)
            pbar.set_postfix(loss=float(running_loss / ((pbar.n + 1) * batch_size)))

        avg_train_loss = running_loss / len(tr_dataset)

        # Validation
        model.eval()
        val_probs = []
        val_trues = []

        with torch.no_grad():
            for batch in dl_va:
                input_ids = batch["input_ids"].to(DEVICE)
                labels = batch["labels"].cpu().numpy()

                logits = model(input_ids).cpu().numpy()
                probs = 1 / (1 + np.exp(-logits))  # sigmoid

                val_probs.append(probs)
                val_trues.append(labels)

        val_probs = np.vstack(val_probs)
        val_trues = np.vstack(val_trues)

        # tune thresholds
        thr = sweep_thresholds(val_trues, val_probs)
        val_preds = (val_probs >= thr).astype(int)
        val_macro_f1 = macro_f1(val_trues, val_preds)

        print(f"Epoch {ep} | Train Loss={avg_train_loss:.4f} | Val F1={val_macro_f1:.4f}")

        if WANDB_AVAILABLE:
            wandb.log({"train_loss": avg_train_loss, "val_f1": val_macro_f1})

        # save best model
        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_thresholds = thr.copy()

        gc.collect()
        torch.cuda.empty_cache()

    # -----------------------------------
    # Save Best Model + Metadata
    # -----------------------------------
    if best_state is None:
        raise RuntimeError("No best model found during training.")

    model_path = save_dir / "simple_embed_best.pth"
    torch.save(best_state, model_path)

    meta = {
        "vocab": itos,
        "stoi": stoi,
        "best_thresholds": best_thresholds.tolist(),
        "best_val_f1": float(best_f1),
    }

    meta_path = save_dir / "simple_embed_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved best model to {model_path}")
    print(f"Saved metadata to {meta_path}")

    # -----------------------------------
    # Test Set Inference + Submission
    # -----------------------------------

    print("Running test inference...")

    test_dataset = SimpleEmbedDataset(
        test_df[text_col].tolist(), labels=None, stoi=stoi, max_len=MAX_LEN_SEQ
    )
    dl_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
    preds = (all_probs >= best_thresholds).astype(int)

    submission = pd.DataFrame({"id": test_df["id"]})
    for i, lbl in enumerate(LABELS):
        submission[lbl] = preds[:, i].astype(int)

    sub_path = save_dir / "submission_simple_embed.csv"
    submission.to_csv(sub_path, index=False)

    print(f"Saved submission to {sub_path}")

    if WANDB_AVAILABLE:
        wandb.finish()


# -------------------------
# CLI / Main
# -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train custom SimpleEmbed model")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--save_dir", type=str, default="saved_models/simple_embed")
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading & preprocessing data...")
    train_df, test_df = load_and_clean()

    train_simple_embed(
        train_df=train_df,
        test_df=test_df,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
