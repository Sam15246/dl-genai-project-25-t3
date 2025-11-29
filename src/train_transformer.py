#!/usr/bin/env python3
"""
train_transformer.py

Single-run fine-tuning script for a transformer (RoBERTa) for multi-label classification.

Usage:
    python src/train_transformer.py --epochs 5 --batch_size 32 --lr 2e-5

This script expects the project layout:
- src/config.py
- src/preprocessing.py (function: load_and_clean -> returns train, test)
- src/dataset.py (class: TransformerDataset)
- src/utils.py (functions: sweep_thresholds, macro_f1)
"""

import os
import json
import argparse
import gc
import math
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

# project imports (assumes this file is in src/)
from config import (
    DEVICE,
    TRAIN_PATH,
    TEST_PATH,
    LABELS,
    MAX_LENGTH_TRANSFORMER,
    BATCH_SIZE,
    EPOCHS,
    TRANSFORMER_NAME,
)
from preprocessing import load_and_clean
from dataset import TransformerDataset
from utils import sweep_thresholds, macro_f1

# Optional wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False


# -------------------------
# Safe loader with fallbacks
# -------------------------
def safe_load_tokenizer_and_model(requested_model_name: str, num_labels: int = None, cache_dir: str = None):
    """
    Attempt to load tokenizer & model with fallbacks. Returns (tokenizer, model, used_name).
    """
    CANDIDATES = [
        requested_model_name,
        "roberta-base",
        "distilroberta-base",
        "bert-base-uncased",
        "distilbert-base-uncased"
    ]
    last_exc = None
    for nm in CANDIDATES:
        try:
            print(f"[safe_load] Trying tokenizer: {nm}")
            tok = AutoTokenizer.from_pretrained(nm, use_fast=True, trust_remote_code=False, cache_dir=cache_dir)
            print(f"[safe_load] Tokenizer OK: {nm} — loading model")
            mdl = AutoModelForSequenceClassification.from_pretrained(
                nm,
                num_labels=(num_labels if num_labels is not None else len(LABELS)),
                problem_type="multi_label_classification",
                trust_remote_code=False,
                cache_dir=cache_dir
            )
            print(f"[safe_load] Model OK: {nm}")
            return tok, mdl, nm
        except Exception as e:
            last_exc = e
            print(f"[safe_load] Failed to load {nm}: {str(e)[:200]}")
            continue
    raise RuntimeError(f"All tokenizer/model fallbacks failed. Last error: {last_exc}")


# -------------------------
# Training / Evaluation
# -------------------------
def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on dataloader and return (y_true, probs).
    y_true: np.array (N, num_labels)
    probs: np.array (N, num_labels)
    """
    model.eval()
    all_probs = []
    all_trues = []
    with torch.no_grad():
        for batch in dataloader:
            # batch: input_ids, attention_mask, (token_type_ids) optionally, labels
            labels = batch.pop("labels").cpu().numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_trues.append(labels)
    if len(all_probs) == 0:
        return np.zeros((0, len(LABELS))), np.zeros((0, len(LABELS)))
    probs = np.vstack(all_probs)
    trues = np.vstack(all_trues)
    return trues, probs


def train_one_run(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = 2e-5,
    max_length: int = MAX_LENGTH_TRANSFORMER,
    save_dir: str = "saved_models",
    use_wandb: bool = False,
    run_name: str = "roberta_single"
) -> Dict[str, Any]:
    """
    Train a transformer model on the provided dataframes. Returns a dictionary with results and saved paths.
    """
    os.makedirs(save_dir, exist_ok=True)
    num_labels = len(LABELS)

    # tokenizer & model
    tokenizer, model, used_name = safe_load_tokenizer_and_model(TRANSFORMER_NAME, num_labels=num_labels)
    model.to(DEVICE)

    # datasets & loaders
    tr_ds = TransformerDataset(train_df["text_clean"].tolist(), train_df[LABELS].values, tokenizer, max_length)
    va_ds = TransformerDataset(val_df["text_clean"].tolist(), val_df[LABELS].values, tokenizer, max_length)
    dl_tr = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    dl_va = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    # optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = max(1, len(dl_tr) * epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    # optional wandb
    if use_wandb and WANDB_AVAILABLE:
        try:
            wandb.init(project=os.environ.get("WANDB_PROJECT", "dl_project"), name=run_name, reinit=True)
            wandb.config.update({"epochs": epochs, "lr": lr, "batch_size": batch_size, "max_length": max_length, "model": used_name})
        except Exception as e:
            print("[wandb] init failed:", e)

    best_f1 = -1.0
    best_state = None
    best_thr = np.array([0.5] * num_labels)

    loss_fn_name = "BCEWithLogitsLoss"

    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(dl_tr, desc=f"[Train] Epoch {ep}/{epochs}", leave=False)
        for batch in pbar:
            labels = batch.pop("labels").to(DEVICE)
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            out = model(**batch, labels=labels)
            loss = out.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * labels.size(0)
            pbar.set_postfix(loss=float(running_loss / ((pbar.n + 1) * batch_size)))

        avg_loss = running_loss / max(1, len(dl_tr.dataset))

        # validation
        val_trues, val_probs = evaluate_model(model, dl_va, DEVICE)
        thr = sweep_thresholds(val_trues, val_probs)
        val_preds = (val_probs >= thr).astype(int)
        val_f1 = float(np.mean([f1_score_col(val_trues[:, i], val_preds[:, i]) for i in range(num_labels)]))  # helper below

        # better metric: macro f1 across labels (same as in notebook)
        val_macro_f1 = macro_f1(val_trues, val_preds)

        print(f"Epoch {ep} | avg_train_loss={avg_loss:.4f} | val_macro_f1={val_macro_f1:.4f}")

        if use_wandb and WANDB_AVAILABLE:
            wandb.log({"epoch": ep, "train_loss": avg_loss, "val_macro_f1": val_macro_f1})

        # update best
        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_thr = thr.copy()

        # housekeeping
        gc.collect()
        torch.cuda.empty_cache()

    # load best state and evaluate test set
    if best_state is None:
        raise RuntimeError("No best model found during training (best_state is None).")

    # Save model state dict to disk
    model_save_path = Path(save_dir) / "roberta_single_state.pth"
    torch.save(best_state, model_save_path)
    print(f"Saved best state dict to {model_save_path}")

    # Save tokenizer identifier + thresholds
    meta = {
        "model_name": used_name,
        "labels": LABELS,
        "thresholds": best_thr.tolist(),
        "best_val_macro_f1": float(best_f1)
    }
    meta_path = Path(save_dir) / "roberta_single_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta to {meta_path}")

    # Test inference using saved state (use tokenizer & a fresh model instance to ensure clean state)
    test_texts = test_df["text_clean"].tolist()
    # reload model from pretrained + load_state
    final_model = AutoModelForSequenceClassification.from_pretrained(used_name, num_labels=num_labels, problem_type="multi_label_classification")
    final_model.load_state_dict(torch.load(model_save_path, map_location="cpu"))
    final_model.to(DEVICE)
    final_model.eval()

    test_probs = []
    batch_size_inf = 256
    for i in range(0, len(test_texts), batch_size_inf):
        batch_texts = test_texts[i:i + batch_size_inf]
        enc = tokenizer(batch_texts, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            logits = final_model(**enc).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            test_probs.append(probs)
    test_probs = np.vstack(test_probs)
    test_preds = (test_probs >= np.array(best_thr)).astype(int)

    submission = pd.DataFrame({"id": test_df["id"]})
    for i, lbl in enumerate(LABELS):
        submission[lbl] = test_preds[:, i].astype(int)
    submission_path = Path(save_dir) / "submission_transformer_roberta_single.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path}")

    # wandb finalize
    if use_wandb and WANDB_AVAILABLE:
        try:
            wandb.finish()
        except Exception:
            pass

    return {
        "model_state_path": str(model_save_path),
        "meta_path": str(meta_path),
        "submission_path": str(submission_path),
        "best_val_macro_f1": float(best_f1),
        "thresholds": best_thr.tolist(),
    }


# -------------------------
# helper function
# -------------------------
from sklearn.metrics import f1_score
def f1_score_col(y_true_col, y_pred_col):
    return f1_score(y_true_col, y_pred_col, zero_division=0)


# -------------------------
# CLI / main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train single-run transformer for multi-label emotion classification")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=MAX_LENGTH_TRANSFORMER)
    p.add_argument("--save_dir", type=str, default="saved_models")
    p.add_argument("--no_wandb", action="store_true", help="Disable wandb logging even if wandb is installed")
    return p.parse_args()


def main():
    args = parse_args()
    use_wandb = (not args.no_wandb) and WANDB_AVAILABLE

    print("Loading data...")
    train_df, test_df = load_and_clean()

    # split train -> train/val
    from sklearn.model_selection import train_test_split
    X = train_df["text_clean"].tolist()
    y = train_df[LABELS].values
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    train_df_run = pd.DataFrame({"text_clean": X_tr})
    val_df_run = pd.DataFrame({"text_clean": X_va})
    for i, lbl in enumerate(LABELS):
        train_df_run[lbl] = y_tr[:, i]
        val_df_run[lbl] = y_va[:, i]

    # ensure test has cleaned text
    if "text_clean" not in test_df.columns:
        test_df["text_clean"] = test_df["text"].astype(str).apply(lambda s: s.lower())

    result = train_one_run(
        train_df=train_df_run,
        val_df=val_df_run,
        test_df=test_df,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        save_dir=args.save_dir,
        use_wandb=use_wandb,
        run_name=f"roberta_single_run_e{args.epochs}_bs{args.batch_size}"
    )

    print("Training finished. Summary:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
