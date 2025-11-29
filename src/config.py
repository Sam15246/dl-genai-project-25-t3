# config.py – central configuration file

import torch
from pathlib import Path

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_PATH = Path("data/train.csv")
TEST_PATH  = Path("data/test.csv")
SAMPLE_PATH = Path("data/sample_submission.csv")

LABELS = ["anger", "fear", "joy", "sadness", "surprise"]

MAX_LENGTH_TRANSFORMER = 128
BATCH_SIZE = 32
EPOCHS = 5
TEST_SIZE = 0.2

TRANSFORMER_NAME = "roberta-base"

# simple embed
EMBED_DIM = 64
MAX_LEN_SEQ = 64

# LSTM
LSTM_HIDDEN = 64
LSTM_DROPOUT = 0.2
