import re
import pandas as pd
from config import TRAIN_PATH, TEST_PATH

PAT_URL = re.compile(r"http\S+|www\S+")
PAT_USER = re.compile(r"@[A-Za-z0-9_]+")
PAT_HASH = re.compile(r"#[A-Za-z0-9_]+")
PAT_NON_ALNUM = re.compile(r"[^a-z0-9\s]")

def clean_text(s):
    s = s.lower()
    s = PAT_URL.sub(" ", s)
    s = PAT_USER.sub(" ", s)
    s = PAT_HASH.sub(" ", s)
    s = PAT_NON_ALNUM.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_and_clean():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    train["text"] = train["text"].fillna("").astype(str)
    test["text"] = test["text"].fillna("").astype(str)

    train["text_clean"] = train["text"].apply(clean_text)
    test["text_clean"] = test["text"].apply(clean_text)

    return train, test
