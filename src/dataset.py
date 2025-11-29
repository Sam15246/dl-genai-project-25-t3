import torch
from torch.utils.data import Dataset

class SimpleEmbedDataset(Dataset):
    def __init__(self, texts, labels=None, stoi=None, max_len=64):
        self.texts = texts
        self.labels = labels
        self.stoi = stoi
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        toks = self.texts[idx].lower().split()
        ids = [self.stoi.get(t, 1) for t in toks][:self.max_len]
        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))

        item = {"input_ids": torch.tensor(ids, dtype=torch.long)}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


class TransformerDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item
