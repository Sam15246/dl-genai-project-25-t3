import torch
import os
from transformers import AutoTokenizer
from .models.transformer_model import load_transformer
from .config import TRANSFORMER_NAME, LABELS, DEVICE

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_NAME)
model = load_transformer(TRANSFORMER_NAME, len(LABELS))

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved_models", "roberta_single_state.pth"))
# Load trained weights
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()

def predict(text):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
