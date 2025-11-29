import torch
from transformers import AutoModelForSequenceClassification

def load_transformer(model_name, num_labels):
    """
    Loads a HuggingFace transformer model for multi-label classification.

    This is a thin wrapper so inference.py can call:
        model = load_transformer(TRANSFORMER_NAME, len(LABELS))

    Args:
        model_name (str): HuggingFace model name (e.g., "roberta-base")
        num_labels (int): number of output labels

    Returns:
        model (nn.Module): transformer model with the correct classification head
    """
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
