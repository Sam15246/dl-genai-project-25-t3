import sys
import os

# This allows us to import modules from the 'src' folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gradio as gr
from inference import predict

def classify(text):
    return predict(text)

gr.Interface(
    fn=classify,
    inputs="text",
    outputs="label",
    title="Emotion Classifier (RoBERTa)",
    description="Enter a sentence and the model predicts emotion probabilities"
).launch()
