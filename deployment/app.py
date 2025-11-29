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
