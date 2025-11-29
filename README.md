---
title: Sentiment Analysis
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: gradio
app_file: deployment/app.py
pinned: false
---
# Emotion Classification using Fine-Tuned RoBERTa

A Deep Learning & GenAI Project (IIT Madras)

This repository contains the implementation of a **multi-label emotion classification system** trained on the official Kaggle dataset for the IITM DL & GenAI course.  
The final deployed model is available on **HuggingFace Spaces**.

---

## 🚀 Live Demo  
**https://huggingface.co/spaces/Sam15246/SentimentAnalysis**

---

## 📂 Repository Structure
```
DL-GENAI-PROJECT-25-T3/
├── data/
├── src/
│   ├── config.py
│   ├── utils.py
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── inference.py
│   ├── train_transformer.py
│   ├── train_simple_embed.py
│   ├── train_lstm.py
│   ├── models/
│   │   └── transformer_model.py
│
├── saved_models/
│   ├── roberta_single_state.pth
│   └── roberta_single_meta.json
│
├── deployment/
│   └── app.py
│
├── notebooks/
│   └── full_experiments.ipynb
│
├── requirements.txt
└── README.md
```

---

## 🧠 Models Implemented
- **RoBERTa-base Transformer (fine-tuned)**
- **SimpleEmbed** (custom embedding model)
- **LSTM classifier**
- **TF-IDF Logistic Regression**
- **Random baseline**

---

## 🧪 Training Commands

### Train RoBERTa
```
python src/train_transformer.py
```

### Train SimpleEmbed
```
python src/train_simple_embed.py
```

### Train LSTM
```
python src/train_lstm.py
```

---

## 🔍 Inference Example
```
from src.inference import predict
predict("I am feeling awesome today!")
```

---

## 🌐 Deployment
The model is deployed using:
- **Gradio UI**
- **HuggingFace Spaces**
- **Git LFS** for storing large `.pth` model files

---

## 📧 Contact
**Syed Ali Mujtaba - 21f1004482**  
GitHub: https://github.com/Sam15246  

