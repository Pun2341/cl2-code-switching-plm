"""
This script handles classification tasks for code-switching (CS) detection. It extracts embeddings using 
pre-trained models and trains an SVM classifier on sentence-level labels to evaluate the model's performance.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from transformers import BertModel, BertTokenizer, DistilBertModel, DistilBertTokenizer, BitsAndBytesConfig
import torch
from rep_extract_embeddings_cs import extract_layer_reps
import os

# Memory optimization for PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load dataset
print("Loading dataset...")
df = pd.read_csv('./replication/data/classification_dataset.csv')

# Assuming the dataset has two columns: 'sentences' and 'labels'
sentences = df['sentences'].values[:200]
labels = df['labels'].values[:200]

# Determine device and model
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("Using GPU with mBERT and 4-bit quantization...")
    model_name = "bert-base-multilingual-cased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    quantized_config = BitsAndBytesConfig(load_in_4bit=True)
    model = BertModel.from_pretrained(
        model_name,
        quantization_config=quantized_config,
        device_map="auto"
    )
else:
    print("Using CPU with DistilBERT...")
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertModel.from_pretrained(model_name).to(device)

# Extract last-layer embeddings
print("Extracting embeddings...")
embeddings = torch.empty((len(sentences), model.config.hidden_size), device="cpu")  # Store on CPU
batch_size = 64
for i in range(0, len(sentences), batch_size):
    print(f"Processing batch {i // batch_size + 1}/{len(sentences) // batch_size + 1}...")
    batch_sentences = sentences[i:i + batch_size]
    batch_sentences = list(map(str, batch_sentences))  # Ensure input is List[str]

    # Clear GPU cache
    if device == "cuda":
        torch.cuda.empty_cache()

    # Extract embeddings
    batch_embeddings = extract_layer_reps(model, True, batch_sentences)

    # Move embeddings to CPU
    batch_embeddings = batch_embeddings.cpu()
    embeddings[i:i + len(batch_sentences), :] = batch_embeddings

# Convert to NumPy for training
embeddings = embeddings.detach().numpy()

# Split the dataset into training and testing sets
print("Training classifier...")
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Train an SVM classifier
svm_classifier = SVC(kernel='rbf')
svm_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_classifier.predict(X_test)

# Print classification report
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
print(classification_report(y_test, y_pred, zero_division=0))

# Plot the F1-score for the last layer
f1_score_last_layer = report["weighted avg"]["f1-score"]
plt.bar([f"{model_name} Last Layer"], [f1_score_last_layer])

# Add labels and title
plt.xlabel('Model')
plt.ylabel('F-1 Score')
plt.title(f'F-1 Score for {model_name} (Last Layer)')

# Show the plot
plt.show()
