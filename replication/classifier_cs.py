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
from replication.extract_embeddings_cs import extract_layer_reps
import os
import json

# Memory optimization for PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load dataset
print("Loading dataset...")
df = pd.read_csv('classification_dataset.csv')

# Assuming the dataset has two columns: 'sentences' and 'labels'
sentences = df['sentences'].values
labels = df['labels'].values

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
batch_size = 512  # Reduced batch size to minimize memory usage
embeddings = torch.empty((len(sentences), model.config.hidden_size), device="cpu")  # Store embeddings on CPU

# Process embeddings in batches
with torch.no_grad():  # Disable gradients to save memory
    for i in range(0, len(sentences), batch_size):
        print(f"Processing batch {i // batch_size + 1}/{(len(sentences) + batch_size - 1) // batch_size}...")
        batch_sentences = sentences[i:i + batch_size]
        batch_sentences = list(map(str, batch_sentences))  # Ensure input is List[str]

        # Extract embeddings and clear GPU cache
        try:
            batch_embeddings = extract_layer_reps(model, True, batch_sentences)
            batch_embeddings = batch_embeddings.cpu()  # Move to CPU
            embeddings[i:i + len(batch_sentences), :] = batch_embeddings
            del batch_embeddings  # Delete to free memory
            torch.cuda.empty_cache()  # Clear GPU cache
        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}: {e}")
            torch.cuda.empty_cache()

# Convert embeddings to NumPy for training
embeddings = embeddings.numpy()

# Split the dataset into training and testing sets
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Train an SVM classifier
print("Training SVM classifier...")
svm_classifier = SVC(kernel='rbf')
svm_classifier.fit(X_train, y_train)

# Predict in batches
print("Predicting in batches...")
test_batch_size = 5000  # Batch size for prediction
y_pred = []

y_pred = svm_classifier.predict(X_test)

# Print classification report
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
print(classification_report(y_test, y_pred, zero_division=0))

# Save the results
results = {
    "model": model_name,
    "f1_score": report["weighted avg"]["f1-score"],
    "accuracy": report["accuracy"],
    "report": report
}

output_file = "./replication/results/svm_classification_results.json"
os.makedirs("./replication/results", exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)
print(f"Results saved to {output_file}")

# Plot the F1-score for the last layer
f1_score_last_layer = report["weighted avg"]["f1-score"]
plt.bar([f"{model_name} Last Layer"], [f1_score_last_layer])

# Add labels and title
plt.xlabel('Model')
plt.ylabel('F-1 Score')
plt.title(f'F-1 Score for {model_name} (Last Layer)')

# Show the plot
plt.show()
