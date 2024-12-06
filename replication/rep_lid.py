"""
This script handles the Language Identification (LID) task for code-switching (CS) data.
It extracts sentence embeddings using a pre-trained model (DistilBERT or mBERT with 4-bit quantization)
and trains an SVM classifier to evaluate the model's ability to detect code-switching on token-level labels.
"""

import torch
from transformers import BertModel, BertTokenizer, DistilBertModel, DistilBertTokenizer, BitsAndBytesConfig
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from collections import Counter

# Load datasets
datasets = {
    "sentimix": pd.read_csv("./replication/data/lid_processed/sentimix_data.csv"),
    "calcs": pd.read_csv("./replication/data/lid_processed/calcs_data.csv")
}

# Determine device and model
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("Using GPU with mBERT and 4-bit quantization...")
    model_name = "bert-base-multilingual-cased"
    quantized_config = BitsAndBytesConfig(load_in_4bit=True)
    model = BertModel.from_pretrained(
        model_name,
        quantization_config=quantized_config,
        device_map="auto"
    )
    tokenizer = BertTokenizer.from_pretrained(model_name)
else:
    print("Using CPU with DistilBERT...")
    model_name = "distilbert-base-uncased"
    model = DistilBertModel.from_pretrained(model_name).to(device)
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# Store F1 scores for each dataset
f1_scores_across_datasets = {dataset_name: None for dataset_name in datasets.keys()}

# Function for batching
def batch_process(sentences, labels, batch_size):
    for i in range(0, len(sentences), batch_size):
        yield sentences[i:i + batch_size], labels[i:i + batch_size]

# Extract representations
def extract_representations(model, tokenizer, sentences, cls_pooling=True):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(model.device)
    outputs = model(**inputs)

    if cls_pooling:
        return outputs.last_hidden_state[:, 0, :]  # CLS token (batch_size, hidden_size)
    else:
        attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (batch_size, seq_len, 1)
        return (outputs.last_hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

# Process each dataset
for dataset_name, data in datasets.items():
    print(f"  Dataset: {dataset_name}")

    # Group sentences and labels by sentence ID
    sentences = data.groupby("sentence_id")["token"].apply(list).tolist()[:100]
    token_labels = data.groupby("sentence_id")["label"].apply(list).tolist()[:100]

    all_sentence_labels = []
    all_representations = []

    # Set batch size
    batch_size = 1024 if device == "cuda" else 256  # Larger batches for GPU, smaller for CPU
    sentence_count = 0

    # Process data in batches
    for batch_sentences, batch_token_labels in batch_process(sentences, token_labels, batch_size):
        for sentence, token_label_seq in zip(batch_sentences, batch_token_labels):
            try:
                sen = ' '.join(sentence)
                representations = extract_representations(model, tokenizer, [sen])

                # Detach, move to CPU, and convert to NumPy
                representations = representations.detach().cpu().numpy()

                # Add representations
                all_representations.extend(representations)

                # Convert token-level labels to sentence-level label using majority vote
                sentence_label = Counter(token_label_seq).most_common(1)[0][0]
                all_sentence_labels.append(sentence_label)

                # Increment sentence count and print progress every 100 sentences
                sentence_count += 1
                if sentence_count % 100 == 0:
                    print(f"Processed {sentence_count} sentences.")
            except Exception as e:
                print(f"Error processing sentence: {sentence}. Exception: {e}")

    print(f"Total sentences processed for {dataset_name}: {sentence_count}")

    # Train and evaluate SVM on the representations
    X = np.array(all_representations)
    y = np.array(all_sentence_labels)  # Use sentence-level labels

    svm = SVC(kernel='rbf')
    svm.fit(X, y)
    predictions = svm.predict(X)

    # Calculate F1 score and store it
    f1 = f1_score(y, predictions, average="weighted")
    print(f"    Dataset: {dataset_name}, F1 score: {f1}")
    f1_scores_across_datasets[dataset_name] = f1

# Plotting F1 scores for the datasets
plt.figure(figsize=(12, 8))

for dataset_name, f1_score_value in f1_scores_across_datasets.items():
    plt.bar(dataset_name, f1_score_value, label=dataset_name)

plt.xlabel("Dataset")
plt.ylabel("F1 Score")
plt.title(f"F1 Score for {model_name} on Each Dataset (Last Layer Only)")
plt.legend()
plt.grid(True)
plt.show()
