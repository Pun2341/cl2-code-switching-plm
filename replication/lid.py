"""
This script handles the Language Identification (LID) task for code-switching (CS) data.
It extracts sentence embeddings using a pre-trained model (DistilBERT or mBERT with 4-bit quantization)
and trains an SVM classifier to evaluate the model's ability to detect code-switching on token-level labels.
"""
import torch
from transformers import BertModel, BertTokenizer, DistilBertModel, DistilBertTokenizer, BitsAndBytesConfig
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from collections import Counter
import json
import os

# Output directory for results and plots
output_dir = "./"
results_file = os.path.join(output_dir, "layer_f1_scores.json")
plot_file = os.path.join(output_dir, "layer_f1_scores_plot.png")

# Load datasets
datasets = {
    "sentimix": pd.read_csv("./processed_data_LID/sentimix_data.csv"),
    "calcs": pd.read_csv("./processed_data_LID/calcs_data.csv")
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

# Layers to extract
target_layers = [3, 6, model.config.num_hidden_layers - 2]

# Store F1 scores
f1_scores_across_layers = {dataset: {} for dataset in datasets.keys()}

# Extract representations
def extract_layer_representations(model, tokenizer, sentences, layer):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[layer][:, 0, :]  # Extract CLS token representations

# Process each dataset
for dataset_name, data in datasets.items():
    print(f"  Dataset: {dataset_name}")

    # Group sentences and labels by sentence ID
    grouped_sentences = data.groupby("sentence_id")["token"].apply(list)
    grouped_labels = data.groupby("sentence_id")["label"].apply(list)
    
    # Sample random sentences and corresponding labels
    sample_indices = grouped_sentences.sample(n=len(grouped_sentences), random_state=42).index

    # Use sampled indices to retrieve sentences and labels
    sentences = grouped_sentences.loc[sample_indices].tolist()
    token_labels = grouped_labels.loc[sample_indices].tolist()

    all_sentence_labels = []
    layer_representations = {layer: [] for layer in target_layers}

    # Set batch size
    batch_size = 1024 if device == "cuda" else 256
    sentence_count = 0

    # Process data in batches
    for i in range(0, len(sentences), batch_size):
        batch_sentences = []
        for sen in sentences[i:i+batch_size]:
          try:
            batch_sentences.append(' '.join(sen))
        #batch_sentences = [' '.join(sen) for sen in sentences[i:i+batch_size]]
            batch_labels = token_labels[i:i+batch_size]
            try:
                for layer in target_layers:
                    representations = extract_layer_representations(model, tokenizer, batch_sentences, layer)
                    layer_representations[layer].extend(representations.cpu().numpy())
                
                # Convert token-level labels to sentence-level
                for labels_seq in batch_labels:
                    sentence_label = Counter(labels_seq).most_common(1)[0][0]
                    all_sentence_labels.append(sentence_label)      
                
            except Exception as e:
                print(f"Error processing batch: {e}")
          except:
            print(f"Error processing batch: {sen}")
        print(f"Processed {sentence_count} sentences...")
        sentence_count += len(batch_sentences)

        # Train and evaluate SVM for each layer
        for layer in target_layers:
            print(f"Training SVM on layer {layer}...")
            X = np.array(layer_representations[layer])
            y = np.array(all_sentence_labels)

        # Batched SVM training
        svm = SGDClassifier(loss='hinge', random_state=42, max_iter=1000)
        batch_size_svm = 512
        for j in range(0, len(X), batch_size_svm):
            print('sv')
            X_batch = X[j:j + batch_size_svm]
            y_batch = y[j:j + batch_size_svm]
            svm.partial_fit(X_batch, y_batch, classes=np.unique(y))

        predictions = svm.predict(X)
        f1 = f1_score(y, predictions, average="weighted")
        print(f"Layer {layer}, F1 score: {f1}")
        f1_scores_across_layers[dataset_name][layer] = f1

# Save results to JSON
print(f"Saving F1 scores to {results_file}...")
with open(results_file, "w") as json_file:
    json.dump(f1_scores_across_layers, json_file, indent=4)

# Plotting results
plt.figure(figsize=(10, 6))
colors = ["blue", "orange", "green"]
for idx, (dataset_name, layer_results) in enumerate(f1_scores_across_layers.items()):
    layers = list(layer_results.keys())
    scores = list(layer_results.values())
    plt.scatter(layers, scores, color=colors[idx], label=dataset_name)
    plt.plot(layers, scores, color=colors[idx], linestyle='--')

plt.xlabel("Layers")
plt.ylabel("F1 Score")
plt.title(f"F1 Score for {model_name} on Selected Layers")
plt.legend()
plt.grid(True)
plt.xticks(target_layers)
plt.savefig(plot_file)
plt.show()
