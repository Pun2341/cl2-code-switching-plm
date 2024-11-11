# Do not run. Run lid_minicons.py instead
import json
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np

# Label mappings as given
label_to_int = {
    'unk': 0, 'lang1': 1, 'lang2': 2, 'other': 3,
    'ne': 4, 'fw': 5, 'mixed': 6, 'ambiguous': 7
}

# Model names as specified
model_names = ["bert-base-multilingual-cased", "xlm-roberta-base"]

# Step 1: Load JSON Data
def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # print(len(data['sentences']))
    return data['sentences'], data['labels']

def tokenize_and_align_labels(sentence, labels, tokenizer):
    tokenized_inputs = tokenizer(sentence, is_split_into_words=True, padding=True, truncation=True, return_tensors="pt")
    word_ids = tokenized_inputs.word_ids(batch_index=0)
    
    # Align labels to tokens
    label_ids = []
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)  # Padding tokens ignored in loss
        else:
            label_ids.append(labels[word_idx])
    
    return tokenized_inputs, label_ids
    
# Step 3: Extract Layer-wise Embeddings for each Token
def get_layer_embeddings(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states  # Layer-wise embeddings
    return hidden_states  # Shape: (layers, batch_size, seq_len, hidden_size)

# Step 4: Evaluate Layer-wise F1 Score
def evaluate_layerwise_svm(X, y, layerwise_f1_scores, model_name):
    from sklearn.model_selection import train_test_split

    # Flatten X and y to prepare for SVM training and testing
    X_flat = np.concatenate(X)  # Shape should be (num_tokens, embedding_dim)
    y_flat = np.concatenate(y)  # Shape should be (num_tokens,)
    
    # Mask out padding labels (-100)
    mask = y_flat != -100
    X_flat = X_flat[mask]  # Apply mask to each layer
    y_flat = y_flat[mask]
    
    # Split train/test
    X_train_layers, X_test_layers, y_train, y_test = train_test_split(X_flat, y_flat, test_size=0.1, random_state=42)

    # Train and evaluate SVM on each layer
    for layer_idx, (X_train, X_test) in enumerate(zip(X_train_layers, X_test_layers)):
        clf = SVC(kernel='linear', random_state=42)
        clf.fit(X_train, y_train)  # Expect 2D array (num_samples, embedding_dim)
        
        # Predict and calculate F1 score
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        layerwise_f1_scores[model_name].append(f1)
        
        print(f"Model: {model_name}, Layer: {layer_idx}, F1 Score: {f1}")

# Main function to process everything
def main(json_path, models):
    sentences, labels = load_data(json_path)

    sentences = sentences[:100]
    labels = labels[:100]
    
    # Initialize layer-wise F1 scores dictionary
    layerwise_f1_scores = {model_name: [] for model_name in models}

    # Process each model
    for model_name in models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to('cpu') 

        # Collect layer embeddings
        X, y = [], []
        for sentence, label in zip(sentences, labels):
            inputs, aligned_labels = tokenize_and_align_labels(sentence, label, tokenizer)
            hidden_states = get_layer_embeddings(model, inputs)

            for layer_embedding in hidden_states:
                # Add each layer's embeddings for current sentence to X and corresponding labels to y
                X_layer = layer_embedding[0].cpu().numpy()  # Shape(len_seq, hidd_dim) # Assume batch size of 1 for each sentence
                y_layer = np.array(aligned_labels)
                X.append(X_layer)
                y.append(y_layer)
        # print(1, len(X))
        # print(2, len((X[18])))
        # print(3, len(X[0][0]))
        print(1, len(y))
        print(2, len(y[0]))
        print(3, y[15][:10])
        print()
        
        # Train and evaluate SVMs layer-wise
    #     evaluate_layerwise_svm(X, y, layerwise_f1_scores, model_name)
    
    # # Plot the results
    # plt.figure(figsize=(12, 6))
    # for model_name, f1_scores in layerwise_f1_scores.items():
    #     plt.plot(range(len(f1_scores)), f1_scores, label=model_name)

    # plt.xlabel("Layer")
    # plt.ylabel("F1 Score")
    # plt.ylim(0.8, 1.0)
    # plt.title("F1 Scores Across Layers for Token-level LID")
    # plt.legend()
    # plt.show()

# File path to JSON data (modify as necessary)
json_path = "processed_sentimix.json"  # Assuming this file contains the prepared SentiMix JSON data

# Execute the main function with two models for analysis
main(json_path, model_names)
