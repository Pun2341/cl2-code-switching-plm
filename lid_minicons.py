import torch
from minicons import cwe
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Load datasets
datasets = {
    "sentimix": pd.read_csv("sentimix_data.csv"),
    "calcs": pd.read_csv("calcs_data.csv")
}

# Define model names and initialize minicons models
model_names = [
    "bert-base-multilingual-cased",
    "xlm-roberta-base",
    "xlm-roberta-large"
]

# Model-dataset pairs
model_dataset_pairs = [(model, dataset) for model in model_names for dataset in datasets.keys()]

# Label mappings
label_to_int = {
    'unk': 0, 'lang1': 1, 'lang2': 2, 'other': 3, 'ne': 4, 'fw': 5, 'mixed': 6, 'ambiguous': 7
}

# Store F1 scores for each model@dataset pair across layers
f1_scores_across_layers = {f"{model}@{dataset}": [] for model, dataset in model_dataset_pairs}

# Process each model and dataset
for model_name in model_names:
    print(f"Processing model: {model_name}")
    
    model = cwe.CWE(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    num_layers = model.layers + 1
    
    for dataset_name, data in datasets.items():
        print(f"  Dataset: {dataset_name}")
        
        # Group sentences and labels by sentence ID
        sentences = data.groupby("sentence_id")["token"].apply(list).tolist()
        labels = data.groupby("sentence_id")["label"].apply(list).tolist()

        sentences = sentences[:200] # Slice the list on this line for smaller datasets
        labels = labels[:200] # Slice the list on this line for smaller datasets
        
        all_labels = []
        all_layer_representations = {layer: [] for layer in range(num_layers)}

        # Extract representations for all tokens at each layer
        for sentence, label_seq in zip(sentences, labels):
            sen = ' '.join(sentence)
            instances = [(sen, token) for token in sentence]
            representations = model.extract_representation(instances, layer=list(range(num_layers)))
            # DEBUG: print(len(representations))
            for layer in range(num_layers):
                 all_layer_representations[layer].extend(representations[layer])
            all_labels.extend(label_seq)

        # Train and evaluate SVM on each layer
        for layer in range(num_layers):
            print(f"    Layer: {layer}")
            X = np.array(all_layer_representations[layer])
            y = np.array(all_labels)
            # DEBUG: print('X', X.shape)
            # DEBUG: print('y',y.shape)

            svm = SVC()
            svm.fit(X, y)
            predictions = svm.predict(X)
            
            # Calculate F1 score and store it
            f1 = f1_score(y, predictions, average="weighted")
            print('model', model_name, dataset_name, 'f1 score for layer', layer, ':', f1)
            f1_scores_across_layers[f"{model_name}@{dataset_name}"].append(f1)

# Plotting F1 scores across layers
plt.figure(figsize=(12, 8))
layer_range = range(num_layers)

# Define colors and line styles for distinction
colors = {
    "bert-base-multilingual-cased": "blue",
    "xlm-roberta-base": "green",
    "xlm-roberta-large": "red"
}
line_styles = {
    "sentimix": "solid",
    "calcs": "dashed"
}

for names, f1_scores in f1_scores_across_layers.items():
    print(names)
    model_name, dataset_name = names.split('@')
    plt.plot(layer_range, f1_scores, label=names, 
             color=colors[model_name], linestyle=line_styles[dataset_name])

plt.xlabel("Model Layer")
plt.ylabel("F1 Score")
plt.title("F1 Score across Layers for Language Identification Task")
plt.legend()
plt.grid(True)
plt.show()


# from minicons import cwe
# from sklearn.svm import SVC
# from sklearn.metrics import f1_score, classification_report
# from sklearn.model_selection import train_test_split
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from collections import Counter
# from sklearn.utils import resample

# # Define label mappings
# int_to_label = ['unk', 'lang1', 'lang2', 'other', 'ne', 'fw', 'mixed', 'ambiguous']

# # Load processed data
# def load_data(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         return json.load(f)

# # Initialize models
# mbert_model = cwe.CWE('bert-base-multilingual-cased', device='cpu')
# xlmr_base_model = cwe.CWE('xlm-roberta-base', device='cpu')
# #xlmr_large_model = cwe.CWE('xlm-roberta-large', device='cpu')

# # Define the datasets for each model
# datasets = {
#     'mBERT_SentiMix': (mbert_model, load_data("processed_sentimix.json")),
#     'mBERT_CALCS': (mbert_model, load_data("processed_calcs.json")),
#     'XLM-R-base_SentiMix': (xlmr_base_model, load_data("processed_sentimix.json")),
#     'XLM-R-base_CALCS': (xlmr_base_model, load_data("processed_calcs.json")),
#     #'XLM-R-large_SentiMix': (xlmr_large_model, load_data("processed_sentimix.json")),
#     #'XLM-R-large_CALCS': (xlmr_large_model, load_data("processed_calcs.json"))
# }

# # Dictionary to store F1 scores across layers for each model
# layer_f1_scores = {key: [] for key in datasets.keys()}

# def extract_and_evaluate(model, data, max_layers, batch_size=32):
#     sentences = data["sentences"][:300]
#     spans = data["spans"][:300]
#     labels = data["labels"][:300]

#     instances = []
#     full_labels = []

#     for sentence_tokens, sentence_spans, sentence_labels in zip(sentences, spans, labels):
#         full_sentence = " ".join(sentence_tokens)
#         for span, label in zip(sentence_spans, sentence_labels):
#             instances.append((full_sentence, span))
#             full_labels.append(label)

#     print(len(instances), len(full_labels))
    
#     X_train, X_test, y_train, y_test = train_test_split(
#         instances, full_labels, test_size=0.3, random_state=42
#     )

#     full_train_labels = np.array(y_train)
#     full_test_labels = np.array(y_test)
    
#     layer_scores = []
#     for layer in range(max_layers):
#         svm = SVC(kernel='linear', C=0.1)
#         train_reps = []
#         test_reps = []

#         for i in range(0, len(X_train), batch_size):
#             batch_train = X_train[i:i + batch_size]
#             batch_reps_train = model.extract_representation(batch_train, layer=layer)
#             train_reps.append(batch_reps_train.cpu().numpy())

#         for i in range(0, len(X_test), batch_size):
#             batch_test = X_test[i:i + batch_size]
#             batch_reps_test = model.extract_representation(batch_test, layer=layer)
#             test_reps.append(batch_reps_test.cpu().numpy())

#         X_train_reps = np.concatenate(train_reps, axis=0)
#         X_test_reps = np.concatenate(test_reps, axis=0)

#         # Train on the train set and evaluate on the test set
#         svm.fit(X_train_reps, full_train_labels[:len(X_train_reps)])
#         y_pred = svm.predict(X_test_reps)
        
#         f1 = f1_score(full_test_labels[:len(X_test_reps)], y_pred, average='macro')
#         layer_scores.append(f1)
#         print(f"Layer {layer}: F1 Score on Test Set = {f1:.4f}")

#     return layer_scores


# # Main function to extract and plot layer representations
# def main():
#     # Process each dataset and model
#     for model_name, (model, data) in datasets.items():
#         print(f"Processing {model_name}...")
#         max_layers = 12 if 'large' not in model_name else 16
#         layer_scores = extract_and_evaluate(model, data, max_layers)
#         layer_f1_scores[model_name] = layer_scores

#     # Plotting
#     plt.figure(figsize=(10, 6))
#     colors = ['blue', 'cyan', 'green', 'yellow']#, 'red', 'orange']
#     for idx, (model_name, scores) in enumerate(layer_f1_scores.items()):
#         plt.plot(scores, label=model_name, color=colors[idx])

#     plt.xlabel("Layer Index")
#     plt.ylabel("Mean F1 Score")
#     plt.ylim(0.8, 1.0)
#     plt.title("LID Model Mean F1 Scores Across Layers")
#     plt.legend(loc="best")
#     plt.show()

# if __name__ == "__main__":
#     main()
