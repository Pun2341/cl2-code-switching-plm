import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from extract_embeddings import batch_write_layer_reps
import matplotlib.pyplot as plt
import numpy as np
from transformers import BertModel, BertTokenizer, XLMRobertaTokenizer, XLMRobertaModel
import pickle
import os
import torch

# Load dataset
print("Loading dataset...")
df = pd.read_csv('classification_dataset.csv')

# Assuming the dataset has two columns: 'sentences' and 'labels'
sentences = df['sentences'].values
# embeddings = df['embedding'].values
labels = df['labels'].values
file_location = "./model_layers"
models = {
        "bert-base-multilingual-cased":
        {
            "model": BertModel.from_pretrained("bert-base-multilingual-cased"),
            "tokenizer": BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        },
        "xlm-roberta-base":
        {
            "model": XLMRobertaModel.from_pretrained("xlm-roberta-base"),
            "tokenizer": XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        },
        "xlm-roberta-large":
        {
            "model": XLMRobertaModel.from_pretrained("xlm-roberta-large"),
            "tokenizer": XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")
        }
    }
# For each embedding type
embedding_types = {'CLS': True}  # , 'mean': False}
# What indicates layer size?
num_layers = 26
f1Scores = np.empty((len(models), num_layers))
for m, (model_name, model_obj) in enumerate(models.items()):
    for typ in embedding_types.keys():
        print(f"Training SVM classifiers using {model_name} {typ} embeddings...")
        # Extract embeddings
        print("Extracting embeddings...")
        if len(os.listdir(f"{file_location}/{model_name}/{typ}")) == 0:
            batch_write_layer_reps(model_obj["model"], model_obj["tokenizer"], embedding_types[typ], list(sentences)[:50], f"{file_location}/{model_name}/{typ}")
        
    embeddings = None
    # for typ in embedding_types.keys():
    for f_idx in range(len(os.listdir(f"{file_location}/{model_name}/CLS"))):
        load = torch.load(f"{file_location}/{model_name}/CLS/model{f_idx}.pt")
        if embeddings is None:
            embeddings = load
        else:
            embeddings = torch.cat((embeddings, load), 1)
    print(embeddings.shape)
    # For each layer
    embeddings = embeddings.detach()
    for i in range(len(embeddings)):
        # Split the dataset into training and testing sets
        print(f"Training SVM classifier for layer {i}...")
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings[i], labels[:50], test_size=0.2, random_state=42)

        # Train an SVM classifier
        svm_classifier = SVC(kernel='rbf')
        svm_classifier.fit(X_train, y_train)

        # Predict on the test set
        y_pred = svm_classifier.predict(X_test)

        # Print classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        print(report)
        f1Scores[m, i] = report["weighted avg"]["f1-score"]
        # Save the model
        with open(f'svm_classifier_layer_{typ}_{i}.pkl', 'wb') as f:
            pickle.dump(svm_classifier, f)

    plt.plot(range(1, len(embeddings)+1), f1Scores[m, :len(embeddings)], label=model_name)

# Add labels and title
plt.xlabel('Layers')
plt.ylabel('F-1 Score')
plt.title('Mean F-1 Scores across layers for different PLMs')

# Add a legend
plt.legend()

# Show the plot
plt.show()


# Load Model
# with open('svm_classifier.pkl', 'rb') as f:
#     svm_classifier = pickle.load(f)

# Predict on new data
# new_data = SOME NEW DATA
# new_embeddings = extract_layer_reps(BertModel.from_pretrained("bert-base-multilingual-cased"), list(new_data))
# y_pred = svm_classifier.predict(new_embeddings)
# print(classification_report(DATA LABELS, y_pred))
