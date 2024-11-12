import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from extract_embeddings import batch_write_layer_reps
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

# For each embedding type
embedding_types = {'CLS': True, 'mean': False}
for typ in embedding_types.keys():
    print(f"Training SVM classifiers using {typ} embeddings...")
    # Extract embeddings
    print("Extracting embeddings...")
    batch_write_layer_reps(BertModel.from_pretrained(
        "bert-base-multilingual-cased"), embedding_types[typ], list(sentences)[:20], file_location+"/"+typ)
    
embeddings = None
#for typ in embedding_types.keys():
for f_idx in range(len(os.listdir(file_location+"/CLS"))):
    load = torch.load(file_location+f"/CLS/model{f_idx}.pt")
    if embeddings is None:
        embeddings = load
    else:
        embeddings = torch.cat((embeddings, load), 1)

# For each layer
print(embeddings.shape)
embeddings = embeddings.detach()
for i in range(len(embeddings)):
    # Split the dataset into training and testing sets
    print(f"Training SVM classifier for layer {i}...")
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings[i], labels[:20], test_size=0.2, random_state=42)

    # Train an SVM classifier
    svm_classifier = SVC(kernel='rbf')
    svm_classifier.fit(X_train, y_train)

    # Predict on the test set
    y_pred = svm_classifier.predict(X_test)

    # Print classification report
    print(classification_report(y_test, y_pred))

    # Save the model
    with open(f'svm_classifier_layer_{typ}_{i}.pkl', 'wb') as f:
        pickle.dump(svm_classifier, f)

# Load Model
# with open('svm_classifier.pkl', 'rb') as f:
#     svm_classifier = pickle.load(f)

# Predict on new data
# new_data = SOME NEW DATA
# new_embeddings = extract_layer_reps(BertModel.from_pretrained("bert-base-multilingual-cased"), list(new_data))
# y_pred = svm_classifier.predict(new_embeddings)
# print(classification_report(DATA LABELS, y_pred))
