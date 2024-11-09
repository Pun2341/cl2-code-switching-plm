import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from extract_embeddings import extract_layer_reps
import numpy as np
from transformers import BertModel, BertTokenizer, XLMRobertaTokenizer, XLMRobertaModel

# Load dataset
df = pd.read_csv('classification_dataset.csv')

# Assuming the dataset has two columns: 'sentences' and 'labels'
sentences = df['sentences'].values
# embeddings = df['embedding'].values
labels = df['labels'].values

embeddings = extract_layer_reps(BertModel.from_pretrained("bert-base-multilingual-cased"), list(sentences))
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42)

# # Convert sentences to TF-IDF features
# vectorizer = TfidfVectorizer()
# X_train_tfidf = vectorizer.fit_transform(X_train)
# X_test_tfidf = vectorizer.transform(X_test)

# Train an SVM classifier
svm_classifier = SVC(kernel='rbf')
svm_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_classifier.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))
