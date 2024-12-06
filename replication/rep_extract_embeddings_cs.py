"""
This module provides utilities to extract sentence representations (embeddings) from pre-trained models 
like mBERT or DistilBERT. Supports CLS pooling or mean pooling and ensures compatibility with tokenized inputs.
"""

from transformers import BertModel, BertTokenizer, DistilBertModel, DistilBertTokenizer
import torch

def extract_layer_reps(model, cls_pooling, sentences):
    """
    Given: a model (e.g., mBERT or DistilBERT), a boolean representing whether the pooling type is CLS (True) or mean (False),
    and a list of sentences.

    Returns: a tensor of shape (num_sentences, hidden_size) containing the last layer's representations.
    """
    if model.config._name_or_path in ["bert-base-multilingual-cased", "distilbert-base-uncased"]:
        if "bert-base-multilingual-cased" in model.config._name_or_path:
            tokenizer = BertTokenizer.from_pretrained(model.config._name_or_path)
        else:
            tokenizer = DistilBertTokenizer.from_pretrained(model.config._name_or_path)
    else:
        raise ValueError("Unknown model")

    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(model.device)
    outputs = model(**inputs)

    if cls_pooling:  # CLS pooling
        return outputs.last_hidden_state[:, 0, :]  # (num_sentences, hidden_size)
    else:  # Mean pooling
        attention_mask = inputs['attention_mask'].unsqueeze(-1)  # (num_sentences, sequence_length, 1)
        return (outputs.last_hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
