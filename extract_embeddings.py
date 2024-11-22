from transformers import BertModel, BertTokenizer, XLMRobertaTokenizer, XLMRobertaModel
import torch

def extract_layer_reps(model, cls_pooling, sentences):
    """
    Given: one of the following models:
        - mBERT ("bert-base-multilingual-cased"),
        - XLM-RoBERTa base ("xlm-roberta-base"),
        - XLM-RoBERTa large ("xlm-roberta-large"),
    a boolean represnting whether the pooling type is CLS (true) or mean (false),
    and list of sentences.

    Returns: a tensor of shape (num_layers, num_sentences, sequence_length, hidden_size)
        containing all of the model's layer representations for all input sentences.
    """

    if model.config._name_or_path == "bert-base-multilingual-cased":
        tokenizer = BertTokenizer.from_pretrained(model.config._name_or_path)
    elif model.config._name_or_path == "xlm-roberta-base" or model.config._name_or_path == "xlm-roberta-large":
        tokenizer = XLMRobertaTokenizer.from_pretrained(model.config._name_or_path)
    else:
        raise ValueError("unknown model")
    
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        
    outputs = model(**inputs, output_hidden_states=True)
    
    if cls_pooling: #cls
        return outputs.last_hidden_state[:, 0, :] #(num_sentences, hidden_size)
    
    else: # mean pooling
        attention_mask = inputs['attention_mask'][:, 1:].unsqueeze(-1)  # (num_sentences, sequence_length excluding first token, 1)
        return (outputs.last_hidden_state[:, 1:, :] * attention_mask).sum(dim=1) / attention_mask.sum(dim=1) #(num_sentences, hidden_size)

#example
# initialize model to "bert-base-multilingual-cased", "xlm-roberta-base", or "xlm-roberta-large"
# model = BertModel.from_pretrained("bert-base-multilingual-cased")
# model = XLMRobertaModel.from_pretrained("xlm-roberta-large")
# layers = extract_layer_reps(model, False, ["Hello how are you?", "Hola cómo estás?"])
# print(layers.shape)