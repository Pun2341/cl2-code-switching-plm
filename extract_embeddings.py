from transformers import BertModel, BertTokenizer, XLMRobertaTokenizer, XLMRobertaModel

def extract_layer(model_version, layer_num, sentence):
    """
    Given a model version of one of the following:
        - mBERT ("bert-base-multilingual-cased"),
        - XLM-RoBERTa base ("xlm-roberta-base"),
        - XLM-RoBERTa large ("xlm-roberta-large"),
    a layer number within range, and a string containing a sentence,
    Returns the the layer_num layer of the model's representation of the sentence.
    """
    if model_version == "bert-base-multilingual-cased":
        model = BertModel.from_pretrained(model_version)
        tokenizer = BertTokenizer.from_pretrained(model_version)
    elif model_version == "xlm-roberta-base" or model_version == "xlm-roberta-large":
        model = XLMRobertaModel.from_pretrained(model_version)
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_version)
    else:
        raise ValueError("unknown model version")
    
    inputs = tokenizer(sentence, return_tensors="pt")
    
    outputs = model(**inputs, output_hidden_states=True)
    print(len(outputs.hidden_states))
    
    return outputs.hidden_states[layer_num][0]  # (sequence_length, hidden_size)