"""
This script processes raw code-switching (CS) data in the .conll format into structured CSV files for downstream tasks.
The processed data is stored in the `lid_processed` folder, with each dataset converted to a CSV format containing tokens
and their corresponding labels.
"""

import re
import json
import pandas as pd


#label_to_int = {
#     'unk': 0,
#     'lang1': 1, # English
#     'lang2': 2, # Spanish
#     'other': 3,
#     'ne': 4,
#     'fw': 5,
#     'mixed': 6,
#     'ambiguous': 7
# }

#int_to_label = ['unk', 'lang1', 'lang2', 'other', 'ne', 'fw', 'mixed', 'ambiguous']



# def process_data(raw_data):
#     sentences = []
#     tokens = []
#     labels = []
#     current_sentence = []
#     for line in raw_data.strip().split("\n"):
#         # Detect a new sentence
#         if line.startswith("# sent_enum") or line.startswith('meta'):
#             if current_sentence:
#                 sentences.append(current_sentence)
#                 current_sentence = []
#         else:
#             # Extract the token and label
#             match = re.match(r"(\S+)\s+(\S+)", line.strip())
#             if match:
#                 token, label_text = match.groups()
#                 label = label_to_int.get(label_text, -1)  # -1 for unknown labels - we should not encounter any

#                 tokens.append(token)
#                 labels.append(label)
#                 current_sentence.append((token, label))
#     # Append the last sentence if it exists
#     if current_sentence:
#         sentences.append(current_sentence)

#     df = pd.DataFrame({'token': tokens, 'label': labels})

#     # In case we want to know which sentence each token belongs to
#     df['sentence_id'] = sum([[i] * len(sent) for i, sent in enumerate(sentences, 1)], [])

#     return df

# Define the label mappings
label_to_int = {
    'unk': 0, 'lang1': 1, 'lang2': 2, 'other': 3, 'ne': 4, 'fw': 5, 'mixed': 6, 'ambiguous': 7
}

# Function to process raw .conll data
def process_data(raw_data):
    data = {
        "sentences": [],
        "labels": [],
        "spans": []
    }
    current_sentence = []
    current_labels = []
    current_spans = []
    start_idx = 0

    for line in raw_data.strip().split("\n"):
        if line.startswith("# sent_enum") or line.startswith('meta'):
            if current_sentence:
                data["sentences"].append(current_sentence)
                data["labels"].append(current_labels)
                data["spans"].append(current_spans)
                current_sentence, current_labels, current_spans = [], [], []
                start_idx = 0
        else:
            match = re.match(r"(\S+)\s+(\S+)", line.strip())
            if match:
                token, label_text = match.groups()
                label = label_to_int.get(label_text, -1)
                current_sentence.append(token)
                current_labels.append(label)
                end_idx = start_idx + len(token)
                current_spans.append((start_idx, end_idx))
                start_idx = end_idx + 1

    if current_sentence:
        data["sentences"].append(current_sentence)
        data["labels"].append(current_labels)
        data["spans"].append(current_spans)

    return data

# Load and save processed data
def save_processed_data(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = f.read()
    processed_data = process_data(raw_data)
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=4)

# Process datasets
save_processed_data('./replication/data/lid_raw/lid_spaeng/train.conll', './replication/data/lid_processed/processed_calcs.json')
save_processed_data('./replication/data/lid_raw/sentimix/train.conll', './replication/data/lid_processed/processed_sentimix.json')