"""
# Replication Folder

This folder contains all the necessary files and data to replicate and extend experiments for the computational linguistics project on code-switching detection and language identification (LID) tasks. The experiments are based on pre-trained language models (e.g., mBERT, DistilBERT) and focus on evaluating their performance on token-level and sentence-level classification tasks.

## Folder Structure

replication/ ├── data/ │ ├── classification_dataset.csv │ ├── lid_processed/ │ │ ├── sentimix_data.csv │ │ ├── calcs_data.csv │ ├── lid_raw/ │ │ ├── (raw .conll files for LID data) │ └── process_lid_data.py ├── rep_lid.py ├── rep_extract_embeddings_cs.py ├── rep_classifier_cs.py


### Files and Their Purpose

- **`data/`**
  - Contains the datasets required for the experiments.
  - `classification_dataset.csv`: Data for sentence-level classification tasks.
  - `lid_processed/`: Processed CSV files for LID experiments.
  - `lid_raw/`: Raw `.conll` files for LID data.
  - `process_lid_data.py`: Script to convert `.conll` files to structured CSVs.

- **`rep_lid.py`**
  - Runs the Language Identification (LID) task by extracting embeddings from a pre-trained model
    and training an SVM on token-level data to classify language usage.

- **`rep_extract_embeddings_cs.py`**
  - Provides a function to extract representations (CLS or mean-pooled embeddings) from pre-trained models.

- **`rep_classifier_cs.py`**
  - Handles the classification task to detect code-switching at the sentence level.
  - Trains and evaluates an SVM classifier using embeddings extracted from pre-trained models.

## How to Run the Files

1. **Preprocessing Data:**
   - Run `process_lid_data.py` to generate processed CSVs for LID experiments.
   - The processed data will be saved in `lid_processed/`.

   ```bash
   python ./replication/data/process_lid_data.py

2. **Language Identification Task:**
   - Use rep_lid.py to extract embeddings and train an SVM for token-level language classification.

    ```bash
    python ./replication/rep_lid.py

3. **Language Identification Task:**
   - Run rep_classifier_cs.py to classify sentences for code-switching using pre-trained embeddings.

   ```bash
   python ./replication/rep_classifier_cs.py

4. **Embedding Extraction:**
    - You can call the functions in rep_extract_embeddings_cs.py to extract embeddings for custom experiments.

## Dependencies

Install the required Python libraries using:

```bash
pip install transformers minicons scikit-learn pandas matplotlib bitsandbytes


