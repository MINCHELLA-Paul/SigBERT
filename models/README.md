# Pretrained Language Models for Embedding Extraction

This directory is intended to store the **pretrained NLP models** used to extract **word-level embeddings** from clinical text.

## Purpose

The models stored here are used upstream in the pipeline to convert raw medical reports into numerical representations (embeddings) that capture the semantic content of the text. These embeddings serve as the foundation for downstream sentence-level and temporal modeling steps.

## Expected Model Format

Each model should be compatible with the Hugging Face 🤗 `transformers` library, and must include:
- A tokenizer (`tokenizer_config.json`, `vocab.json`, `merges.txt`, etc.)
- A model file (`pytorch_model.bin` or equivalent)
- A `config.json` describing the architecture and task

You can place the model either:
- In a local folder (e.g., `models/OncoBERT_v1.0/`)
- Or refer to a Hugging Face model ID if downloaded dynamically

## Recommended Architecture

The pipeline is designed to work best with **RoBERTa-based models**, such as:
- [`CamemBERT`](https://huggingface.co/camembert-base): for general-purpose French NLP
- [`OncoBERT`](https://aacrjournals.org/cancerres/article/84/6_Supplement/3475/739847/Abstract-3475-Prediction-of-nausea-or-vomiting-and): a domain-specific model fine-tuned on French oncology reports *(recommended)*

## Usage in the Project

Once the model is placed in this directory, it is automatically loaded by the notebook: [`compute_sent_embd.ipynb`](../notebooks/compute_sent_embd.ipynb)


This notebook:
- Loads the model and tokenizer from `models/`
- Tokenizes the raw reports
- Computes **sentence embeddings** using either the CLS token or the Arora/SIF method
- Exports processed data for temporal modeling

*Example Structure*

models/  
├── OncoBERT_v1.0/  
│ ├── config.json  
│ ├── pytorch_model.bin  
│ ├── tokenizer_config.json  
│ ├── vocab.json  
│ └── merges.txt  
└── README.md  


---

Make sure the models stored here are licensed appropriately and do not contain any sensitive patient-related information.