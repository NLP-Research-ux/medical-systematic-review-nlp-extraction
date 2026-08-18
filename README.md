# Artificial Intelligence and Natural Language Processing for Automatic Data Extraction in Medical Systematic Reviews

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-brightgreen.svg)](https://www.python.org/)
[![Framework: Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)

Official codebase and benchmark evaluation suite for the Master's dissertation: **"Artificial Intelligence and Natural Language Processing for Automatic Data Extraction in Medical Systematic Reviews: Evaluating Classical Machine Learning, Transformers, and Generative Large Language Models"**.

---

## Abstract & Overview

Systematic reviews represent the highest tier of evidence-based healthcare decision-making, yet manual data extraction of Population, Intervention, Comparison, and Outcome (**PICO**) trial characteristics remains a severe labor-intensive bottleneck. 

This repository provides an end-to-end reproducible pipeline evaluating **three distinct NLP paradigms** across a curated 5,000-record benchmark derived from the PubMed 20k Randomized Controlled Trial (RCT) corpus:
1. **Classical Baseline**: TF-IDF vectorization paired with a Linear Support Vector Machine (SVM).
2. **Representation Learning**: Dense biomedical sentence embeddings via `all-MiniLM-L6-v2`.
3. **Generative Large Language Models**: Zero-shot structured clinical prompt engineering.

---

## Benchmark Evaluation Results

All models were evaluated on the exact same 5,000-sentence standardized holdout test set:

| Model Architecture | Extraction Paradigm | Overall Accuracy | Precision (Methods) | Precision (Results) | Macro-Weighted F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **TF-IDF + Linear SVM** | Classical ML | 66.4% | 0.68 | 0.64 | 0.6512 |
| **MiniLM SentenceTransformers** | Dense Embeddings | 73.8% | 0.74 | 0.71 | 0.7245 |
| **Zero-Shot LLM API** | Generative AI | **91.0%** | **0.93** | **0.91** | **0.9028** |

> **Key Finding**: Zero-shot LLMs overcame domain-specific vocabulary barriers without task-specific supervised training, achieving an **F1-score of 0.9028**, outperforming classical baselines by over 18%.

---

## Repository Structure

```text
├── data/
│   ├── README.md               # Data dictionary and PICO labeling protocol
│   └── benchmark_dataset.csv   # 5,000-record benchmark dataset
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data ingestion, cleaning, and train/test splits
│   ├── classical_baseline.py   # TF-IDF + Linear SVM model
│   ├── transformer_embedding.py# SentenceTransformer dense embedding model
│   ├── llm_zero_shot_extractor.py # Zero-shot LLM structured extraction pipeline
│   └── evaluate_metrics.py     # Multi-class evaluation and error analysis
├── app/
│   └── app.py                  # Interactive Streamlit clinical prototype
├── requirements.txt            # Python library dependencies
├── .env.example                # Environment variables template
└── README.md                   # Project documentation & replication guide
```

---

## Getting Started & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/[your-username]/medical-systematic-review-nlp-extraction.git
cd medical-systematic-review-nlp-extraction
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys (Optional for LLM pipeline)
```bash
cp .env.example .env
# Edit .env and paste your OpenAI API key:
# OPENAI_API_KEY=your_key_here
```

---

## Running the Pipelines

### Run Classical Baseline & Transformer Models
```bash
python src/classical_baseline.py
python src/transformer_embedding.py
```

### Run Multi-Class Benchmark Evaluation
```bash
python src/evaluate_metrics.py
```

### Launch Interactive Streamlit Prototype
```bash
streamlit run app/app.py
```
Open `http://localhost:8501` in your browser to test interactive PICO extraction from clinical abstracts.

---

## Citation

If using this codebase or benchmark dataset in academic research, please cite:

```bibtex
@mastersthesis{medical_nlp_extraction_2026,
  title={Artificial Intelligence and Natural Language Processing for Automatic Data Extraction in Medical Systematic Reviews},
  author={[Author Name]},
  year={2026},
  school={[University Name]},
  type={Master's Thesis}
}
```
