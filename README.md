# Clinical Sentence-Role Classification Using TF-IDF and a Linear SVM

This project evaluates an interpretable support vector machine (SVM) for
classifying clinical abstract sentences into four roles:

- BACKGROUND
- OBJECTIVES
- METHODS
- RESULTS

The evaluated task is sentence-role classification. It does not extract
Population, Intervention, Comparator and Outcome (PICO) entities.

## Dataset

The evaluation uses the original PubMed 20k RCT train and test files from:

https://github.com/Franck-Dernoncourt/pubmed-rct

The original OBJECTIVE label is renamed OBJECTIVES. CONCLUSIONS sentences
are excluded to match the four-class scope.

The original article-level partitions are preserved. Exact duplicate
sentences are removed within each partition, and test sentences also
present in training are excluded.

The final dataset contains:

| Split | Articles | Sentences |
|---|---:|---:|
| Training | 15,000 | 151,798 |
| Test | 2,500 | 25,261 |

No article IDs overlap between training and testing.

The previous benchmark_dataset.csv is excluded from the revised evaluation.
An audit found only five distinct sentence bases per class after removing
final parenthetical additions. Its perfect classification score did not
demonstrate generalisation to varied clinical abstracts.

These revised results concern a filtered, deduplicated four-class subset,
not the original five-class PubMed benchmark.

## Method

- TF-IDF features: unigrams and bigrams
- Maximum vocabulary: 10,000 features
- Stop-word list: English
- Classifier: scikit-learn LinearSVC
- C: 1.0
- Random seed: 42
- Maximum iterations: 10,000
- Solver setting: dual="auto"

The TF-IDF vocabulary and classifier are fitted using training data only.
Model settings are fixed; no hyperparameter search was performed.

## Verified Results

The completed run produced:

- Accuracy: 82.20%
- Macro F1: 0.7483
- Weighted F1: 0.8186
- Correct predictions: 20,765
- Misclassified sentences: 4,496

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| BACKGROUND | 0.6796 | 0.6645 | 0.6720 | 3,550 |
| OBJECTIVES | 0.6553 | 0.5056 | 0.5708 | 2,328 |
| METHODS | 0.8368 | 0.8948 | 0.8649 | 9,719 |
| RESULTS | 0.8887 | 0.8829 | 0.8858 | 9,664 |

The largest individual confusion is RESULTS predicted as METHODS
(958 sentences). OBJECTIVES has the lowest recall; 769 OBJECTIVES
sentences were predicted as BACKGROUND.

The Streamlit prototype has been removed. Transformer and LLM evaluations
are excluded, and their previous fixed-output fallbacks have been removed.

## Reproduce the Evaluation on Windows

From the project root, using Python 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/download_dataset.py
python src/data_loader.py
python src/classical_baseline.py
python src/evaluate_metrics.py
```

Create the virtual environment only once. On subsequent visits, activate
the existing environment.

Save installed package versions using:

```powershell
python -m pip freeze | Out-File -Encoding utf8 results/pubmed20k_svm/environment.txt
```

The evaluated run used Python 3.12.8 and scikit-learn 1.9.1.
Full installed versions are recorded in environment.txt. Small numerical
differences may occur with different software versions.

The downloader records source URLs and SHA-256 checksums in
data/pubmed20k/source_manifest.json. It downloads from the upstream master
branch, so retain the downloaded source files and compare their hashes
when reproducing this particular run.

## Saved Outputs

Outputs are written to results/pubmed20k_svm:

- confusion_matrix.csv, .png and .pdf
- classification_report.csv and .txt
- metrics.json
- top_terms.csv and .txt
- predictions.csv
- misclassified_sentences.csv
- error_examples.csv
- training_records.csv
- run_metadata.json
- environment.txt
- svm_pipeline.joblib

The confusion matrix uses actual classes in rows and predicted classes
in columns. Top terms are ranked by positive SVM coefficients.

Error examples include features that favour the predicted class over
the reference class. These help explain model decisions but require
contextual interpretation. Decision margins are not probabilities.

evaluate_metrics.py recalculates metrics from predictions.csv.
It contains no predefined performance results.

## Limitations

This evaluation does not measure PICO extraction accuracy, reviewer
workload reduction, or suitability as an independent second reviewer.
It uses one fixed model configuration and one held-out test partition.
Selected error examples may reveal ambiguous labels or mixed sentence
content, but they do not establish the prevalence of those issues.

## Dataset Citation

Dernoncourt, F. and Lee, J.Y. (2017) ‘PubMed 200k RCT: a Dataset for
Sequential Sentence Classification in Medical Abstracts’, Proceedings
of the Eighth International Joint Conference on Natural Language
Processing, Volume 2: Short Papers, pp. 308–313.

https://aclanthology.org/I17-2052/
