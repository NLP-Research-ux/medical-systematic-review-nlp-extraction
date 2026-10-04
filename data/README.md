# Dataset Documentation

## Active Dataset

The SVM evaluation uses the original PubMed 20k RCT corpus:

https://github.com/Franck-Dernoncourt/pubmed-rct

Run this command from the project root to download the source files:

```powershell
python src/download_dataset.py
```

Files are saved in data/pubmed20k:

- train.txt
- test.txt
- source_manifest.json

The manifest records download source URLs and SHA-256 checksums.
The same hashes are recorded in the evaluation's run_metadata.json.

## Task and Labels

The task is sentence-role classification, not PICO entity extraction.

Four classes are evaluated:

- BACKGROUND
- OBJECTIVES
- METHODS
- RESULTS

The source label OBJECTIVE is renamed OBJECTIVES.
CONCLUSIONS sentences are excluded.

## Preparation and Split Integrity

The original article-level training and test partitions are preserved.
There are no overlapping article IDs.

Exact duplicate sentences are removed within each partition. Test
sentences whose exact text also occurs in training are excluded.

| Preparation step | Training | Test |
|---|---:|---:|
| CONCLUSIONS sentences excluded | 27,168 | 4,571 |
| Four-class sentences before deduplication | 152,872 | 25,564 |
| Duplicate sentences removed within split | 1,074 | 111 |
| Test sentences also present in training removed | — | 192 |
| Final sentences | 151,798 | 25,261 |
| Final articles | 15,000 | 2,500 |

## Final Class Distribution

| Class | Training | Test |
|---|---:|---:|
| BACKGROUND | 21,438 | 3,550 |
| OBJECTIVES | 13,826 | 2,328 |
| METHODS | 58,680 | 9,719 |
| RESULTS | 57,854 | 9,664 |
| Total | 151,798 | 25,261 |

These are results on a filtered, deduplicated four-class subset and
should not be presented as the original five-class benchmark.


## Reproducibility

Keep the downloaded source files unchanged after evaluation.
The downloader uses the upstream master branch; matching recorded
checksums is necessary to confirm the same files were used.

The upstream authors note that some PubMed abstracts may be protected
by copyright. Refer readers to the source repository and its notices.

## Citation

Dernoncourt, F. and Lee, J.Y. (2017) ‘PubMed 200k RCT: a Dataset for
Sequential Sentence Classification in Medical Abstracts’, Proceedings
of the Eighth International Joint Conference on Natural Language
Processing, Volume 2: Short Papers, pp. 308–313.

https://aclanthology.org/I17-2052/