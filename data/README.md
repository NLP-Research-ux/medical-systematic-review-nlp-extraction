# PubMed 20k RCT Extraction Benchmark Dataset

This dataset comprises 5,000 clinically validated sentences curated from randomized controlled trial abstracts indexed in PubMed.

### Data Schema:
- `record_id`: Unique integer identifier.
- `article_id`: PubMed PMID identifier.
- `section`: Header section as reported in original abstract.
- `pico_category`: Gold-standard multi-class target label:
  1. `BACKGROUND`: Disease context, epidemiological significance, clinical rationale.
  2. `OBJECTIVES`: Specific study aim, trial hypothesis, primary question.
  3. `METHODS`: Study design, patient sample size, randomized interventions, comparator treatments.
  4. `RESULTS`: Quantitative outcomes, statistical effect sizes, primary trial endpoints.
- `sentence_text`: Raw extracted clinical sentence passage.
