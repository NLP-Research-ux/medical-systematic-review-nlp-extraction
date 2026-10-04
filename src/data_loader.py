"""Load original PubMed RCT splits for four-class sentence classification."""

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LABELS = ["BACKGROUND", "OBJECTIVES", "METHODS", "RESULTS"]


def read_rct(path):
    path = Path(path)
    records = []
    article_id = None
    position = 0
    excluded_conclusions = 0

    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\r\n")

            if not line.strip():
                continue

            if line.startswith("###"):
                article_id = line[3:].strip()
                position = 0
                continue

            if article_id is None or "\t" not in line:
                raise ValueError(
                    f"Invalid corpus format in {path.name}, "
                    f"line {line_number}: {line[:100]!r}"
                )

            label, sentence = line.split("\t", 1)
            label = label.strip()
            sentence = sentence.strip()
            position += 1

            # The original corpus uses the singular label OBJECTIVE.
            if label == "OBJECTIVE":
                label = "OBJECTIVES"

            # Exclude this class to match the client's four-class scope.
            if label == "CONCLUSIONS":
                excluded_conclusions += 1
                continue

            if label not in LABELS:
                raise ValueError(
                    f"Unexpected label {label!r} in {path.name}, "
                    f"line {line_number}"
                )

            if not sentence:
                raise ValueError(
                    f"Empty sentence in {path.name}, line {line_number}"
                )

            records.append({
                "article_id": article_id,
                "sentence_position": position,
                "pico_category": label,
                "sentence_text": sentence,
            })

    frame = pd.DataFrame(records)

    if frame.empty:
        raise ValueError(f"No usable sentences found in {path.name}")

    if set(frame["pico_category"]) != set(LABELS):
        raise ValueError(
            f"All four requested classes must occur in {path.name}. "
            f"Found: {sorted(frame['pico_category'].unique())}"
        )

    return frame, excluded_conclusions


def load_benchmark_data(data_dir=None):
    directory = (
        Path(data_dir)
        if data_dir is not None
        else ROOT / "data" / "pubmed20k"
    )

    for filename in ["train.txt", "test.txt"]:
        if not (directory / filename).is_file():
            raise FileNotFoundError(
                f"Missing {directory / filename}. "
                "Run: python src/download_dataset.py"
            )

    train, train_excluded = read_rct(directory / "train.txt")
    test, test_excluded = read_rct(directory / "test.txt")

    # Preserve the original article-level train/test partitions.
    overlap = set(train["article_id"]) & set(test["article_id"])
    if overlap:
        raise ValueError(
            f"Data leakage: {len(overlap)} article IDs occur in both splits."
        )

    original_train = len(train)
    original_test = len(test)

    # Remove exact duplicate text within each split.
    train = train.drop_duplicates(subset=["sentence_text"]).copy()
    test = test.drop_duplicates(subset=["sentence_text"]).copy()
    test_duplicates = original_test - len(test)

    # Remove test sentences whose exact text also occurs in training.
    seen_in_training = test["sentence_text"].isin(
        set(train["sentence_text"])
    )
    text_overlap_removed = int(seen_in_training.sum())
    test = test.loc[~seen_in_training].copy()

    for split_name, frame in [("train", train), ("test", test)]:
        if set(frame["pico_category"]) != set(LABELS):
            raise ValueError(
                f"A class disappeared from {split_name} "
                "after duplicate removal."
            )

    audit = {
        "task": "Four-class sentence-role classification",
        "split": "Original PubMed 20k RCT train/test partitions",
        "label_mapping": "OBJECTIVE -> OBJECTIVES",
        "conclusions_excluded_train": train_excluded,
        "conclusions_excluded_test": test_excluded,
        "train_rows_before_deduplication": original_train,
        "test_rows_before_deduplication": original_test,
        "train_duplicate_rows_removed": original_train - len(train),
        "test_duplicate_rows_removed": test_duplicates,
        "test_sentences_also_in_train_removed": text_overlap_removed,
        "article_overlap": len(overlap),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_articles": int(train["article_id"].nunique()),
        "test_articles": int(test["article_id"].nunique()),
        "train_class_counts": train["pico_category"].value_counts().to_dict(),
        "test_class_counts": test["pico_category"].value_counts().to_dict(),
    }

    return (
        train.reset_index(drop=True),
        test.reset_index(drop=True),
        audit,
    )


if __name__ == "__main__":
    _, _, audit = load_benchmark_data()
    print(json.dumps(audit, indent=2))