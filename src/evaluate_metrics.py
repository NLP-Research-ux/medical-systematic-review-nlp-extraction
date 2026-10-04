"""Calculate metrics from actual saved predictions."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from data_loader import LABELS, ROOT


def evaluate_predictions(predictions, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required = {"true_label", "predicted_label", "sentence_text"}

    if predictions.empty or not required.issubset(predictions.columns):
        raise ValueError("Missing or invalid predictions. Train the SVM first.")

    for column in ["true_label", "predicted_label"]:
        if not predictions[column].isin(LABELS).all():
            raise ValueError(f"Unexpected class in {column}")

    actual = predictions["true_label"]
    predicted = predictions["predicted_label"]

    # Confusion matrix: actual classes in rows, predictions in columns.
    matrix = confusion_matrix(actual, predicted, labels=LABELS)

    matrix_df = pd.DataFrame(
        matrix,
        index=LABELS,
        columns=LABELS,
    )
    matrix_df.index.name = "actual_label"
    matrix_df.to_csv(output_dir / "confusion_matrix.csv")

    report_text = classification_report(
        actual,
        predicted,
        labels=LABELS,
        digits=4,
        zero_division=0,
    )

    metrics = classification_report(
        actual,
        predicted,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    # Save per-class results and average results separately from accuracy.
    report_rows = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, dict)
    }
    pd.DataFrame(report_rows).transpose().to_csv(
        output_dir / "classification_report.csv"
    )

    (output_dir / "classification_report.txt").write_text(
        report_text,
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    # Export a figure suitable for the thesis and presentation.
    figure, axis = plt.subplots(figsize=(8, 6))

    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=LABELS,
    ).plot(
        ax=axis,
        cmap="Blues",
        colorbar=False,
        values_format="d",
        xticks_rotation=25,
    )

    axis.set_title("TF-IDF + Linear SVM: Four-Class Test Results")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=300)
    figure.savefig(output_dir / "confusion_matrix.pdf")
    plt.close(figure)

    # Export all genuine errors.
    errors = predictions.loc[actual != predicted].copy()

    if "decision_margin" in errors.columns:
        errors = errors.sort_values("decision_margin", ascending=False)

    errors.to_csv(
        output_dir / "misclassified_sentences.csv",
        index=False,
    )

    # Selecting up to three examples, favouring different error types.
    examples = errors.drop_duplicates(
        subset=["true_label", "predicted_label"]
    ).head(3)

    if len(examples) < 3:
        extra = errors.drop(examples.index).head(3 - len(examples))
        examples = pd.concat([examples, extra])

    examples.to_csv(output_dir / "error_examples.csv", index=False)

    print("\nCONFUSION MATRIX — rows: actual; columns: predicted")
    print(matrix_df.to_string())
    print("\nPER-CLASS CLASSIFICATION REPORT")
    print(report_text)
    print(f"Misclassified sentences: {len(errors)}")

    if errors.empty:
        print("No misclassifications occurred in this run.")

    return metrics


if __name__ == "__main__":
    output_dir = ROOT / "results" / "pubmed20k_svm"
    predictions_path = output_dir / "predictions.csv"

    if not predictions_path.exists():
        raise SystemExit(
            "No saved predictions yet. "
            "Run the updated classical_baseline.py first."
        )

    predictions = pd.read_csv(predictions_path)
    evaluate_predictions(predictions, output_dir)