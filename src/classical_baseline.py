"""Train and evaluate a real TF-IDF + Linear SVM baseline."""

import hashlib
import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from data_loader import ROOT, load_benchmark_data
from evaluate_metrics import evaluate_predictions


def main():
    output_dir = ROOT / "results" / "pubmed20k_svm"
    output_dir.mkdir(parents=True, exist_ok=True)

    train, test, audit = load_benchmark_data()

    print(f"Training sentences: {len(train):,}", flush=True)
    print(f"Test sentences: {len(test):,}", flush=True)
    print("Training TF-IDF + Linear SVM ...", flush=True)

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=10000,
                stop_words="english",
            ),
        ),
        (
            "svm",
            LinearSVC(
                C=1.0,
                random_state=42,
                max_iter=10000,
                dual="auto",
            ),
        ),
    ])

    # Fit the vocabulary and model using training data only.
    pipeline.fit(train["sentence_text"], train["pico_category"])

    print("Predicting held-out test sentences ...", flush=True)
    predicted = pipeline.predict(test["sentence_text"])
    decision_scores = pipeline.decision_function(test["sentence_text"])

    vectorizer = pipeline.named_steps["tfidf"]
    model = pipeline.named_steps["svm"]
    feature_names = vectorizer.get_feature_names_out()

    saved_predictions = test.rename(
        columns={"pico_category": "true_label"}
    ).copy()
    saved_predictions["predicted_label"] = predicted

    sorted_scores = np.sort(decision_scores, axis=1)

    # This score gap is not a probability or calibrated confidence.
    saved_predictions["decision_margin"] = (
        sorted_scores[:, -1] - sorted_scores[:, -2]
    )

    # Export the strongest positive coefficients for each class.
    top_terms = []

    for label, coefficients in zip(model.classes_, model.coef_):
        top_indices = np.argsort(coefficients)[-15:][::-1]

        for rank, feature_index in enumerate(top_indices, start=1):
            top_terms.append({
                "class": label,
                "rank": rank,
                "term": feature_names[feature_index],
                "coefficient": float(coefficients[feature_index]),
            })

    terms_df = pd.DataFrame(top_terms)
    terms_df.to_csv(output_dir / "top_terms.csv", index=False)
    (output_dir / "top_terms.txt").write_text(
        terms_df.to_string(index=False),
        encoding="utf-8",
    )

    # Identify features favouring the predicted class over the true class
    # for each error. These assist our later written interpretation.
    saved_predictions["features_favouring_prediction"] = ""

    wrong_indices = np.flatnonzero(
        predicted != test["pico_category"].to_numpy()
    )
    class_index = {
        label: index
        for index, label in enumerate(model.classes_)
    }

    if len(wrong_indices):
        wrong_features = vectorizer.transform(
            test.iloc[wrong_indices]["sentence_text"]
        )

        for row_number, original_index in enumerate(wrong_indices):
            features = wrong_features.getrow(row_number)

            predicted_class = class_index[predicted[original_index]]
            true_class = class_index[
                test.iloc[original_index]["pico_category"]
            ]

            coefficient_difference = (
                model.coef_[predicted_class, features.indices]
                - model.coef_[true_class, features.indices]
            )
            contributions = features.data * coefficient_difference

            ranked = np.argsort(contributions)[::-1]
            positive = [
                index for index in ranked
                if contributions[index] > 0
            ][:5]

            saved_predictions.loc[
                original_index, "features_favouring_prediction"
            ] = "; ".join(
                f"{feature_names[features.indices[index]]} "
                f"({contributions[index]:.4f})"
                for index in positive
            )

    saved_predictions.to_csv(
        output_dir / "predictions.csv",
        index=False,
    )

    train[[
        "article_id",
        "sentence_position",
        "pico_category",
    ]].to_csv(
        output_dir / "training_records.csv",
        index=False,
    )

    metrics = evaluate_predictions(saved_predictions, output_dir)

    audit.update({
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "model": "TF-IDF + LinearSVC",
        "ngram_range": [1, 2],
        "max_features": 10000,
        "stop_words": "english",
        "svm_C": 1.0,
        "random_state": 42,
        "max_iter": 10000,
        "dual": "auto",
        "vocabulary_size": len(feature_names),
        "hyperparameter_tuning": "None; fixed baseline settings",
        "misclassified_sentences": int(len(wrong_indices)),
        "decision_margin_note": (
            "Top-two decision-score gap; not a probability."
        ),
        "evaluation_scope": (
            "Filtered, deduplicated four-class subset of PubMed 20k RCT; "
            "not the original five-class benchmark or PICO extraction."
        ),
        "source_sha256": {
            filename: hashlib.sha256(
                (ROOT / "data" / "pubmed20k" / filename).read_bytes()
            ).hexdigest()
            for filename in ["train.txt", "test.txt"]
        },
    })

    (output_dir / "run_metadata.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    joblib.dump(pipeline, output_dir / "svm_pipeline.joblib")

    print("\nTOP DISCRIMINATIVE TERMS")
    print(terms_df.to_string(index=False))

    print("\nSELECTED MISCLASSIFIED SENTENCES")
    examples = pd.read_csv(output_dir / "error_examples.csv")
    for number, row in enumerate(examples.to_dict("records"), start=1):
        print(f"\nExample {number}")
        print(f"Actual: {row['true_label']}")
        print(f"Predicted: {row['predicted_label']}")
        print(f"Sentence: {row['sentence_text']}")
        print(
            "Features favouring prediction: "
            f"{row['features_favouring_prediction']}"
        )

    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro avg']['f1-score']:.4f}")
    print(f"Results and trained model saved in: {output_dir}")


if __name__ == "__main__":
    main()