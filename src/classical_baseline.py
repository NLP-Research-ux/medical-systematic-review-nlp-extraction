import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
from data_loader import load_benchmark_data

def train_and_evaluate_baseline():
    train_df, test_df = load_benchmark_data()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, stop_words="english")
    
    X_train = vectorizer.fit_transform(train_df["sentence_text"])
    y_train = train_df["pico_category"]
    
    X_test = vectorizer.transform(test_df["sentence_text"])
    y_test = test_df["pico_category"]
    
    model = LinearSVC(C=1.0, random_state=42, max_iter=2000)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    
    print("=== CLASSICAL BASELINE (TF-IDF + LINEAR SVM) RESULTS ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1-Score: {f1:.4f}")
    print("
Classification Report:
", classification_report(y_test, preds))
    return acc, f1

if __name__ == "__main__":
    train_and_evaluate_baseline()
