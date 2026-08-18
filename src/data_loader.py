import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_benchmark_data(filepath=None, test_size=0.2, random_state=42):
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "data", "benchmark_dataset.csv")
    
    df = pd.read_csv(filepath)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["pico_category"]
    )
    return train_df, test_df

if __name__ == "__main__":
    train, test = load_benchmark_data()
    print(f"Loaded benchmark dataset successfully: Train={len(train)}, Test={len(test)}")
