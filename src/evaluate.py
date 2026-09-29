"""Evaluate the saved Gradient Boosting model on the processed test split."""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "loan_default_processed.csv"
MODEL_PATH = ROOT / "models" / "loan_default_model.pkl"


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop("Status", axis=1)
    y = df["Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = joblib.load(MODEL_PATH)
    pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred, zero_division=0))
    print("Recall:", recall_score(y_test, pred, zero_division=0))
    print("F1 Score:", f1_score(y_test, pred, zero_division=0))


if __name__ == "__main__":
    main()
