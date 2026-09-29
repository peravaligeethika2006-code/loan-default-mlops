"""
Optional MLflow training wrapper.

The original notebook does not contain MLflow code.
This file is provided because the requested project structure
contains train_mlflow.py.
"""

from pathlib import Path
import sys
import pickle

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))

try:
    from train import build_models
except ImportError:
    raise SystemExit(
        "Could not import build_models from train.py."
    )

DATA_PATH = ROOT / "data" / "processed" / "loan_default_processed.csv"
MODEL_PATH = ROOT / "models" / "loan_default_model.pkl"


def main():

    # Load processed dataset
    df = pd.read_csv(DATA_PATH)

    X = df.drop("Status", axis=1)
    y = df["Status"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Get Gradient Boosting model
    model = build_models()["Gradient Boosting"]

    # Start MLflow run
    with mlflow.start_run():

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        pred = model.predict(X_test)

        # Accuracy
        acc = accuracy_score(y_test, pred)

        # Log parameters
        mlflow.log_param("model", "Gradient Boosting")
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        # Log metric
        mlflow.log_metric("accuracy", acc)

        # Save model locally
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)

        # Log model to MLflow
        mlflow.sklearn.log_model(
            model,
            name="gradient_boosting_model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        print()
        print("MLflow experiment completed successfully.")
        print("Model: Gradient Boosting")
        print(f"Accuracy: {acc:.4f}")
        print("Model saved to:", MODEL_PATH)
        print("MLflow run completed.")


if __name__ == "__main__":
    main()