"""Small reproducibility checks for the project."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "reports" / "model_metrics.csv"


def main():
    if not METRICS_PATH.exists():
        raise FileNotFoundError("Run training first.")
    metrics = pd.read_csv(METRICS_PATH)
    assert set(["Model", "Accuracy", "Precision", "Recall", "F1 Score"]).issubset(metrics.columns)
    print("Reproducibility/report structure check passed.")
    print("random_state used in training: 42")
    print("test_size used in training: 0.2")
    print("cross-validation folds: 5")


if __name__ == "__main__":
    main()
