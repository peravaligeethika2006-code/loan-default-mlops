"""Basic data validation checks."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "loan_default_processed.csv"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    assert "Status" in df.columns, "Status target column is missing."
    assert df.isnull().sum().sum() == 0, "Processed data still contains missing values."
    assert len(df) > 0, "Processed dataset is empty."
    print("Data validation passed.")
    print("Shape:", df.shape)


if __name__ == "__main__":
    main()
