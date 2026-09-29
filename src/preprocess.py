"""
Preprocess the loan default dataset.

This script follows the preprocessing steps from the notebook:
- load raw CSV/ZIP
- remove ID, year, rate_of_interest, Interest_rate_spread, Upfront_charges
- median imputation for numeric columns
- mode imputation for categorical columns
- remove duplicate rows
- one-hot encode categorical columns
- save processed data
"""

from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "Loan_Default.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "loan_default_processed.csv"


def load_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            csv_files = [n for n in z.namelist() if n.lower().endswith(".csv") and not n.endswith("/")]
            if not csv_files:
                raise FileNotFoundError("No CSV file found inside the ZIP file.")
            with z.open(csv_files[0]) as f:
                return pd.read_csv(f)
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    required = "Status"
    if required not in df.columns:
        raise ValueError("Target column 'Status' was not found.")

    drop_cols = [
        "ID", "year",
        "rate_of_interest", "Interest_rate_spread", "Upfront_charges"
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        mode = df[col].mode()
        if not mode.empty:
            df[col] = df[col].fillna(mode.iloc[0])

    df = df.drop_duplicates()

    df = pd.get_dummies(df, drop_first=True)
    return df


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_PATH}\n"
            "Copy your loan dataset ZIP into data/raw/ and name it "
            "'loan_default_dataset.zip'."
        )

    df = load_data(RAW_PATH)
    processed = preprocess(df)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(PROCESSED_PATH, index=False)

    print("Original shape:", df.shape)
    print("Processed shape:", processed.shape)
    print("Missing values remaining:", int(processed.isnull().sum().sum()))
    print("Processed data saved to:", PROCESSED_PATH)


if __name__ == "__main__":
    main()
