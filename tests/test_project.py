import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocess import preprocess
import pandas as pd


def test_preprocess_removes_unwanted_columns():
    df = pd.DataFrame({
        "ID": [1, 2],
        "year": [2019, 2020],
        "rate_of_interest": [None, None],
        "Interest_rate_spread": [None, None],
        "Upfront_charges": [None, None],
        "Status": [0, 1],
        "Income": [1000, None],
        "Type": ["A", "B"],
    })
    out = preprocess(df)
    for col in ["ID", "year", "rate_of_interest", "Interest_rate_spread", "Upfront_charges"]:
        assert col not in out.columns
    assert out.isnull().sum().sum() == 0
    assert "Status" in out.columns
