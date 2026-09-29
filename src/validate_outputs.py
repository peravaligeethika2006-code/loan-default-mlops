"""Check that expected project outputs exist."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    ROOT / "data" / "processed" / "loan_default_processed.csv",
    ROOT / "models" / "loan_default_model.pkl",
    ROOT / "outputs" / "feature_importance.csv",
    ROOT / "reports" / "model_metrics.csv",
]


def main():
    missing = [str(p) for p in REQUIRED if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing outputs:\n" + "\n".join(missing))
    print("Output validation passed.")


if __name__ == "__main__":
    main()
