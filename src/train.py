"""
Train and compare the four models used in the notebook:
Logistic Regression, Decision Tree, Random Forest, Gradient Boosting.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "loan_default_processed.csv"
MODEL_PATH = ROOT / "models" / "loan_default_model.pkl"
SCALER_PATH = ROOT / "artifacts" / "standard_scaler.pkl"
FEATURE_PATH = ROOT / "outputs" / "feature_importance.csv"
METRICS_PATH = ROOT / "reports" / "model_metrics.csv"
CM_PATH = ROOT / "outputs" / "confusion_matrices.csv"


def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_split=20, min_samples_leaf=10, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=20,
            min_samples_leaf=10, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42
        ),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run src/preprocess.py first.")

    df = pd.read_csv(DATA_PATH)
    X = df.drop("Status", axis=1)
    y = df["Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = build_models()
    predictions = {}
    train_predictions = {}

    # Logistic Regression uses scaled features; tree models use original features.
    models["Logistic Regression"].fit(X_train_scaled, y_train)
    predictions["Logistic Regression"] = models["Logistic Regression"].predict(X_test_scaled)
    train_predictions["Logistic Regression"] = models["Logistic Regression"].predict(X_train_scaled)

    for name in ["Decision Tree", "Random Forest", "Gradient Boosting"]:
        models[name].fit(X_train, y_train)
        predictions[name] = models[name].predict(X_test)
        train_predictions[name] = models[name].predict(X_train)

    rows = []
    for name, pred in predictions.items():
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
            "Train Accuracy": accuracy_score(y_train, train_predictions[name]),
            "Test Accuracy": accuracy_score(y_test, pred),
        })

    metrics = pd.DataFrame(rows)
    metrics["Gap"] = metrics["Train Accuracy"] - metrics["Test Accuracy"]

    # Preserve the notebook's final model choice.
    best_name = "Gradient Boosting"
    best_model = models[best_name]

    # 5-fold stratified cross-validation on the selected model.
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model, X, y, cv=kfold, scoring="accuracy")
    print("K-Fold Accuracy Scores:", cv_scores)
    print("Mean Accuracy:", cv_scores.mean())
    print("Standard Deviation:", cv_scores.std())

    # Feature importance.
    feature_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": best_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    # Save outputs.
    MODEL_PATH.parent.mkdir(exist_ok=True)
    FEATURE_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.parent.mkdir(exist_ok=True)
    CM_PATH.parent.mkdir(exist_ok=True)
    SCALER_PATH.parent.mkdir(exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    feature_importance.to_csv(FEATURE_PATH, index=False)
    metrics.to_csv(METRICS_PATH, index=False)

    # Store confusion matrices in a simple CSV.
    cm_rows = []
    for name, pred in predictions.items():
        cm = confusion_matrix(y_test, pred)
        cm_rows.append({
            "Model": name,
            "TN": cm[0, 0], "FP": cm[0, 1],
            "FN": cm[1, 0], "TP": cm[1, 1]
        })
    pd.DataFrame(cm_rows).to_csv(CM_PATH, index=False)

    print("\nModel comparison:")
    print(metrics.to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print("Model saved to:", MODEL_PATH)
    print("Feature importance saved to:", FEATURE_PATH)
    print("Metrics saved to:", METRICS_PATH)


if __name__ == "__main__":
    main()
