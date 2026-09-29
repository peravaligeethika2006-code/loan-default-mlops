import os
import json

import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# Load data
# ============================================================

# These variables should already be created if this file is
# adapted from your training pipeline.
#
# X_train, X_test, y_train, y_test
#
# If they are not available, load them here from your
# processed dataset.

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed/loan_default_processed.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop("Status", axis=1)
y = df["Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# MLflow setup
# ============================================================

mlflow.set_experiment(
    "Loan Default Prediction"
)

client = MlflowClient()

model_name = "Loan_Default_Production_Model"


# ============================================================
# Random Forest parameters
# ============================================================

params = {
    "n_estimators": 200,
    "max_depth": 15,
    "random_state": 42,
    "class_weight": "balanced"
}


# ============================================================
# Train and log Random Forest
# ============================================================

with mlflow.start_run(
    run_name="RandomForest_Registry_V2"
) as run:

    mlflow.log_params(params)

    mlflow.log_param(
        "model_family",
        "RandomForest"
    )

    print(
        "[INFO] Training RandomForest model..."
    )

    rf_model = RandomForestClassifier(
        **params
    )

    rf_model.fit(
        X_train,
        y_train
    )

    print(
        "[INFO] Evaluating model..."
    )

    y_pred = rf_model.predict(
        X_test
    )

    y_proba = rf_model.predict_proba(
        X_test
    )[:, 1]

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            y_pred
        ),

        "precision": precision_score(
            y_test,
            y_pred
        ),

        "recall": recall_score(
            y_test,
            y_pred
        ),

        "f1_score": f1_score(
            y_test,
            y_pred
        ),

        "roc_auc": roc_auc_score(
            y_test,
            y_proba
        )
    }

    mlflow.log_metrics(
        metrics
    )

    print(
        "[INFO] Attaching preprocessing "
        "pipeline to model artifacts..."
    )

    preprocessor_path = "models/preprocessor.pkl"

    if os.path.exists(preprocessor_path):

        mlflow.log_artifact(
            preprocessor_path,
            artifact_path="preprocessing_pipeline"
        )

    else:

        print(
            "[WARNING] models/preprocessor.pkl "
            "was not found. Skipping artifact."
        )

    print(
        "[INFO] Pushing model "
        "to MLflow Registry..."
    )

    mlflow.sklearn.log_model(
        sk_model=rf_model,
        name="random_forest_model",
        registered_model_name=model_name,
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ]
    )

    print(
        "[SUCCESS] Random Forest model "
        "logged and registered."
    )


# ============================================================
# Find newly registered model versions
# ============================================================

print(
    "\n[INFO] Searching registered model versions..."
)

all_versions = client.search_model_versions(
    f"name='{model_name}'"
)

new_models = [
    mv
    for mv in all_versions
    if mv.current_stage == "None"
]


# ============================================================
# Move new model versions to Staging
# ============================================================

for mv in new_models:

    print(
        f"[INFO] Moving model version "
        f"{mv.version} to Staging..."
    )

    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Staging",
        archive_existing_versions=False
    )


# ============================================================
# Select model version
# ============================================================

all_versions = client.search_model_versions(
    f"name='{model_name}'"
)

staging_models = [
    mv
    for mv in all_versions
    if mv.current_stage == "Staging"
]

if not staging_models:

    raise RuntimeError(
        "No model version is currently in Staging."
    )


# Select the latest staging version
champion = max(
    staging_models,
    key=lambda mv: int(mv.version)
)


# ============================================================
# Get MLflow run information
# ============================================================

run = client.get_run(
    champion.run_id
)


# ============================================================
# Generate production model report
# ============================================================

report = {
    "registry_status": "READY_FOR_DEPLOYMENT",

    "model_lineage": {
        "registered_name": model_name,

        "version": int(
            champion.version
        ),

        "current_stage":
            champion.current_stage,

        "run_id":
            champion.run_id,

        "artifact_uri":
            champion.source
    },

    "performance_metrics": {
        "recall":
            run.data.metrics.get(
                "recall"
            ),

        "f1_score":
            run.data.metrics.get(
                "f1_score"
            ),

        "roc_auc":
            run.data.metrics.get(
                "roc_auc"
            ),

        "accuracy":
            run.data.metrics.get(
                "accuracy"
            ),

        "precision":
            run.data.metrics.get(
                "precision"
            )
    },

    "hyperparameters":
        run.data.params,

    "preprocessing_dependency":
        "models/preprocessor.pkl"
}


# ============================================================
# Save production report
# ============================================================

os.makedirs(
    "artifacts",
    exist_ok=True
)

report_path = (
    "artifacts/"
    "production_model_report.json"
)

with open(
    report_path,
    "w"
) as f:

    json.dump(
        report,
        f,
        indent=4
    )


print()
print(
    "[SUCCESS] Production model report created."
)

print(
    "[INFO] Report saved to:",
    report_path
)

print(
    "[INFO] Registered model:",
    model_name
)

print(
    "[INFO] Model version:",
    champion.version
)

print(
    "[INFO] Current stage:",
    champion.current_stage
)

print(
    "\n[SUCCESS] Lab 6 Registry Pipeline "
    "completed successfully!"
)