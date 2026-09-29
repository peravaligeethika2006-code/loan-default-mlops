# Loan Default Prediction

This project reorganizes the uploaded notebook into the folder structure shown in the reference screenshots.

## Project flow

Raw dataset → preprocessing → encoding → train/test split → model training → evaluation → cross-validation → feature importance → saved model.

## Models

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

The notebook's final selected model is Gradient Boosting.

## Folder structure

```text
loan-default-project/
├── artifacts/
├── configs/
│   └── project_config.json
├── data/
│   ├── processed/
│   └── raw/
│       └── loan_default_dataset.zip   <-- put your dataset here
├── logs/
├── mlruns/
├── models/
├── notebooks/
│   └── loan_default_analysis.ipynb
├── outputs/
├── reports/
├── src/
│   ├── evaluate.py
│   ├── preprocess_pipeline.py
│   ├── preprocess.py
│   ├── train_mlflow.py
│   ├── train.py
│   ├── validate_data.py
│   ├── validate_outputs.py
│   └── validate_reproducibility.py
├── tests/
│   └── test_project.py
├── .gitignore
├── README.md
├── requirements.txt
├── run_lab3_baseline.py
└── run_lab5_pipeline.py
```

## Run

1. Put the ZIP dataset in `data/raw/loan_default_dataset.zip`.
2. Install packages:

```bash
pip install -r requirements.txt
```

3. Run:

```bash
python run_lab5_pipeline.py
```

Or run each step separately:

```bash
python src/preprocess.py
python src/train.py
python src/evaluate.py
```

## Important

The original notebook does not contain MLflow tracking. `src/train_mlflow.py` is included only because the requested reference structure contains that file; it is optional.
