# Machine Learning Model Comparison

A portfolio version of my University of Tasmania classification project comparing **Logistic Regression** and a **Decision Tree** for fictional heat-safety alerts.

## Problem
The target variable is `heat_safety_alert`, where `1` means an alert should be issued and `0` means normal planning is expected to be sufficient.

## Workflow
- Load and inspect the dataset
- Remove identifier and target columns from the feature matrix
- One-hot encode categorical features
- Use a stratified 75/25 train-test split
- Standardise features for Logistic Regression
- Train Logistic Regression and Decision Tree models
- Tune Decision Tree depth and minimum leaf size
- Compare accuracy, precision, recall and F1

## Final results

| Model | Accuracy | Recall | F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.75 | 0.58 | 0.67 |
| Decision Tree | 0.60 | 0.65 | 0.58 |

The Logistic Regression model performed better overall, while the Decision Tree produced higher recall for positive heat-safety alerts. In a safety-focused scenario, recall matters because false negatives represent missed alerts.

## Run

Place the dataset at `data/heat_safety_dataset.csv`, then run:

```bash
pip install -r requirements.txt
python model_comparison.py
```

## Skills demonstrated
`Python` · `pandas` · `scikit-learn` · `Classification` · `Feature Encoding` · `StandardScaler` · `Model Evaluation` · `Decision Tree Tuning`

## Note
This is a cleaned portfolio version. University assessment instructions and marking material are not included.
