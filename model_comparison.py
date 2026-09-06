from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path("data/heat_safety_dataset.csv")
TARGET = "heat_safety_alert"
ID_COLUMN = "event_id"
RANDOM_STATE = 42


def evaluate_model(name, y_true, y_pred):
    print(f"\n{name}")
    print("=" * len(name))
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 3))
    print("Precision:", round(precision_score(y_true, y_pred), 3))
    print("Recall:", round(recall_score(y_true, y_pred), 3))
    print("F1:", round(f1_score(y_true, y_pred), 3))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Place the CSV there before running."
        )

    data = pd.read_csv(DATA_PATH)
    clean_data = data.copy()

    X = clean_data.drop(columns=[ID_COLUMN, TARGET])
    y = clean_data[TARGET]
    X_encoded = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logistic_model = LogisticRegression(
        C=1.0, max_iter=1000, random_state=RANDOM_STATE
    )
    logistic_model.fit(X_train_scaled, y_train)
    logistic_predictions = logistic_model.predict(X_test_scaled)

    tree_model = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=8,
        random_state=RANDOM_STATE,
    )
    tree_model.fit(X_train, y_train)
    tree_predictions = tree_model.predict(X_test)

    evaluate_model("Logistic Regression", y_test, logistic_predictions)
    evaluate_model("Decision Tree", y_test, tree_predictions)


if __name__ == "__main__":
    main()
