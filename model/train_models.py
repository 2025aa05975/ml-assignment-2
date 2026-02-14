import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import roc_auc_score, matthews_corrcoef


def load_dataset():
    # reading dataset
    data = pd.read_csv("bank.csv", sep=";")

    # converting categorical columns to numbers
    for column in data.columns:
        if data[column].dtype == "object":
            encoder = LabelEncoder()
            data[column] = encoder.fit_transform(data[column])

    X = data.drop("y", axis=1)
    y = data["y"]

    # scaling features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=0.2, random_state=42)


def calculate_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = {}

    results["Accuracy"] = accuracy_score(y_test, y_pred)
    results["AUC"] = roc_auc_score(y_test, y_prob)
    results["Precision"] = precision_score(y_test, y_pred)
    results["Recall"] = recall_score(y_test, y_pred)
    results["F1 Score"] = f1_score(y_test, y_pred)
    results["MCC"] = matthews_corrcoef(y_test, y_pred)

    return results


def train_all_models():
    X_train, X_test, y_train, y_test = load_dataset()

    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }

    final_results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = calculate_metrics(model, X_test, y_test)
        final_results[name] = metrics

    return final_results
