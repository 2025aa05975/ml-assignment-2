import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix
)

# Folder to save models
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

def preprocess_dataset(df):
    """Encode categorical columns and scale features."""
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == "object":
            df_copy[col] = LabelEncoder().fit_transform(df_copy[col])
    
    X = df_copy.drop("y", axis=1)
    y = df_copy["y"]

    X_scaled = StandardScaler().fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def get_metrics(trained_model, X_test, y_test):
    """Compute evaluation metrics for a given model."""
    y_pred = trained_model.predict(X_test)
    auc_val = roc_auc_score(y_test, trained_model.predict_proba(X_test)[:, 1]) \
        if hasattr(trained_model, "predict_proba") else 0.0

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc_val,
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }

def train_all_models(df):
    """Train all 6 classification models, save them, and return evaluation metrics."""
    X_train, X_test, y_train, y_test = preprocess_dataset(df)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1200),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(eval_metric="logloss")  # modern XGBoost
    }

    evaluation_results = {}

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        
        # Save trained model
        model_file = os.path.join(MODEL_DIR, f"{name.replace(' ', '_').lower()}.pkl")
        with open(model_file, "wb") as f:
            pickle.dump(clf, f)
        
        # Compute metrics
        evaluation_results[name] = get_metrics(clf, X_test, y_test)

    print(f"All models trained and saved in '{MODEL_DIR}' folder.")
    return evaluation_results
