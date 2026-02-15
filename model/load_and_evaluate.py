import pickle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix
)


def load_model(model_name):
    with open(f"model/{model_name}.pkl", "rb") as f:
        return pickle.load(f)


def evaluate_model(model, X, y):

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1 Score": f1_score(y, y_pred),
        "MCC": matthews_corrcoef(y, y_pred)
    }

    cm = confusion_matrix(y, y_pred)

    return metrics, cm
