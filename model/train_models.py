import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_and_save_models():

    # Load dataset
    data = pd.read_csv("bank.csv", sep=";")

    # Encode categorical variables
    for col in data.columns:
        if data[col].dtype == "object":
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])

    X = data.drop("y", axis=1)
    y = data["y"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "logistic": LogisticRegression(),
        "decision_tree": DecisionTreeClassifier(),
        "knn": KNeighborsClassifier(),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(),
        "xgboost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }

    # Train and save
    for name, model in models.items():
        model.fit(X_train, y_train)

        with open(f"model/{name}.pkl", "wb") as f:
            pickle.dump(model, f)

    print("All models trained and saved successfully.")
