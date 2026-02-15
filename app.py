import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

st.title("Machine Learning Assignment 2")
st.write("Comparison of Classification Models on Bank Marketing Dataset")

# -------------------------------
# Download Sample Test Dataset
# -------------------------------
if os.path.exists("bank_test.csv"):
    with open("bank_test.csv", "rb") as f:
        st.download_button(
            label="Download Sample Test Dataset",
            data=f,
            file_name="bank_test.csv",
            mime="text/csv"
        )

# -------------------------------
# Load all saved models
# -------------------------------
MODEL_DIR = "model"
saved_models = {}
if os.path.exists(MODEL_DIR):
    for model_file in os.listdir(MODEL_DIR):
        if model_file.endswith(".pkl"):
            model_name = model_file.replace(".pkl", "").replace("_", " ").title()
            with open(os.path.join(MODEL_DIR, model_file), "rb") as f:
                saved_models[model_name] = pickle.load(f)

# -------------------------------
# Upload Test Dataset
# -------------------------------
uploaded_file = st.file_uploader("Upload CSV Dataset (Test Data Only)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=";")
    st.success("Dataset uploaded successfully!")

    if not saved_models:
        st.warning("No trained models found in 'model/' folder. Please train models first.")
    else:
        # Model selection
        chosen_model = st.selectbox("Select a Machine Learning Model", list(saved_models.keys()))
        model = saved_models[chosen_model]

        # Preprocess dataset like training
        df_copy = df.copy()
        for col in df_copy.columns:
            if df_copy[col].dtype == "object":
                df_copy[col] = df_copy[col].astype('category').cat.codes

        X_test = df_copy.drop("y", axis=1)
        y_test = df_copy["y"]

        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)

        # Predict
        y_pred = model.predict(X_test_scaled)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_prob = None

        # Compute metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, matthews_corrcoef,
            confusion_matrix
        )

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "MCC": matthews_corrcoef(y_test, y_pred),
        }
        if y_prob is not None:
            metrics["AUC"] = roc_auc_score(y_test, y_prob)
        else:
            metrics["AUC"] = 0.0

        # Display metrics
        st.subheader("Evaluation Metrics")
        for key, value in metrics.items():
            st.write(f"{key}: {value:.4f}")

        # Confusion matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        st.pyplot(fig)

else:
    st.info("Please upload a CSV file to continue.")
