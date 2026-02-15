import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from model.load_and_evaluate import load_model, evaluate_model


st.title("Machine Learning Assignment 2")
st.write("Comparison of Classification Models on Bank Dataset")

# -------------------------------
# Download Sample Test Dataset
# -------------------------------
with open("bank_test.csv", "rb") as file:
    st.download_button(
        label="Download Sample Test Dataset",
        data=file,
        file_name="bank_test.csv",
        mime="text/csv"
    )

# -------------------------------
# Upload Test Dataset
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Test CSV File",
    type=["csv"]
)

model_dict = {
    "Logistic Regression": "logistic",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost"
}

selected_model = st.selectbox(
    "Select a Model",
    list(model_dict.keys())
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file, sep=";")
    data.columns = data.columns.str.strip().str.lower()

    X = data.drop("y", axis=1)
    y = data["y"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = load_model(model_dict[selected_model])
    metrics, cm = evaluate_model(model, X, y)

    st.subheader("Evaluation Metrics")
    for key, value in metrics.items():
        st.write(f"{key}: {value:.4f}")

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

else:
    st.info("Please upload the test dataset to evaluate the model.")
