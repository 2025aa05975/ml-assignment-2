import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from model.train_models import train_all_models

st.title("Machine Learning Assignment 2")
st.write("Comparison of Classification Models on Bank Marketing Dataset")

# -------------------------------
# Sample Dataset Download
# -------------------------------
with open("bank_test.csv", "rb") as f:
    st.download_button(
        label="Download Sample Test Dataset",
        data=f,
        file_name="bank_test.csv",
        mime="text/csv"
    )

# -------------------------------
# Upload Dataset
# -------------------------------
uploaded_file = st.file_uploader("Upload CSV Dataset (Test Data Only)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=";")
    st.success("Dataset uploaded successfully!")

    # Train models & get metrics (also saves all models)
    metrics_dict = train_all_models(df)

    # Model selection
    chosen_model = st.selectbox("Select a Machine Learning Model", list(metrics_dict.keys()))

    st.subheader("Evaluation Metrics")
    model_metrics = metrics_dict[chosen_model]
    for metric, value in model_metrics.items():
        if metric != "Confusion Matrix":
            st.write(f"{metric}: {value:.4f}")

    st.subheader("Confusion Matrix")
    cm = model_metrics["Confusion Matrix"]
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    st.pyplot(fig)

else:
    st.info("Please upload a CSV file to continue.")
