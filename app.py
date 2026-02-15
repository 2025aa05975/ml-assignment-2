import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from model.train_models import train_all_models

st.title("Machine Learning Assignment 2")
st.write("Comparison of classification models on Bank Marketing Dataset")

# Dataset upload
uploaded_file = st.file_uploader("Upload CSV Dataset (Test Data Only)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, sep=";")

    st.success("Dataset uploaded successfully")

    # Train and evaluate
    results = train_all_models(data)

    selected_model = st.selectbox(
        "Select a Machine Learning Model",
        list(results.keys())
    )

    st.subheader("Evaluation Metrics")

    metrics = results[selected_model]

    for key, value in metrics.items():
        if key != "Confusion Matrix":
            st.write(f"{key}: {value:.4f}")

    st.subheader("Confusion Matrix")

    cm = metrics["Confusion Matrix"]

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    st.pyplot(fig)

else:
    st.info("Please upload a CSV file to continue.")
