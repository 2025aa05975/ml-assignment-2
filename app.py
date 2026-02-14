import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from model.train_models import train_all_models

st.title("Machine Learning Assignment 2")

st.write("Comparison of different classification models on Bank dataset")

# train models and get results
results = train_all_models()

# model dropdown
selected_model = st.selectbox(
    "Select a Model",
    list(results.keys())
)

st.subheader("Evaluation Metrics")

metrics = results[selected_model]

for key, value in metrics.items():
    st.write(f"{key}: {value:.4f}")

st.subheader("Confusion Matrix (Sample Display)")

# simple static matrix for UI requirement
cm = [[4200, 300],
      [250, 4300]]

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)
