# Machine Learning Assignment 2

## a. Problem Statement

The objective of this assignment is to build, evaluate, and compare multiple machine learning classification models on a real-world dataset. The task involves training different classifiers, evaluating their performance using standard metrics, and deploying the results through an interactive Streamlit web application. The goal is to analyze how different models perform on the same dataset and understand their strengths and limitations.

---

## b. Dataset Description  [1 Mark]

The Bank Marketing dataset is used for this assignment. The dataset contains information related to marketing campaigns conducted by a banking institution. The objective is to predict whether a client will subscribe to a term deposit, making it a binary classification problem.

- Type: Binary Classification  
- Number of Features: 16  
- Number of Instances: More than 45,000  
- Target Variable: `y` (subscription: yes or no)  
- Source: UCI / Kaggle Bank Marketing Dataset  

The dataset consists of both categorical and numerical features. Categorical variables were encoded using Label Encoding, and feature scaling was applied using StandardScaler prior to training the models.

---

## c. Models Used and Evaluation Metrics  [6 Marks]

The following six machine learning classification models were implemented and evaluated using standard performance metrics:

- Accuracy  
- AUC (Area Under the ROC Curve)  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

---

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|----------|------|-----------|--------|----------|------|
| Logistic Regression | 0.8950 | 0.8197 | 0.5000 | 0.1579 | 0.2400 | 0.2386 |
| Decision Tree | 0.8950 | 0.6626 | 0.5000 | 0.3684 | 0.4242 | 0.3731 |
| kNN | 0.8840 | 0.6462 | 0.2500 | 0.0526 | 0.0870 | 0.0711 |
| Naive Bayes | 0.8508 | 0.7511 | 0.3182 | 0.3684 | 0.3415 | 0.2587 |
| Random Forest (Ensemble) | 0.9061 | 0.9019 | 0.7500 | 0.1579 | 0.2609 | 0.3163 |
| XGBoost (Ensemble) | 0.8950 | 0.8441 | 0.5000 | 0.2632 | 0.3448 | 0.3117 |

---

## d. Observations on Model Performance  [3 Marks]

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | Acts as a strong baseline model with good accuracy and AUC, but low recall indicates difficulty in identifying positive cases. |
| Decision Tree | Achieves better recall and F1 score compared to Logistic Regression, but shows lower AUC, indicating weaker ranking capability. |
| kNN | Exhibits lower recall and MCC, suggesting sensitivity to class imbalance and dependence on distance-based classification. |
| Naive Bayes | Provides moderate performance with balanced recall and F1 score, despite its strong independence assumptions. |
| Random Forest (Ensemble) | Achieves the highest accuracy and AUC, indicating strong overall performance, though recall remains limited. |
| XGBoost (Ensemble) | Shows strong AUC and balanced F1 score, providing improved performance through boosted ensemble learning. |

---

## Deployment

The machine learning models are deployed using a Streamlit web application. The application provides:
- CSV dataset upload option (test data)
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix visualization

The application was deployed using Streamlit Community Cloud.
