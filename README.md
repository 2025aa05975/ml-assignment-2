# Machine Learning Assignment 2

## a. Problem Statement

The objective of this assignment is to build, evaluate, and compare multiple machine learning classification models on a real-world dataset. The task involves training different classifiers, evaluating their performance using standard metrics, and deploying the results through an interactive Streamlit web application. The goal is to analyze how different models perform on the same dataset and understand their strengths and limitations.

---

## b. Dataset Description

The Bank Marketing dataset is used for this assignment. The dataset contains information related to marketing campaigns conducted by a banking institution. The objective is to predict whether a client will subscribe to a term deposit, making it a binary classification problem.

- Type: Binary Classification  
- Number of Features: 16  
- Number of Instances: More than 45,000  
- Target Variable: `y` (subscription: yes or no)  
- Source: UCI / Kaggle Bank Marketing Dataset  

The dataset consists of both categorical and numerical features. Categorical variables were encoded using Label Encoding, and feature scaling was applied using StandardScaler prior to training the models.

---

## c. Models Used and Evaluation Metrics

The following six machine learning classification models were implemented and evaluated using standard performance metrics:

- Accuracy  
- AUC (Area Under the ROC Curve)  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

---

### Comparison Table

| ML Model Name       | Accuracy | AUC    | Precision | Recall  | F1 Score | MCC    |
|--------------------|---------|--------|-----------|--------|----------|--------|
| Decision Tree      | 0.8630  | 0.6497 | 0.3700    | 0.3776 | 0.3737   | 0.2968 |
| KNN                | 0.8950  | 0.7250 | 0.5385    | 0.2143 | 0.3066   | 0.2938 |
| Logistic Regression| 0.8928  | 0.8634 | 0.5152    | 0.1735 | 0.2595   | 0.2547 |
| Naive Bayes        | 0.8376  | 0.8009 | 0.3185    | 0.4388 | 0.3691   | 0.2833 |
| Random Forest      | 0.9028  | 0.9160 | 0.6000    | 0.3061 | 0.4054   | 0.3827 |
| XGBoost            | 0.8751  | 0.8215 | 0.4336    | 0.5000 | 0.4645   | 0.3955 |

---

## d. Observations on Model Performance 

| ML Model Name       | Observation about model performance |
|--------------------|-------------------------------------|
| Decision Tree      | Good overall accuracy, but low precision and F1, indicating difficulty in predicting positive class accurately. |
| KNN                | High accuracy but very low recall, missing many positive cases; sensitive to class imbalance. |
| Logistic Regression| High accuracy but extremely low recall, poor at detecting positive class. |
| Naive Bayes        | Moderate accuracy, better recall than precision, suggests bias toward predicting positives. |
| Random Forest      | Highest accuracy and good overall balance of metrics; ensemble reduces overfitting. |
| XGBoost            | Slightly lower accuracy than Random Forest but better recall and F1; handles class imbalance well. |

---

## Deployment

The machine learning models are deployed using a Streamlit web application. The application provides:

- CSV dataset upload option (test data)  
- Model selection dropdown  
- Display of evaluation metrics  
- Confusion matrix visualization  

The application was deployed using Streamlit Community Cloud.
