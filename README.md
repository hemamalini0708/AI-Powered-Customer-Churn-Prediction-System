![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-red)
![AUC](https://img.shields.io/badge/AUC-0.88-brightgreen)

# AI-Powered Customer Churn Prediction System

An end-to-end Machine Learning project that predicts whether a telecom customer will **Churn (Leave)** or **Stay**, using the Telco Customer Churn dataset.

This project demonstrates a complete production-ready ML pipeline — from advanced preprocessing to cloud deployment.

---

## Problem Statement

Customer churn is a major challenge for telecom companies.  
The objective of this project is to build a classification model that identifies high-risk customers and enables proactive retention strategies.

Target Variable:
- 1 → Churn
- 0 → No Churn

---

## Dataset

Dataset: IBM Telco Customer Churn Dataset  
Features include:

- Tenure  
- Monthly Charges  
- Total Charges  
- Contract Type  
- Internet Services  
- Payment Method  
- Demographic information  

Final Model uses **17 selected customer attributes**.

---

## Machine Learning Pipeline

### 1️. Data Preprocessing
- Iterative Imputation for missing values
- Gaussian Winsorization for outlier treatment
- Quantile Transformation for skewness reduction
- Multicollinearity removal using correlation analysis

### 2️. Class Imbalance Handling
- SMOTE balancing  
  (Original: 27:73 → Balanced: 50:50)

### 3️. Feature Selection
- Chi-Square Test
- ANOVA F-Test

---

##  Models Trained & Compared

- Logistic Regression  
- Random Forest  
- XGBoost  
- SVM  
- Gradient Boosting  
- KNN  
- Naive Bayes  
- Decision Tree  
- AdaBoost  

### Final Selected Model:
**Gradient Boosting Classifier**

Performance:
- AUC: ~0.88  
- Accuracy: ~79%

---

## Model Evaluation

Evaluation Metrics:
- Accuracy Score  
- Confusion Matrix  
- Classification Report  
- ROC-AUC Curve Comparison  

The ROC curve was used to compare performance across all models.

<img width="1174" height="711" alt="image" src="https://github.com/user-attachments/assets/d4e1de55-358b-437c-ae5c-494955b435c9" />

---

## Deployment

The trained model is deployed using **Streamlit**.

Features of Web App:
- Real-time churn prediction
- Probability confidence score
- Actionable business recommendations

Deployed on **Streamlit Cloud**.

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app/telecom_churn_streamlit.py
```

**Tech Stack**
- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Streamlit

**Project Structure**
```
AI-Powered-Customer-Churn-Prediction-System/
│
├── app/
├── data/
├── model/
├── src/
├── requirements.txt
├── README.md
└── .gitignore
```

**Key Highlights**

✔ Advanced preprocessing techniques

✔ Feature engineering & selection

✔ Class imbalance handling with SMOTE

✔ Multi-model comparison

✔ ROC-based model selection

✔ Cloud deployment

---

**Author**

Hema Malini Gangumalla

Aspiring Data Scientist

📧 hemamalinig07@gmail.com

---
**License**

MIT License
---

