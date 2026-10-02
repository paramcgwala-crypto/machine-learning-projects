# Telecom Customer Churn Prediction 📞

## Project Overview

This project targets customer retention by building a predictive model that identifies telecom subscribers who are highly likely to cancel their subscriptions (churn).

## Dataset Features

- **Demographics:** Gender, Senior Citizen status, Partners, and Dependents.
- **Services:** Phone service, Multiple lines, Internet type, Online security, Streaming choices.
- **Account Info:** Contract type, Paperless billing, Payment method, Monthly charges, Total charges.
- **Churn:** Target Variable (Yes/No status of customer exit)

## Workflow & Model

1. **Data Inspection:** Validated column structures and cast total charges into appropriate numerical scales.
2. **Preprocessing:** Cleaned missing values and prepared structured categorical features.
3. **Feature Encoding:** Processed categorical markers into machine-readable numeric dummy fields.
4. **Data Normalization:** Scaled numeric features uniformly via `StandardScaler`.
5. **Model Training:** Deployed a **Logistic Regression** classifier for clear classification boundaries.
6. **Evaluation:** Assessed via Classification Report, Precision-Recall trade-offs, and Confusion Matrix.
