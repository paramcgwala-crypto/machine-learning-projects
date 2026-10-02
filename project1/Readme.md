# Titanic: Machine Learning from Disaster 🚢

## Project Overview
This project predicts the survival probability of passengers aboard the Titanic using clinical and demographic data such as Age, Sex, Passenger Class, and Fare. It implements an end-to-end Machine Learning pipeline.

## Dataset Features
- **Survived:** Target Variable (0 = No, 1 = Yes)
- **Pclass:** Ticket class (1st, 2nd, 3rd)
- **Sex:** Passenger gender
- **Age:** Passenger age (Missing values handled via Median Imputation)
- **Fare:** Ticket price

## Workflow & Model
1. **Exploratory Data Analysis (EDA):** Analyzed survival rates across different genders and classes.
2. **Data Preprocessing:** Handled missing continuous data and dropped non-predictive columns (`PassengerId`, `Name`, `Ticket`, `Cabin`).
3. **Categorical Encoding:** Converted categorical text columns into numbers using One-Hot Encoding (`pd.get_dummies`).
4. **Feature Scaling:** Normalized continuous variables using `StandardScaler`.
5. **Model Training:** Trained a standard **Logistic Regression** model.
6. **Evaluation:** Evaluated performance using Accuracy, Precision, Recall, and Confusion Matrix.
