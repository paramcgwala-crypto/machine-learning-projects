# Heart Failure Clinical Indicator Prediction 🏥

## Project Overview

A high-impact healthcare predictive model designed to assess patient clinical records and determine the probability of heart failure or structural cardiovascular risk events.

## Dataset Features

- **Age / Sex:** Patient demographics
- **ChestPainType:** Categorized clinical pain symptoms (ATA, NAP, ASY, TA)
- **RestingBP / Cholesterol:** Vital clinical thresholds
- **FastingBS:** Fasting blood sugar indicators
- **RestingECG:** Electrocardiogram status metrics
- **MaxHR / ExerciseAngina:** Stress test response indicators
- **HeartDisease:** Target Variable (0 = Normal, 1 = Heart Disease Risk)

## Workflow & Model

1. **Structural Auditing:** Confirmed shape constraints and verified full data integrity (0 missing values).
2. **Clinical Correlation:** Checked feature behavior and risk statistics across genders and chest pain types.
3. **One-Hot Encoding:** Encoded complex text parameters into binary indicator arrays.
4. **Feature Scaling:** Uniformly distributed clinical feature variances using `StandardScaler`.
5. **Model Training:** Utilized **Logistic Regression** to ensure full mathematical explainability for medical validation.
6. **Evaluation:** Checked standard confusion structures alongside exact Accuracy, Precision, and Recall scores.
