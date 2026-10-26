# Predictive Maintenance: Machine Failure Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange) ![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

An end-to-end machine learning project that predicts whether an industrial machine will fail, using sensor readings such as temperature, rotational speed, torque and tool wear. The goal is to let a maintenance team act **before** a breakdown happens.

## Table of Contents

- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Author](#author)

## Business Problem

Unplanned machine downtime is expensive. If a model can flag machines that are likely to fail, maintenance can be scheduled in advance. In this setting, **missing a real failure (low recall) is costly, and too many false alarms (low precision) waste maintenance effort**, so both are tracked together with F1 score.

## Dataset

- **Source:** AI4I 2020 Predictive Maintenance Dataset (synthetic data modelled on real industrial maintenance data)
- **Size:** 10,000 records, 14 columns, no missing values, no duplicates
- **Target:** `Machine failure` (1 = failed, 0 = did not fail)
- **Class imbalance:** only 339 failures (3.39%) vs 9,661 non-failures

| Column                  | Description                                                    |
| ----------------------- | -------------------------------------------------------------- |
| Type                    | Product quality variant (L / M / H)                            |
| Air temperature [K]     | Ambient air temperature                                        |
| Process temperature [K] | Temperature of the machining process                           |
| Rotational speed [rpm]  | Spindle speed                                                  |
| Torque [Nm]             | Torque applied                                                 |
| Tool wear [min]         | Time the tool has been in use                                  |
| TWF, HDF, PWF, OSF, RNF | Individual failure-mode flags (removed from inputs, see below) |

## Methodology

1. **Data quality checks:** missing values, duplicates, class distribution.
2. **Data leakage prevention:** the five failure-mode columns (TWF, HDF, PWF, OSF, RNF) directly describe why a machine failed, so they were dropped from the inputs. `UDI` and `Product ID` were dropped as identifiers.
3. **Feature engineering** (using basic physics and domain logic):
   - `Power` = Torque x Rotational speed (converted to watts)
   - `Temp_diff` = Process temperature - Air temperature
   - `Strain` = Tool wear x Torque
4. **Encoding:** `Type` mapped to ordinal values (L=0, M=1, H=2).
5. **Train/test split:** 80/20, stratified on the target so both sets keep the same failure ratio.
6. **Scaling:** `StandardScaler` fitted on the training set only and applied to the test set (no leakage).
7. **Modelling:** Logistic Regression (class-weight balanced), KNN (k=5), Gaussian Naive Bayes, Decision Tree (max_depth=6, min_samples_leaf=10) and Random Forest (200 trees, max_depth=10).
8. **Evaluation:** Accuracy, Precision, Recall and F1 on both train and test sets. Accuracy alone is misleading here, because predicting "no failure" for every machine would already give about 96.6%.
9. **Additional analysis:** Linear Regression (R²) and K-Means clustering.

## Results

| Model                | Train F1  | Test Accuracy | Test Precision | Test Recall | Test F1   |
| -------------------- | --------- | ------------- | -------------- | ----------- | --------- |
| Logistic Regression  | 0.275     | 0.858         | 0.178          | 0.882       | 0.296     |
| KNN                  | 0.632     | 0.975         | 0.875          | 0.309       | 0.457     |
| Gaussian Naive Bayes | 0.482     | 0.962         | 0.443          | 0.456       | 0.449     |
| Decision Tree        | 0.895     | 0.992         | 1.000          | 0.765       | 0.867     |
| **Random Forest**    | **0.929** | **0.992**     | **0.965**      | **0.809**   | **0.880** |

**Best model: Random Forest.** On the test set (68 real failures) it detected **55 failures**, missed 13, and raised only 2 false alarms.

|                        | Predicted: No failure | Predicted: Failure |
| ---------------------- | --------------------- | ------------------ |
| **Actual: No failure** | 1930                  | 2                  |
| **Actual: Failure**    | 13                    | 55                 |

**Additional analysis**

- **Linear Regression** (predicting process temperature): Train R² = 0.77, Test R² = 0.76. The close scores suggest no overfitting.
- **K-Means (k=3):** one cluster has a failure rate of about 5.9%, compared with about 2.2% and 1.7% in the other two. This can help in identifying higher-risk machines.

## Key Findings

- The most important features for the Random Forest were the engineered ones: **Strain, Power** and **Rpm**, followed by Torque and Temp_diff.
- Removing the failure-mode columns was essential. Keeping them would have made the task artificially easy.
- Logistic Regression has high recall (0.88) but very low precision (0.18), so it would generate many false alarms.
- Tree-based models clearly outperformed distance- and probability-based models (KNN, Naive Bayes) on this data.

## Limitations

- The test set has only 68 failure cases, so the reported F1 may vary with a different split.
- Cross-validation and hyperparameter tuning have not been done yet.
- The dataset is synthetic, so performance on real factory data may differ.

## Future Work

- Add stratified k-fold cross-validation for more reliable estimates.
- Tune hyperparameters and the decision threshold to balance precision and recall.
- Try gradient boosting models.
- Build a simple app where a user enters sensor values and gets a failure prediction.

## Project Structure

```
├── ai4i2020.csv          # dataset
├── ai4i2020.ipynb        # full notebook (EDA, modelling, evaluation)
└── README.md
```

## How to Run

```bash
git clone <your-repo-link>
cd <your-repo-folder>
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook
```

Open `ai4i2020.ipynb` and choose **Run All**. Keep `ai4i2020.csv` in the same folder as the notebook.

## Tech Stack

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn
