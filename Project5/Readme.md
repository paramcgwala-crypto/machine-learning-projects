# Adult Census Income Prediction

## Problem
Predict whether a person earns more than 50K per year or not, using census data (age, education, occupation, hours per week, etc.). This is a binary classification problem.

## Dataset
- Adult Census Income dataset (adult.csv)
- About 32,500 rows and 15 columns
- Target column: `income` (<=50K or >50K)
- Data is imbalanced: about 76% of people earn <=50K

## Steps Followed
1. Loaded the data and checked its shape, info and statistics
2. Visualised the target column to check the class imbalance
3. Cleaned the data: replaced '?' with NaN, removed missing values and duplicates, dropped `fnlwgt` and `education`
4. Converted the target to 0/1 and used one-hot encoding (`pd.get_dummies`) for text columns
5. Train-test split (80/20) with stratify
6. Scaled the data (used for Logistic Regression)
7. Trained 3 models and predicted on both train and test data
8. Evaluated using TP, TN, FP, FN, Confusion Matrix, Accuracy, Precision, Recall and F1 score
9. Compared all models and checked overfitting using train vs test scores
10. Plotted feature importance using Random Forest

## Models Used
| Model | Why used |
|-------|----------|
| Logistic Regression | Simple and fast baseline model |
| Decision Tree | Easy if-else rules, easy to explain |
| Random Forest | Many trees together, handles complex patterns, overfits less than one tree |

## Results
| Model | Train Acc | Test Acc | Precision | Recall | F1 |
|-------|-----------|----------|-----------|--------|----|
| Logistic Regression | ___ | ___ | ___ | ___ | ___ |
| Decision Tree | ___ | ___ | ___ | ___ | ___ |
| Random Forest | ___ | ___ | ___ | ___ | ___ |

**Best model:** ___ (highest F1 score of ___)

## Key Insights
- Accuracy alone is misleading because the data is imbalanced, so F1 score and the confusion matrix were also used
- Top features: ___, ___, ___
- (Write one line on overfitting, like "Random Forest has higher train accuracy than test accuracy, so it overfits a little")

## Tools Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## Future Improvements
- Hyperparameter tuning with GridSearchCV
- Handle imbalance using SMOTE or class weights
- Try XGBoost