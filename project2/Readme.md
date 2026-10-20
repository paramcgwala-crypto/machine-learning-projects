# Bengaluru House Price Prediction

## Problem

Predict the price (in lakhs) of houses in Bengaluru using details like area, BHK, bathrooms and location. The project also classifies each house as Cheap or Expensive.

## Dataset

- Bengaluru house prices dataset (bengaluru_house_prices.csv)
- About 13,300 rows and 9 columns
- Target column: `price` (in lakhs)
- About 11,300 rows were left after cleaning and outlier removal

## Steps Followed

1. Loaded the data and checked missing values and statistics
2. Cleaned the data: dropped `society` and `availability`, filled `balcony` with median, removed missing and duplicate rows
3. Made a `bhk` column, converted `total_sqft` to numbers and grouped rare locations as 'other'
4. Removed outliers using area per bedroom, bathroom count and price per sqft
5. Made 5 EDA graphs: price distribution, area vs price, BHK vs price, top 10 costliest locations and correlation heatmap
6. One-hot encoding (`pd.get_dummies`), train-test split (80/20) and scaling
7. Trained 4 regression models and compared them with R2, MAE, MSE and RMSE
8. Plotted actual vs predicted price, error distribution and feature importance
9. Part B: made Cheap vs Expensive classes using median price and trained 2 classification models
10. Compared all models using TP, TN, FP, FN, Confusion Matrix, Accuracy, Precision, Recall and F1

## Models Used

| Model                | Type           | Why used                               |
| -------------------- | -------------- | -------------------------------------- |
| Linear Regression    | Regression     | Simple baseline                        |
| KNN                  | Regression     | Predicts from 5 nearest similar houses |
| Decision Tree        | Regression     | Easy if-else rules                     |
| Random Forest        | Regression     | Many trees together                    |
| Logistic Regression  | Classification | Separates Cheap and Expensive houses   |
| Gaussian Naive Bayes | Classification | Uses probability to decide the class   |

## Results

### Part A: Regression (predict price)

| Model             | Train R2 | Test R2 | Test MAE | Test RMSE |
| ----------------- | -------- | ------- | -------- | --------- |
| Linear Regression | 0.739    | 0.743   | 30.1     | 56.9      |
| KNN               | 0.777    | 0.570   | 31.2     | 73.5      |
| Decision Tree     | 0.872    | 0.602   | 30.4     | 70.7      |
| Random Forest     | 0.955    | 0.740   | 26.7     | 57.2      |

**Best regression model:** Linear Regression (Test R2 = 0.743)

### Part B: Classification (Cheap vs Expensive, test data)

| Model                    | Accuracy | Precision | Recall | F1    |
| ------------------------ | -------- | --------- | ------ | ----- |
| Best Regression (Linear) | 0.818    | 0.767     | 0.905  | 0.830 |
| Logistic Regression      | 0.852    | 0.868     | 0.825  | 0.846 |
| Gaussian Naive Bayes     | 0.637    | 0.827     | 0.335  | 0.477 |

**Best classification model:** Logistic Regression (F1 = 0.846)

## Key Insights

- Total area (`total_sqft`) is the most important feature for price, then bathrooms, BHK and area type
- Linear Regression has almost equal Train and Test R2, so it does not overfit
- Random Forest has the lowest MAE, but its Train R2 (0.955) is much higher than Test R2 (0.740), so it overfits
- Gaussian Naive Bayes is the weakest because most columns are 0/1 location columns, so it misses many expensive houses
- Errors are higher for very costly houses

## Tools Used

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## Future Improvements

- Log transform of price
- Hyperparameter tuning with GridSearchCV
- Add location-based price features
- Try XGBoost
