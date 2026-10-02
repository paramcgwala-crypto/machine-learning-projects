# Bengaluru House Price Prediction 🏡

## Project Overview

This is a Supervised Learning **Regression** project designed to predict real estate property prices in Bengaluru based on various structural and locational features.

## Dataset Features

- **Location:** The area/neighborhood in Bengaluru
- **Size/BHK:** Number of bedrooms
- **Total_Sqft:** Total area of the property in square feet
- **Bath:** Number of bathrooms
- **Price:** Target Variable (Continuous value representing price in Lakhs)

## Workflow & Model

1. **Data Cleaning:** Handled missing entries, removed anomalies/outliers in square footage and BHK counts.
2. **Feature Engineering:** Simplified structural descriptions into uniform numeric features.
3. **Dimensionality Reduction:** Managed high-cardinality locations by tagging rare areas as 'other'.
4. **Model Training:** Implemented a **Linear Regression** framework along with Cross-Validation techniques to ensure stability.
5. **Evaluation:** Performance measured using R-squared score and Mean Absolute Error (MAE).
