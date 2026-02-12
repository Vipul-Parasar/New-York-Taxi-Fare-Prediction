# 🚕 New York City Taxi Fare Prediction

## Overview

This project builds a machine learning model to predict taxi fares in New York City using historical ride data from Kaggle.

Given:
- Pickup date & time
- Pickup latitude & longitude
- Dropoff latitude & longitude
- Passenger count

The objective is to predict the `fare_amount` for unseen rides and submit predictions to Kaggle.

Dataset Link:  
https://www.kaggle.com/c/new-york-city-taxi-fare-prediction

Evaluation Metric: **Root Mean Squared Error (RMSE)**

---

## Dataset Summary

### Training Set
- 55.4 million rows
- 5.4 GB
- Columns:
  - key
  - fare_amount (target)
  - pickup_datetime
  - pickup_longitude
  - pickup_latitude
  - dropoff_longitude
  - dropoff_latitude
  - passenger_count

### Test Set
- 9,914 rows
- Same columns except fare_amount
- Submission format:
  key,fare_amount

---

## Project Workflow

### 1. Data Download
- Used `opendatasets` to download dataset from Kaggle
- Extracted 1.56GB archive
- Worked in Google Colab

---

### 2. Data Preparation

Since the dataset is very large:
- Used 1% sample (~550K rows) for experimentation
- Dropped `key` column
- Optimized dtypes (float32, uint8)
- Removed missing coordinates
- Removed geographic outliers using test set bounds:
  - Latitude between 40 and 42
  - Longitude between -75 and -72

Final cleaned dataset size: ~542K rows

---

## Exploratory Data Analysis (EDA)

Key observations:
- Friday and Saturday are the busiest days
- Taxi demand peaks between 5 PM and 11 PM
- No strong correlation between fare and day of week
- Average fare increases year over year (inflation effect)
- Average ride displacement ≈ 3.33 km

Visualizations performed:
- Passenger count distribution
- Fare distribution (KDE plot)
- Pickup and dropoff scatter plots
- Hourly demand analysis
- Monthly and yearly fare trends
- High-fare pickup and dropoff mapping using Folium

---

## Feature Engineering

### Baseline Features
- Pickup longitude & latitude
- Dropoff longitude & latitude
- Passenger count

### Engineered Features

1. Ride Displacement
- Calculated using Haversine formula
- Produced major RMSE reduction

2. Time Features
- Year
- Month
- Day
- Hour
- Day of week
- Only Year had noticeable impact

3. Distance from Key Landmarks

Added pickup and dropoff distance from:
- JFK Airport
- LGA Airport
- EWR Airport
- Times Square
- Metropolitan Museum
- World Trade Center

These features significantly improved performance.

---

## Model Training

### Baseline Model (Mean Prediction)
Validation RMSE ≈ 9.68

### Linear Regression
Validation RMSE ≈ 9.68

After adding displacement feature:
Validation RMSE ≈ 5.41

After adding landmark features:
Validation RMSE ≈ 4.80

---

## Models Evaluated

- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net
- KNN
- Decision Tree
- Random Forest
- Gradient Boosting

### Best Model: Gradient Boosting Regressor

Parameters:
- max_depth = 6
- n_estimators = 200
- learning_rate = 0.5
- loss = squared_error

Validation RMSE ≈ 3.74  
Kaggle Score ≈ 3.25 (using 1% of dataset)

---

## Hyperparameter Tuning Strategy

- Tune n_estimators first
- Fix best value
- Tune max_depth
- Tune learning_rate
- Iterate for marginal gains

Validation RMSE closely matched Kaggle score, confirming proper split strategy.

---

## Files Generated

- train_input.csv
- train_target.csv
- val_input.csv
- val_target.csv
- test_inputs.csv
- Submission CSVs for each model

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Folium
- Google Colab
- Kaggle API

---

## Key Learnings

- Feature engineering has greater impact than model complexity.
- Distance-based features are critical in geospatial regression problems.
- Domain knowledge significantly improves model performance.
- Proper validation strategy is essential for leaderboard consistency.

---

## Future Improvements

- Train on full 55M rows
- Use XGBoost or LightGBM
- Vectorize Haversine computation
- Apply cross-validation
- Add weather and traffic features
- Improve memory efficiency
