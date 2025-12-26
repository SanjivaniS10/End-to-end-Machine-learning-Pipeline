# 🏠 End-to-End Housing Price Prediction using Machine Learning

## 📌 Project Overview
This project implements a **complete end-to-end machine learning workflow** for predicting **median house values** using housing data.  
It covers everything from **data loading and exploration** to **feature engineering, model training, evaluation, hyperparameter tuning, and model persistence**.

The project follows **industry-standard ML practices** using **Scikit-learn Pipelines and ColumnTransformers**, ensuring reproducibility and scalability.

---

## 🎯 Objectives
- Perform **Exploratory Data Analysis (EDA)** on housing data
- Apply **train-test splitting** with stratified sampling
- Handle **missing values and categorical features**
- Engineer **custom attributes**
- Build **ML pipelines**
- Train and compare multiple regression models
- Tune hyperparameters using **GridSearchCV**
- Evaluate performance using **RMSE**
- Save the final trained model for reuse

---

## 📊 Dataset Information
- **Rows:** 2000  
- **Features:** 10  
- **Target Variable:** `median_house_value`

### Features
- `longitude`
- `latitude`
- `housing_median_age`
- `total_rooms`
- `total_bedrooms`
- `population`
- `households`
- `median_income`
- `ocean_proximity` *(categorical)*

---

## 🔍 Exploratory Data Analysis (EDA)
- Dataset inspection using `.info()` and `.describe()`
- Distribution analysis using histograms
- Category frequency analysis for `ocean_proximity`
- Geographical visualization using scatter plots
- Correlation analysis and scatter matrix visualization
- Income-based stratification to avoid sampling bias

---

## 🧠 Feature Engineering
Custom features were created to improve model performance:
- `rooms_per_household`
- `population_per_household`
- `bedrooms_per_room`

A **custom Scikit-learn transformer (`CombinedAttributesAdder`)** was implemented for seamless integration into pipelines.

---

## ⚙️ Data Preprocessing Pipeline
### Numerical Pipeline
- Median imputation (`SimpleImputer`)
- Custom feature generation
- Feature scaling (`StandardScaler`)

### Categorical Pipeline
- One-hot encoding (`OneHotEncoder`)

Both pipelines were combined using **ColumnTransformer** for clean preprocessing.

---

## 🤖 Models Implemented
- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**

---

## 📈 Model Evaluation
Evaluation metric used:
- **RMSE (Root Mean Squared Error)**

### Cross-Validation Results
| Model | Mean RMSE |
|------|-----------|
| Linear Regression | ~143,415 |
| Decision Tree | ~201,548 |
| Random Forest | ~144,351 |

---

## 🔧 Hyperparameter Tuning
- Performed using **GridSearchCV**
- Optimized `n_estimators`, `max_features`, and `bootstrap`
- Best Model:
```python
RandomForestRegressor(n_estimators=30, max_features=2)
