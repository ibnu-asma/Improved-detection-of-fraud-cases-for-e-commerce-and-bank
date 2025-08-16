# AI Mastery: Fraud Detection Challenge (Week 8 & 9)

## Project Overview

This project is aimed at improving the detection of fraud in e-commerce and bank transactions. The task involved performing data analysis, building and training machine learning models, ensuring model explainability, deploying models via an API, and creating a dashboard for visualizing fraud insights.

### Tasks Covered

- **Data Analysis and Preprocessing**
- **Model Building and Training**
- **Model Explainability using SHAP and LIME**
- **Model Deployment using Flask & Docker**
- **Building a Dashboard using Flask and Dash**

## Table of Contents

1. [Data Analysis and Preprocessing](#data-analysis-and-preprocessing)
2. [Model Building and Training](#model-building-and-training)
3. [Model Explainability](#model-explainability)
4. [Model Deployment and API Development](#model-deployment-and-api-development)
5. [Dashboard with Flask and Dash](#dashboard-with-flask-and-dash)
6. [References](#references)
7. [Installation](#installation)

## Data Analysis and Preprocessing

1. **Handling Missing Values**:  
   - Imputed or dropped missing values.

2. **Data Cleaning**:  
   - Removed duplicates.  
   - Corrected data types.

3. **Exploratory Data Analysis (EDA)**:  
   - Univariate and bivariate analysis.

4. **Geolocation Analysis**:  
   - Converted IP addresses to integer format.  
   - Merged `Fraud_Data.csv` with `IpAddress_to_Country.csv`.

5. **Feature Engineering**:  
   - Transaction frequency and velocity for `Fraud_Data.csv`.  
   - Time-based features: `hour_of_day`, `day_of_week`.

6. **Normalization and Scaling**:  
   - Applied normalization and scaling techniques.

7. **Encode Categorical Features**:  
   - Converted categorical features using encoding techniques.

## Model Building and Training

- **Data Preparation**:  
  - Separated features and target.  
  - Split the dataset into train and test sets.

- **Model Selection**:  
  Used various models and compared their performance:  
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - Gradient Boosting  
  - Multi-Layer Perceptron (MLP)  
  - Convolutional Neural Network (CNN)  
  - Recurrent Neural Network (RNN)  
  - Long Short-Term Memory (LSTM)

- **Model Training & Evaluation**:  
  - Trained and evaluated models using cross-validation.

- **MLOps**:  
  - Implemented experiment tracking and model versioning using MLflow.

## Model Explainability

- **SHAP (Shapley Additive exPlanations)**:  
  - Used SHAP to explain model predictions.  
  - Visualized feature importance with SHAP plots:  
    - Summary plot  
    - Force plot  
    - Dependence plot

- **LIME (Local Interpretable Model-agnostic Explanations)**:  
  - Used LIME for local model interpretability.  
  - Created LIME plots for individual prediction explanations.

## Model Deployment and API Development

- **Flask API**:  
  - Created a Flask application to serve the fraud detection model.  
  - Defined necessary API endpoints.

- **Dockerization**:  
  - Dockerized the Flask application for deployment.  
  - Dockerfile example:

    ```Dockerfile
    FROM python:3.8-slim
    WORKDIR /app
    COPY . .
    RUN pip install -r requirements.txt
    EXPOSE 5000
    CMD ["python", "serve_model.py"]
    ```

- **API Testing**:  
  - Ran tests to ensure API endpoints return expected results.

- **Logging**:  
  - Integrated Flask-Logging to track requests, errors, and predictions.

## Dashboard with Flask and Dash

- **Flask Endpoint**:  
  - Served fraud data and statistics via Flask API.

- **Dash Visualization**:  
  - Developed interactive dashboard for fraud detection insights:
    - Total transactions, fraud cases, and fraud percentages.
    - Line chart displaying fraud cases over time.
    - Geolocation fraud analysis.
    - Bar chart comparing fraud cases across devices and browsers.

## References

1. [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. [Kaggle - IEEE Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/code)
3. [Kaggle - Fraud E-Commerce Dataset](https://www.kaggle.com/datasets/vbinh002/fraud-ecommerce/code)
4. [Fraud Detection Overview](https://complyadvantage.com/insights/what-is-fraud-detection/)
5. [MLflow](https://www.mlflow.org/)
6. [SHAP Documentation](https://github.com/slundberg/shap)
7. [LIME Documentation](https://github.com/marcotcr/lime)

## Installation

### Requirements

- Python 3.8+
- Flask
- Dash
- MLflow
- SHAP
- LIME
- Docker

### Steps to Install

1. Clone this repository:

   ```bash
   git clone https://github.com/ab3lT/10-Academy-Week-8
   cd 10-Academy-Week-8


2. Install dependencies:

   ```pip install -r requirements.txt```
