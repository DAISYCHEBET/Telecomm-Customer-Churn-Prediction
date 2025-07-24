# Telecommunication Customer Churn Prediction

## Overview
This project aims to predict customer churn in a telecommunications company using an Artificial Neural Network (ANN). Customer churn is a key performance metric in the telecom industry, and predicting it allows the business to take proactive steps to retain at-risk customers.

Cleaned and preprocessed customer dataset, selected the most influential features using SHAP analysis, and built an ANN model using TensorFlow/Keras. The project is also deployed as an interactive Streamlit web app that allows both individual and batch predictions.

## Dataset
- **Source**:  Kaggle 
- **Records**: Customer information including service usage, contract type, and billing
- **Target Variable**: `Churn Value` (0 = Stayed, 1 = Churned)

### Features Used (based on SHAP selection)
- Senior Citizen  
- Partner  
- Dependents  
- Tenure Months  
- Online Security  
- Tech Support  
- Paperless Billing  
- Monthly Charges  
- Internet Service: Fiber optic  
- Contract: Month-to-month  
- Payment Method: Electronic check  



## Methods Used
- Data preprocessing (scaling, encoding, null handling)
- Feature selection using SHAP
- Neural network modeling with Keras
- Evaluation using classification report & confusion matrix
- Deployment using Streamlit (web interface)



## Results
- **Model**: Artificial Neural Network (3-layer)
- **Performance**: Evaluated using test data and classification metrics
- **Insights**:
  - Customers with fiber optic internet and month-to-month contracts are more likely to churn
  - Features like tech support and tenure had strong predictive power



## Streamlit App Features
- **Sidebar Upload**: Upload an Excel file for batch prediction
- **Visualization Panel**: Displays churn trends and feature distributions
- **Individual Prediction**: Select input features and predict churn
- **Download Reports**: Download filtered lists of customers likely to churn or stay



## Technologies Used
- Python 
- pandas, numpy  
- scikit-learn  
- TensorFlow/Keras  
- SHAP  
- Streamlit  
- matplotlib, seaborn


## How to Run the App Locally
1. Clone the repository

2. Install dependencies:
pip install -r requirements.txt

3. Run the Streamlit app:
streamlit run app.py
