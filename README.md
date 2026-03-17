📊 Explainable Customer Churn Prediction (XAI)

🚀 Live Demo:
https://explainablechurnprediction-9wwor7vfa2dshrt49gkwyb.streamlit.app/

👨‍💻 GitHub Profile:
https://github.com/misterk1269

📌 Project Overview

Customer churn is a critical problem for many businesses such as banks, telecom companies, and subscription platforms. When customers leave a service, companies lose revenue and growth opportunities.

This project builds an Explainable AI powered Customer Churn Prediction system using Machine Learning and Deep Learning techniques.

The model predicts whether a customer is likely to churn and provides transparent explanations using SHAP, helping businesses understand the factors influencing churn.

🎯 Key Objectives

✔ Predict customer churn probability
✔ Identify high-risk customers
✔ Provide interpretable explanations using Explainable AI
✔ Build an interactive web application for real-time predictions

⚙️ Tech Stack

Programming Language
Python

Machine Learning
TensorFlow / Keras
Scikit-learn

Explainable AI
SHAP (SHapley Additive Explanations)

Web Application
Streamlit

Data Processing
Pandas
NumPy

Visualization
Matplotlib

🧠 Model Pipeline

1️⃣ Data preprocessing and feature scaling using StandardScaler

2️⃣ Model training using TensorFlow Neural Network

3️⃣ Model saved as:

churn_model.h5

4️⃣ Scaler saved as:

scaler.save

5️⃣ Explainability added using SHAP

6️⃣ Web interface built using Streamlit

✨ Application Features

🔹 Interactive customer input interface

🔹 Real-time churn prediction

🔹 Churn probability output

🔹 Low Risk / High Risk classification

🔹 SHAP Explainability Visualization to understand feature impact

🔹 Deployed as a live Streamlit web application

📊 Example Inputs

The application accepts customer attributes such as:

Credit Score

Age

Tenure

Balance

Number of Products

Customer Activity Status

Based on these features, the model predicts the likelihood of churn.

🚀 Run Locally

Clone the repository

git clone https://github.com/misterk1269

Install dependencies

pip install -r requirements.txt

Run the Streamlit app

streamlit run app.py
☁️ Deployment

The application is deployed using Streamlit Cloud, allowing users to interact with the model directly through a web browser.

Live application:

https://explainablechurnprediction-9wwor7vfa2dshrt49gkwyb.streamlit.app/
