from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

app = FastAPI(title="Churn Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model & scaler
model = load_model("churn_model.h5")
scaler = joblib.load("scaler.save")

# ✅ User-friendly input schema
class CustomerData(BaseModel):
    CreditScore: float
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: str         # "Yes" / "No"
    IsActiveMember: str    # "Active" / "Inactive"
    EstimatedSalary: float
    Geography: str         # "France", "Germany", "Spain"
    Gender: str            # "Male", "Female"

@app.get("/")
def read_root():
    return {"message": "Churn Prediction API is running!"}


@app.post("/predict")
def predict(data: CustomerData):
    # ✅ Encode categorical fields
    has_cr_card = 1 if data.HasCrCard.lower() == "yes" else 0
    is_active = 1 if data.IsActiveMember.lower() == "active" else 0
    
    geo_germany = 1 if data.Geography.lower() == "germany" else 0
    geo_spain = 1 if data.Geography.lower() == "spain" else 0
    # France default → (0,0)
    
    gender_male = 1 if data.Gender.lower() == "male" else 0

    # Final DataFrame (exact training order)
    df = pd.DataFrame([[
        data.CreditScore, data.Age, data.Tenure, data.Balance,
        data.NumOfProducts, has_cr_card, is_active, data.EstimatedSalary,
        geo_germany, geo_spain, gender_male
    ]], columns=[
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "EstimatedSalary",
        "Geography_Germany", "Geography_Spain", "Gender_Male"
    ])

    # Scale
    X = scaler.transform(df)

    # Predict
    prob = float(model.predict(X)[0][0])

    return {
        "churn_probability": prob,
        "prediction": "Churn" if prob > 0.4 else "Not Churn"
    }
