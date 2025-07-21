from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="Water Quality Prediction API")

# Load XGBoost models
model_ph = joblib.load("models/xgboost_ph.pkl")
model_tur = joblib.load("models/xgboost_tur.pkl")
model_cond = joblib.load("models/xgboost_cond.pkl")


# Define expected input structure
class InputFeatures(BaseModel):
    pH: float
    Tur: float
    Cond: float


# Prediction endpoint
@app.post("/predict/")
def predict_next_month(data: InputFeatures):
    input_data = np.array([[data.pH, data.Tur, data.Cond]])

    return {
        "predicted_pH_next": round(model_ph.predict(input_data)[0], 4),
        "predicted_Turbidity_next": round(model_tur.predict(input_data)[0], 4),
        "predicted_Conductivity_next": round(model_cond.predict(input_data)[0], 6),
    }
