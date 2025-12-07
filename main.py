from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# Load trained model
model = joblib.load("aflatoxin_model.pkl")

# Initialize FastAPI app
app = FastAPI()

# Allow all CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request schema
class InputData(BaseModel):
    temp_c: float
    relative_humidity_pct: float
    co2_ppm: float
    moisture_pct: float
    storage_day: int

# Define risk level logic
def risk_level(ppb):
    if ppb < 5:
        return "low"
    elif ppb < 20:
        return "medium"
    else:
        return "high"

# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    X = [[
        data.moisture_pct,
        data.temp_c,
        data.relative_humidity_pct,
        data.co2_ppm,
        data.storage_day
    ]]

    ppb = model.predict(X)[0]
    risk = risk_level(ppb)

    return {
        "predicted_ppb": round(float(ppb), 2),
        "predicted_risk": risk
    }
