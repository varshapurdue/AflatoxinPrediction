from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# Load your trained model
model = joblib.load("aflatoxin_model.pkl")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for all origins (can restrict later if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["https://app.lovable.so"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input structure
class InputData(BaseModel):
    temp_c: float
    relative_humidity_pct: float
    co2_ppm: float
    moisture_pct: float

# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    # Prepare input in the same order used for training
    X = [[
        data.moisture_pct,
        data.temp_c,
        data.relative_humidity_pct,
        data.co2_ppm
    ]]
    
    # Get predicted ppb from model
    ppb = model.predict(X)[0]

    # Categorize risk level
    if ppb < 5:
        risk = "low"
    elif ppb < 10:
        risk = "medium"
    else:
        risk = "high"

    return {
        "predicted_ppb": round(ppb, 2),
        "predicted_risk": risk
    }
