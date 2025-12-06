from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

model = joblib.load("aflatoxin_model.pkl")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    temp_c: float
    relative_humidity_pct: float
    co2_ppm: float
    moisture_pct: float

@app.post("/predict")
def predict(data: InputData):
    # Add dummy storage_day to match training shape
    storage_day = 30  # Can be dynamic if needed

    X = [[
        data.moisture_pct,
        data.temp_c,
        data.relative_humidity_pct,
        data.co2_ppm,
        storage_day  # Add 5th feature
    ]]

    ppb = model.predict(X)[0]

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
