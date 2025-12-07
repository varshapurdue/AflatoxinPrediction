from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

model = joblib.load("aflatoxin_model.pkl")  # this should match the file you exported

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
    storage_day: int

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

    if ppb < 5:
        risk = "low"
    elif ppb < 20:
        risk = "medium"
    else:
        risk = "high"

    return {
        "predicted_ppb": round(ppb, 2),
        "predicted_risk": risk
