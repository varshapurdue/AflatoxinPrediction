from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load your trained model
model = joblib.load("aflatoxin_model.pkl")

app = FastAPI()

# Define input schema
class InputData(BaseModel):
    temp_c: float
    relative_humidity_pct: float
    co2_ppm: float
    moisture_pct: float

@app.post("/predict")
def predict(data: InputData):
    X = [[
        data.moisture_pct,
        data.temp_c,
        data.relative_humidity_pct,
        data.co2_ppm
    ]]
    ppb = model.predict(X)[0]

    # Risk category
    if ppb < 5:
        risk = "low"
    elif ppb < 10:
        risk = "medium"
    else:
        risk = "high"

    return {"predicted_ppb": round(ppb, 2), "predicted_risk": risk}
