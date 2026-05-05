from fastapi import FastAPI
import joblib
import numpy as np

# Initialize app
app = FastAPI()

# Load model ONCE
model = joblib.load("student_model.joblib")

@app.get("/")
def home():
    return {"message": "ML API running 🚀"}

@app.post("/predict")
def predict(hours_studied: float, attendance: float, sleep_hours: float):
    data = np.array([[hours_studied, attendance, sleep_hours]])
    prediction = model.predict(data)[0]

    return {
        "prediction": int(prediction)
    }