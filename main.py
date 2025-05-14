from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor

MODEL_PATH = "wait_time_model.pkl" # Have to store: Unique for each queue
INITIAL_WAIT_TIME = 10.0 # First show the hard coded one, then use the first actual wait time to train the model

class PredictRequest(BaseModel):
    hour: int
    weekday: int
    queue_length: int

class TrainRequest(PredictRequest):
    actual_wait_time: float

app = FastAPI()

def load_model():
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        return model
    else:
        # Initialize a new RandomForest model and train it with a dummy sample
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        X = np.array([[10, 1, 20]])
        y = np.array([INITIAL_WAIT_TIME])
        model.fit(X, y)
        joblib.dump(model, MODEL_PATH)
        return model

@app.post("/predict")
def predict(data: PredictRequest):
    model = load_model()
    X = np.array([[data.hour, data.weekday, data.queue_length]])
    prediction = model.predict(X)[0]
    return {"predicted_wait_time": round(prediction, 2)}

@app.post("/train_and_predict")
def train_and_predict(data: TrainRequest):
    model = load_model()
    X_new = np.array([[data.hour, data.weekday, data.queue_length]])
    y_new = np.array([data.actual_wait_time])

    # Load existing training data or initialize new arrays
    if os.path.exists("train_data.npz"):
        data_file = np.load("train_data.npz") # Have to store: Unique for each queue
        X_train = data_file["X"]
        y_train = data_file["y"]
    else:
        X_train = np.array([[10, 1, 20]])
        y_train = np.array([INITIAL_WAIT_TIME])

    # Append new data
    X_train = np.vstack([X_train, X_new])
    y_train = np.append(y_train, y_new)

    # Retrain the model with updated data
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model and training data
    joblib.dump(model, MODEL_PATH)
    np.savez("train_data.npz", X=X_train, y=y_train)

    prediction = model.predict(X_new)[0]
    return {"predicted_wait_time": round(prediction, 2)}
