# StorageDemandPrediction

Goal: Predict future storage demand for customers based on historical usage patterns.
Input Data: Customer storage consumption history.
Output: Predicted storage demand (in GB/TB) for a future period.


# Set Up the Development Environment
Install Necessary Tools
You'll need Python and key ML libraries. Install them using:

pip install numpy pandas matplotlib scikit-learn seaborn tensorflow torch

Or, if using Jupyter Notebook:
pip install jupyterlab

# Project Structure

storage-demand-prediction/

│── data/                  # Raw & processed datasets

│── models/                # Trained ML models

│── src/                   # Source code

│── notebooks/             # Jupyter notebooks for analysis

│── scripts/               # Training and evaluation scripts

│── app/                   # API/Web App to expose predictions

│── requirements.txt       # List of dependencies

│── README.md


**Overview**  

This project predicts future storage demand for customers based on historical usage patterns. It involves data processing, training an LSTM deep learning model, deploying the model as a FastAPI service, and validating predictions.

Features

Data Simulation: Generates synthetic storage usage data.

Machine Learning Model: Uses an LSTM-based neural network for predictions.

REST API: FastAPI-based endpoints for making predictions.

Validation & Testing: Ensures accurate and reliable forecasts.

Interactive Swagger UI: For easy API testing.


Project Steps

1️⃣ Data Generation

We generate synthetic storage usage data for customers over a period of 12 months:

import numpy as np
import pandas as pd

np.random.seed(42)
data = {
    'customer_id': np.repeat(range(1, 1001), 12),
    'month': list(range(1, 13)) * 1000,
    'storage_used_tb': np.random.uniform(5, 50, 12000) + np.linspace(0, 20, 12000)
}

df = pd.DataFrame(data)
df.to_csv('data/storage_usage.csv', index=False)

2️⃣ Model Training

We train an LSTM model using TensorFlow/Keras on the generated dataset.

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Preprocessing data
scaler = MinMaxScaler()
df_pivot = df.pivot(index='customer_id', columns='month', values='storage_used_tb').fillna(0)
df_scaled = scaler.fit_transform(df_pivot)

# LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(12, 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(df_scaled.reshape(-1, 12, 1), df_scaled[:, -1], epochs=50)

# Save model & scaler
model.save('models/storage_prediction_model.h5')
import joblib
joblib.dump(scaler, 'models/scaler.pkl')


3️⃣ API Development

FastAPI serves the trained model for predictions.

from fastapi import FastAPI
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import uvicorn

app = FastAPI()

# Load trained model & scaler
model = tf.keras.models.load_model('models/storage_prediction_model.h5')
scaler = joblib.load('models/scaler.pkl')

# Load data for reference
df = pd.read_csv('data/storage_usage.csv')
df_pivot = df.pivot(index='customer_id', columns='month', values='storage_used_tb').fillna(0)
df_scaled = pd.DataFrame(scaler.transform(df_pivot), columns=df_pivot.columns, index=df_pivot.index)

@app.get("/predict/{customer_id}")
def predict(customer_id: int):
    if customer_id not in df_scaled.index:
        return {"error": "Customer ID not found"}
    
    customer_data = df_scaled.loc[customer_id].values[:-1].reshape(1, 12, 1)
    predicted_storage = model.predict(customer_data)[0][0]
    predicted_storage = scaler.inverse_transform([[0] * 11 + [predicted_storage]])[0][-1]
    
    return {"customer_id": customer_id, "predicted_storage_tb": predicted_storage}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


4️⃣ Testing the API
✅ Using Swagger UI

Open http://127.0.0.1:8000/docs

Click GET /predict/{customer_id}.

Click Try it out, enter a customer ID, and execute.

✅ Using Python
import requests
response = requests.get("http://127.0.0.1:8000/predict/1")
print(response.json())

✅ Using Postman

Open Postman.

Send a GET request to http://127.0.0.1:8000/predict/1.

Check the response.

5️⃣ Deployment

Run locally:
uvicorn main:app --host 0.0.0.0 --port 8000

For production:
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app

Contributing

Fork the repo.

Create a feature branch.

Commit changes and open a PR.

License

This project is licensed under the MIT License.
