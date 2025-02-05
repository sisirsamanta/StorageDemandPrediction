from fastapi import FastAPI
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import uvicorn


# Explicitly define Mean Squared Error (MSE) as a custom object
custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}

# Load the trained LSTM model
model = tf.keras.models.load_model('models/storage_prediction_model.h5', custom_objects=custom_objects)

# Load and preprocess data (ensure same scaler is used as in training)
df = pd.read_csv('data/productionData.csv')
#df['customer_id'] = df['customer_id'].astype(int)  # Ensure integer indexing
df_pivot = df.pivot(index='customer_id', columns='month', values='storage_used_tb').fillna(0)
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_pivot), columns=df_pivot.columns, index=df_pivot.index)

app = FastAPI()

@app.get("/predict/{customer_id}")
def predict(customer_id: int):
    if customer_id not in df_scaled.index:
        return {"error": "Customer ID not found"}
    
    # Prepare input for model (reshape to match LSTM input format)
    customer_data = df_scaled.loc[customer_id].values[:-1].reshape(1, -1, 1)
    predicted_storage = model.predict(customer_data)[0][0]
    
    # Reverse scale the prediction
    predicted_storage = scaler.inverse_transform([[0] * 11 + [predicted_storage]])[0][-1]
    
    return {"customer_id": customer_id, "predicted_storage_tb": predicted_storage}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)