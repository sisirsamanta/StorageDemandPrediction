import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

num_customers = 1000
num_months = 12

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Prepare data for LSTM model
X_train = train.iloc[:, :-1].values.reshape(-1, num_months, 1)
y_train = train.iloc[:, -1].values
X_test = test.iloc[:, :-1].values.reshape(-1, num_months, 1)
y_test = test.iloc[:, -1].values

# Define LSTM model
model = keras.Sequential([
    layers.LSTM(128, return_sequences=True, input_shape=(num_months, 1)),
    layers.LSTM(64, return_sequences=False),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), verbose=1)


X_test = test.iloc[:, :-1].values.reshape(-1, num_months, 1)
y_test = test.iloc[:, -1].values

y_pred = model.predict(X_test)

# Compare actual vs. predicted values
plt.plot(y_test, label='Actual Storage')
plt.plot(y_pred, label='Predicted Storage', linestyle='dashed')
plt.legend()

# Save the trained model
model.save('models/storage_prediction_model.h5')

print("Model training complete and saved successfully.")