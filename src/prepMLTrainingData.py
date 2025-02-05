from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

df = pd.read_csv('data/storage_usage.csv')

# Pivot table to structure data for ML model

df_pivot = df.pivot(index='customer_id', columns='month', values='storage_used_tb').fillna(0)

# Normalize data
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_pivot), columns=df_pivot.columns, index=df_pivot.index)

# Split data into training and testing sets
train, test = train_test_split(df_scaled, test_size=0.2, random_state=42)

# Save datasets
train.to_csv('data/train.csv', index=True)
test.to_csv('data/test.csv', index=True)

print("Training and testing datasets are ready.")
