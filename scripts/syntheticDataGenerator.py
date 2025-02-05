import numpy as np
import pandas as pd

# Simulating storage consumption for 1000 customers over 12 months
num_customers = 1000
num_months = 12
np.random.seed(42)

data = {
    'customer_id': np.repeat(range(1, num_customers + 1), num_months),
    'month': list(range(1, num_months + 1)) * num_customers,
    'storage_used_tb': np.random.uniform(5, 15, num_customers * num_months) + np.linspace(0, 50, num_customers * num_months)
}

df = pd.DataFrame(data)
df.to_csv('data/storage_usage.csv', index=False)
print(df.head())