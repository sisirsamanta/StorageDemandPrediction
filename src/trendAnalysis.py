import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('data/storage_usage.csv')

# Plotting storage usage trend
sns.lineplot(data=df, x='month', y='storage_used_tb', hue='customer_id', legend=None)
plt.title('Customer Storage Usage Over Time')
plt.xlabel('Month')
plt.ylabel('Storage Used (TB)')
plt.show()
