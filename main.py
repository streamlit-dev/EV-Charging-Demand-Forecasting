import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 1500
dates = pd.date_range(start='2024-01-01', periods=n, freq='h')

df = pd.DataFrame({
    'timestamp': dates,
    'temperature': np.random.normal(28, 5, n).round(1),
    'traffic_flow': np.random.randint(20, 200, n),
    'charging_price': np.random.choice([12, 15, 18, 22], n),
    'day_of_week': dates.day_name(),
    'hour': dates.hour,
    'station_id': np.random.choice(['ST01','ST02','ST03','ST04'], n)
})

df['energy_consumed_kWh'] = (
    df['traffic_flow'] * 0.4 + 
    (30 - abs(df['temperature'] - 28)) * 2 +
    (22 - df['charging_price']) * 3 +
    np.random.normal(0, 5, n) +
    np.where(df['hour'].isin([8,9,18,19,20]), 20, 0)
).clip(10, 250).round(1)

print("Dataset ready!")
print(df.head())
# Average demand by hour
hourly = df.groupby('hour')['energy_consumed_kWh'].mean()

plt.bar(hourly.index, hourly.values)
plt.title("Peak Hour Analysis - EV Demand")
plt.xlabel("Hour of Day")
plt.ylabel("Avg Energy (kWh)")
plt.savefig("peak_hour.png")
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df[['temperature', 'traffic_flow', 'charging_price', 'hour']]
y = df['energy_consumed_kWh']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print(f"Model Accuracy: {model.score(X_test, y_test)*100:.2f}%")