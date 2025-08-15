import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Sample_Data.csv")
print(df.head())

# 5-day moving average
df['MA_5'] = df['Values'].rolling(window=5).mean()

# Plot 1: Voltage + Moving Average + Trendline
plt.figure(figsize=(10, 6))
plt.plot(df['Timestamp'], df['Voltage'], label='Voltage', color='blue')
plt.plot(df['Timestamp'], df['MA_5'], label='5-day MA', color='red')

# Add trendline
x_numeric = np.arange(len(df))
coeffs = np.polyfit(x_numeric, df['Voltage'], 1)
trendline = np.polyval(coeffs, x_numeric)
plt.plot(df['Timestamp'], trendline, label='Trendline', color='green', linestyle='--')

plt.xlabel("Timestamp")
plt.ylabel("Voltage")
plt.legend()
plt.title("Voltage vs Timestamp with Moving Average & Trendline")
plt.savefig("static/plot_voltage.png")
plt.close()