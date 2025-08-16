import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

df = pd.read_csv(
    'C:/Users/navneet ranjan/OneDrive/Desktop/assignment/Sample_Data.csv'
)
print(df.head(10))
print("*" * 40)
print(df.info())
print('*' * 40)
df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True)
print(df['Timestamp'].dtypes)

# Force 'values' column to numeric, coercing invalid entries (e.g., header as data) to NaN
df['Values'] = pd.to_numeric(df['Values'], errors='coerce')

# Drop any rows with invalid data
df = df.dropna(subset=['Timestamp', 'Values'])

plt.figure(figsize=(12,6))
plt.plot(df['Timestamp'], df['Values'], label='Values')
df['values_MA5'] = df['Values'].rolling(window=5).mean()
plt.plot(df['Timestamp'], df['values_MA5'], label='5-Day Moving Avg', linestyle='--')
plt.title('Values Over Time with 5-Day Moving Average')
plt.xlabel('Timestamp')
plt.ylabel('values')
plt.legend()
plt.tight_layout()
plt.show()

# Find local peaks and lows
peaks, _ = find_peaks(df['Values'])
lows, _ = find_peaks(-df['Values'])
peaks_table = df.iloc[peaks][['Timestamp', 'Values']].reset_index(drop=True)
lows_table = df.iloc[lows][['Timestamp', 'Values']].reset_index(drop=True)

print("Local Peaks:")
print(peaks_table)
print("\nLocal Lows:")
print(lows_table)

# Tabulate instances where values < 20
below_20_table = df[df['Values'] < 20][['Timestamp', 'Values']].reset_index(drop=True)
print("\nValues below 20:")
print(below_20_table)

# Find timestamps where downward slope accelerates in each downward cycle
df['dV'] = df['Values'].diff()
df['ddV'] = df['dV'].diff()
downward_accel = df[(df['dV'] < 0) & (df['ddV'] < 0)]
print("\nTimestamps where downward slope accelerates:")
print(downward_accel[['Timestamp', 'Values', 'dV', 'ddV']])
