import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# 1. Read data into a DataFrame
df = pd.read_csv('C:/Users/navneet ranjan/OneDrive/Desktop/assignment/Sample_Data.csv', header=0, parse_dates=['Timestamp'])
print(df.head(10))


print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("*" * 40)

#converting timestamp data type from object to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%d/%m/%Y %H:%M")

#Boxplots
plt.figure(figsize=(6,4))
plt.boxplot(df['Values'])
plt.title("Boxplot of Values")
plt.xlabel("Values")
plt.show()


# Remove outliers using IQR method
Q1 = df['Values'].quantile(0.25)
Q3 = df['Values'].quantile(0.75)
IQR = Q3 - Q1

df_clean = df[(df['Values'] >= Q1 - 1.5*IQR) & (df['Values'] <= Q3 + 1.5*IQR)]

print("\nData after removing outliers:")
print(df_clean)

#Boxplots
plt.figure(figsize=(6,4))
plt.boxplot(df_clean['Values'])
plt.title("Boxplot of Values after removing ourliers")
plt.xlabel("Values")
plt.show()

min_val = df_clean['Values'].min()
max_val = df_clean['Values'].max()
avg_val = df_clean['Values'].mean()

print(f"\nMin Value: {min_val}")
print(f"Max Value: {max_val}")
print(f"Average Value: {avg_val}")


plt.figure(figsize=(6,4))
plt.hist(df_clean['Values'], bins=10, color='skyblue', edgecolor='black')
plt.title("Histogram of Values")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(df_clean['Timestamp'], bins=10, color='skyblue', edgecolor='black')
plt.title("Histogram of Values")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.xticks(rotation=270)
plt.show()


df_12th_onwards = df_clean.iloc[11:].reset_index(drop=True)
sampled_df = df_12th_onwards.iloc[::100, :].copy()
x_pos = np.arange(len(sampled_df))

colors = ['red' if val < avg_val else 'green' for val in sampled_df['Values']]

# Step 4: Plot with spacing
plt.figure(figsize=(14,6))
plt.bar(x_pos, sampled_df['Values'], color=colors, width=0.6)  # width < 1 adds spacing
plt.axhline(avg_val, color='blue', linestyle='--', label=f'Average ({avg_val:.2f})')
plt.title("Bar Chart from 12th Row Onwards (Sampled Every 100th Row with Gaps)")
plt.xlabel("Timestamp Index")
plt.ylabel("Values")

tick_interval = 10
plt.xticks(x_pos[::tick_interval], sampled_df['Timestamp'].dt.strftime('%Y-%m-%d')[::tick_interval], rotation=45)

plt.legend()
plt.tight_layout()
plt.show()