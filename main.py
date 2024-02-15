import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt


df= pd.read_csv('data.csv')
print(df)

df['UT time'] = df['UT time'].replace('24:00:00', '00:00')
# Assuming your data frame is named 'df'
# Ensure 'Datetime' is in the datetime format
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['UT time'], format='%d-%m-%Y %H:%M')
df.set_index('Datetime', inplace=True)

# Extract the 'Rainfall (kg/m2)' column
rainfall_data = df['Rainfall (kg/m2)']

# Apply seasonal decomposition
decomposition = seasonal_decompose(rainfall_data, model='additive', period=365)  # Assuming yearly seasonality

# Plot the decomposition components
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(rainfall_data, label='Original')
plt.legend(loc='upper left')
plt.title('Original Rainfall Data')

plt.subplot(4, 1, 2)
plt.plot(decomposition.trend, label='Trend')
plt.legend(loc='upper left')
plt.title('Trend Component')

plt.subplot(4, 1, 3)
plt.plot(decomposition.seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.title('Seasonal Component')

plt.subplot(4, 1, 4)
plt.plot(decomposition.resid, label='Residuals')
plt.legend(loc='upper left')
plt.title('Residuals')

plt.tight_layout()
plt.show()
