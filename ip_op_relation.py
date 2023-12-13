import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("mod_data.csv")

plt.scatter(df['Temperature (K)'], df['Rainfall_next_15min'])
plt.title('Temperature vs. Rainfall_next_15min')
plt.xlabel('Temperature (K)')
plt.ylabel('Rainfall_next_15min')
plt.show()

# Scatter plot for Humidity vs. Rainfall_next_15min
plt.scatter(df['Relative Humidity (%)'], df['Rainfall_next_15min'])
plt.title('Humidity vs. Rainfall_next_15min')
plt.xlabel('Relative Humidity (%)')
plt.ylabel('Rainfall_next_15min')
plt.show()


plt.scatter(df['Pressure (hPa)'], df['Rainfall_next_15min'])
plt.title('Pressure (hPa) vs. Rainfall_next_15min')
plt.xlabel('Pressure (hPa)')
plt.ylabel('Rainfall_next_15min')
plt.show()


plt.scatter(df['Wind speed (m/s)'], df['Rainfall_next_15min'])
plt.title('Wind speed (m/s) vs. Rainfall_next_15min')
plt.xlabel('Wind speed (m/s)')
plt.ylabel('Rainfall_next_15min')
plt.show()

plt.scatter(df['Wind direction'], df['Rainfall_next_15min'])
plt.title('Wind direction vs. Rainfall_next_15min')
plt.xlabel('Wind direction')
plt.ylabel('Rainfall_next_15min')
plt.show()


# plt.scatter(df['Rain'], df['Rainfall_next_15min'])
# plt.title('Rain vs. Rainfall_next_15min')
# plt.xlabel('Rain')
# plt.ylabel('Rainfall_next_15min')
# plt.show()
# # ...

# To display all plots together, you can use subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))

# Temperature vs. Rainfall_next_15min
axes[0, 0].scatter(df['Temperature (K)'], df['Rainfall_next_15min'])
axes[0, 0].set_title('Temperature vs. Rainfall_next_15min')
# axes[0, 0].set_xlabel('Temperature (K)')
# axes[0, 0].set_ylabel('Rainfall_next_15min')

# Humidity vs. Rainfall_next_15min
axes[0, 1].scatter(df['Relative Humidity (%)'], df['Rainfall_next_15min'])
axes[0, 1].set_title('Humidity vs. Rainfall_next_15min')
# axes[0, 1].set_xlabel('Relative Humidity (%)')
# axes[0, 1].set_ylabel('Rainfall_next_15min')


axes[0, 2].scatter(df['Pressure (hPa)'], df['Rainfall_next_15min'])
axes[0, 2].set_title('Pressure (hPa) vs. Rainfall_next_15min')
# axes[0, 2].set_xlabel('Pressure (hPa)')
# axes[0, 2].set_ylabel('Rainfall_next_15min')


axes[1, 0].scatter(df['Wind speed (m/s)'], df['Rainfall_next_15min'])
axes[1, 0].set_title('Wind speed (m/s)) vs. Rainfall_next_15min')
# axes[1, 0].set_xlabel('Wind speed (m/s)')
# axes[1, 0].set_ylabel('Rainfall_next_15min')

axes[1, 1].scatter(df['Wind direction'], df['Rainfall_next_15min'])
axes[1, 1].set_title('Wind direction vs. Rainfall_next_15min')
# axes[1, 1].set_xlabel('Wind direction')
# axes[1, 1].set_ylabel('Rainfall_next_15min')


# axes[1, 2].scatter(df['Rain'], df['Rainfall_next_15min'])
# axes[1, 2].set_title('Rain vs. Rainfall_next_15min')
# # axes[1, 2].set_xlabel('Rain')
# # axes[1, 2].set_ylabel('Rainfall_next_15min')

# Add more subplots for other variables if needed

plt.tight_layout()
plt.show()