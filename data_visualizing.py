import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
df= pd.read_csv('mod_data.csv')
#print(df)

df['UT time'] = df['UT time'].replace('24:00:00', '00:00')

# Convert to datetime with inferred format
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['UT time'], format='%d-%m-%Y %H:%M')
df.set_index('Datetime', inplace=True)

plt.figure(figsize=(12,6))
plt.plot(df['Temperature (K)'], label='Temperature')
plt.title('Temperature over time')
plt.xlabel('Time')
plt.ylabel('Temperature (K)')
plt.legend
plt.show()

numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

df_diff= df[numeric_columns].diff().dropna()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_diff)
scalled_diff_df=pd.DataFrame(scaled_data,columns=numeric_columns)


correlation_matrix = scalled_diff_df.corr()

plt.figure(figsize=(20, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix for diffrence')
plt.show()

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import CategoricalNB
# from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
# from sklearn.preprocessing import LabelEncoder,MinMaxScaler
# import joblib
#
# df= pd.read_csv("mod_rain_data.csv")
#
# le=LabelEncoder()
# df['Rain']=le.fit_transform(df['Rain'])
# test=df['Rain']
# print(test)
# df['Rainfall_next_15min']=le.transform(df['Rainfall_next_15min'])
#
# test2=le.inverse_transform(test)
# print('test2',test2)