import pandas as pd

df= pd.read_csv('modified_data.csv')
#print(df)

df['UT time'] = df['UT time'].replace('24:00:00', '00:00')

# Convert to datetime with inferred format
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['UT time'], format='%d-%m-%Y %H:%M')
df.set_index('Datetime', inplace=True)

df['Rainfall_next_15min']=df['Rain'].shift(-1)

df=df.dropna()

df.to_csv("mod_rain_data.csv",index=False)