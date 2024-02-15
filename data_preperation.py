import pandas as pd

# df=pd.read_csv("data.csv")
# print(df)
#
# df['Rainfall_mm_per_hr']=df['Rainfall (kg/m2)'] *4 # As data is every 15 min we multiply by 4
#
# print(df['Rainfall_mm_per_hr'])
#
# df.to_csv("modified_data.csv",index=False)

df= pd.read_csv("modified_data.csv")
print(df)

threshold = [0, 0.01,0.1, 1.5,  float("inf")]
labels = ['No rain',  'Low', 'Mid', 'Heavy']
print(len(threshold))
print(len(labels))

df['Rain_values']=pd.cut(df['Rainfall_mm_next_15'],bins=threshold,labels=labels,include_lowest=True)

df.to_csv("new_values.csv",index=False)

# threshold=[0,0.1,0.5,1.5,2.5,float("inf")]
# labels= ['No rain','Very low','Low','Mid','Heavy']
#
# df['Rain']=pd.cut(df['Rainfall (kg/m2)'],bins=threshold,labels=labels ,include_lowest=True)
#
# df= df.drop(columns=['Rainfall (kg/m2)'])
#
# df.to_csv("modified_data.csv",index=False)