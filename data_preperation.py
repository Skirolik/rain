import pandas as pd

df=pd.read_csv("data.csv")
# print(df)

threshold=[0,0.1,0.5,1.5,2.5,float("inf")]
labels= ['No rain','Very low','Low','Mid','Heavy']

df['Rain']=pd.cut(df['Rainfall (kg/m2)'],bins=threshold,labels=labels ,include_lowest=True)

df= df.drop(columns=['Rainfall (kg/m2)'])

df.to_csv("modified_data.csv",index=False)