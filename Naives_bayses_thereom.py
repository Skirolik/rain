import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import joblib
import numpy as np


df= pd.read_csv("mod_rain_data.csv")

le=LabelEncoder()
df['Rain']=le.fit_transform(df['Rain'])

df['Rainfall_next_15min']=le.transform(df['Rainfall_next_15min'])
text_format=df['Rainfall_next_15min']

X = df[['Temperature (K)', 'Relative Humidity (%)', 'Pressure (hPa)', 'Wind speed (m/s)',  'Rain']]
y = df['Rainfall_next_15min']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


#min max scaler
# scaler=MinMaxScaler()
# X_train_scaled=scaler.fit_transform(X_train)
# X_test_scaled=scaler.fit_transform(X_test)

# Train naive Bayes Model
model= CategoricalNB()
model.fit(X_train,y_train)


joblib.dump(model,'naive_bayes_model.pkl')
joblib.dump(text_format,'label_encoder.pkl')

# joblib.dump(scaler,'min_max_scaler.pkl')

#Make predictions
y_pred=model.predict(X_test)
#evaluatecategoricalNB
accuracy=accuracy_score(y_test,y_pred)
conf_matrix=confusion_matrix(y_test,y_pred)
classification_report=classification_report(y_test,y_pred)


print(y_pred)
print(f"Accuracy:{accuracy:.2f}")
print("Confusin Matrix:")
print(conf_matrix)
print("Clasification Report")
print(classification_report)

# new_data = np.array([[303.91, 65, 1007, 2.62, 0]])
#
# op=model.predict(new_data)
# print('OP',op)