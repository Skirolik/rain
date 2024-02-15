import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder,MinMaxScaler


df= pd.read_csv("mod_rain_data.csv")
head=df.info()


le=LabelEncoder()


print(df['Rain_values'].isnull().sum())


df['Rain_values']=le.fit_transform(df['Rain_values'])
print("Original Values -> Encoded Labels:")
for original_value, encoded_label in zip(le.classes_, range(len(le.classes_))):
    print(f"{original_value} -> {encoded_label}")


print(df['Rain_values'].value_counts())

X = df[['Temperature (K)', 'Relative Humidity (%)', 'Pressure (hPa)', 'Wind speed (m/s)',  'Rainfall_mm_per_hr','Short-wave irradiation (Wh/m2)']]
y = df['Rain_values']

##Split train test
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

gnb= GaussianNB()
gnb.fit(X,y)

plt.figure(figsize=(12, 8))
for i, feature in enumerate(X.columns):
    plt.subplot(2, 3, i + 1)  # Adjust the subplot grid based on the number of features

    # Plot the Gaussian distribution for each class
    for class_label in np.unique(y):
        subset = X[y == class_label][feature]
        sns.histplot(subset, kde=True, label=f'Class {class_label}', color=f'C{class_label}', common_norm=False)

        # Plot the Gaussian fit
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, gnb.theta_[class_label, i])
        p /= p.sum()  # Normalize the distribution
        plt.plot(x, p, color=f'C{class_label}', linestyle='--', linewidth=2)

    plt.title(f'Gaussian Distribution for {feature}')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()

plt.tight_layout()
plt.show()





