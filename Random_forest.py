
#Data processing
import pandas as pd
import numpy as np

#Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV,train_test_split
from scipy.stats import randint
from sklearn.metrics import confusion_matrix,classification_report

#Tree Visulalization
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
import matplotlib.pyplot as plt

from sklearn.tree import plot_tree



df=pd.read_csv("mod_rain_data.csv")
head=df.info()

df['Rain_values']=df['Rain_values'].map({'No rain':0,'Low':1,'Mid':2,'Heavy':3})

print(df['Rain_values'].value_counts())

X = df[['Temperature (K)', 'Relative Humidity (%)', 'Pressure (hPa)', 'Wind speed (m/s)',  'Rainfall_mm_per_hr']]
y = df['Rain_values']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# param_dist={'n_estimators':randint(10,500),'max_depth':randint(1,50)}

rf=RandomForestClassifier(n_estimators=324,max_depth=27)

#to find best hyperparameters
# #
# rand_search=RandomizedSearchCV(rf,param_distributions=param_dist,n_iter=5,cv=20)
#
# rand_search.fit(X_train,y_train)
#
# best_rf=rand_search.best_estimator_
#
# print('Best hyperparameters:',rand_search.best_params_)
#
# rf=RandomForestClassifier(n_estimators=100)
rf.fit(X_train,y_train)

y_pred=rf.predict(X_test)
print('Predicted values',y_pred)
#
accuracy= accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
precision=precision_score(y_test,y_pred,average=None)
print('precision micro:',precision)
recall=recall_score(y_test,y_pred,average=None)
print('recall Score:',recall)

conf_matrix=confusion_matrix(y_test,y_pred)
print(conf_matrix)

class_report=classification_report(y_test,y_pred)
print(class_report)
#
# feature_importance=pd.Series(rf.feature_importances_,index=X_train.columns).sort_values(ascending=False)
#
# feature_importance.plot.bar()
# plt.show()
# #
# cm=confusion_matrix(y_test,X_test)
# confusion_matrix(confusion_matrix=cm).plot()




# Plot the specified decision tree within the Random Forest

# for i in range(5):
#     tree_index = i
#     plt.figure(figsize=(12, 8))
#     plot_tree(rf.estimators_[tree_index], feature_names=X_train.columns, filled=True, max_depth=2, impurity=True,
#               proportion=True)
#     plt.show()

