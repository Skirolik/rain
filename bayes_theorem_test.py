import pandas as pd
import numpy as np
import joblib


model=joblib.load('naive_bayes_model.pkl')



min_values=np.array([292.11,18.55,986.9,0.01,0])
max_vales=np.array([309.72,93.4,1006.21,11.39,4])

new_data = np.array([[303.91, 65, 1006, 2.62, 0]])

clipped_data=np.clip(new_data,min_values,max_vales)
print('clip',clipped_data)

op=model.predict(clipped_data)
print('OP',op[0])
