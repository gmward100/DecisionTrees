# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 10:25:39 2018

@author: gward
"""

import numpy as np
from RandomForest import RandomForest
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

iris_data = pd.read_csv('Iris.csv', keep_default_na=False)
iris_data['species'] = iris_data['species'].str.replace('setosa','1')
iris_data['species'] = iris_data['species'].str.replace('versicolor','0')
iris_data['species'] = iris_data['species'].str.replace('virginica','0')

#x = iris_data.as_matrix(columns=['sepal_length','sepal_width','petal_length','petal_width'])
x = np.array(iris_data[iris_data.columns[0:4]].values,dtype=np.float32)
y = np.array(iris_data['species'].astype(np.float32))

nRandom = 10
x_rand = np.reshape(np.random.uniform(-2.0,2.0,nRandom*x.shape[0]),[x.shape[0], nRandom])
x_new = np.zeros([x.shape[0],nRandom+x.shape[1]])
x_new[:,0:4] = x[:,:]
x_new[:,4:] = x_rand[:,:]
x = x_new

print(y)
print(x)
print(x.shape)

skf = StratifiedKFold(n_splits=5, random_state=101, shuffle=False)
print(skf.get_n_splits(x, y))
np.random.seed(1001)
for train_index, test_index in skf.split(x, y):
    print("TRAIN:", train_index[:10], "TEST:", test_index[:10])
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]  
    rf = RandomForest(n_estimators=100)
    #rf = RandomForest(n_estimators=100,criterion='entropy')
    rf.fit(x_train,y_train)
    y_pred = rf.predict(x_test)

    print('prediction = ')
    print(y_pred[0:10])
    print(y_pred[-10:])    
    print('truth = ')
    print(y_test[0:10])
    print(y_test[-10:])
    print('Avg test error = {}'.format(np.mean((y_pred-y_test)**2)))
    
    rfsk = RandomForestClassifier(n_estimators=100)
    #rfsk = RandomForestClassifier(n_estimators=100,criterion='entropy')    
    rfsk.fit(x_train,y_train)
    y_pred = rfsk.predict_proba(x_test)
    print(y_pred[0:10,1])
    print(y_pred[-10:,1])        
    print('sklearn avg test error = {}'.format(np.mean((y_pred[:,1]-y_test)**2)))
    print('-------------------------------')