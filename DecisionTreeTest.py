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
import matplotlib.pyplot as plt
np.random.seed(111)
n_features = 2
n_samples = 500
x = np.random.uniform(-10.0,10.0,n_samples*n_features)
x = x.reshape([n_samples,n_features])
xMean = np.random.uniform(-3.0,3.0,n_features)
#xMean = np.zeros(n_features)
xMean = xMean.reshape([1,n_features])
cov = np.zeros([n_features,n_features],dtype=np.float32)
cov[0,0] = 4.0
cov[1,1] = 2.0
covInv = np.linalg.inv(cov)
xDemean = np.subtract(x,xMean)
#
pTrue = np.exp(-0.5*np.sum(xDemean*np.matmul(xDemean,covInv),axis=1))
yRand = np.random.uniform(0.0,1.0,n_samples)
whrTrue = np.where(yRand <= pTrue)[0]
whrFalse = np.where(yRand > pTrue)[0]
whrTF = yRand <= pTrue
y = ['True' if whrTF[indx] else 'False' for indx in range(n_samples)]
yf = np.zeros(n_samples)
yf[whrTrue] = 1.0
plt.figure(1)
plt.clf()
plt.plot(x[whrTrue,0],x[whrTrue,1],'ro')
plt.plot(x[whrFalse,0],x[whrFalse,1],'bx')

skf = StratifiedKFold(n_splits=5, random_state=1331, shuffle=False)
print(skf.get_n_splits(x, yf))
for train_index, test_index in skf.split(x, yf):
    print("TRAIN:", train_index[:10], "TEST:", test_index[:10])
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = yf[train_index], yf[test_index]  
    rfsk = RandomForestClassifier(n_estimators=100)
    #rfsk = RandomForestClassifier(n_estimators=100,criterion='entropy')    
    rfsk.fit(x_train,y_train)
    y_pred = rfsk.predict_proba(x_test)
    print(y_pred[0:10,1])
    print(y_pred[-10:,1])        
    print('sklearn avg test error = {}'.format(np.mean((y_pred[:,1]-y_test)**2)))
    print('-------------------------------')