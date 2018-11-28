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
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def EvaluateFeatures(x,y,max_features,minimum_class_sum,criterion,max_estimators):
    ySum = np.sum(y,axis = 0)
    minClassSum = int(np.min(ySum))    
    x_rf = np.zeros([x.shape[0],max_features])
    feature_indicies = np.arange(x.shape[1],dtype=np.int32) 
    output_weights = np.zeros(x.shape[1])
    best_feature_indicies = []
    for iFtr in range(max_features):
        max_oob_score = 0.0
        iFtrBest = 0
        for iFtrNew in feature_indicies:
            if iFtrNew in best_feature_indicies:
                continue
            x_rf[:,len(best_feature_indicies)] = x[:,iFtrNew]
            rf = RandomForestClassifier(n_estimators=np.min([2*minClassSum,max_estimators]),oob_score=True)
            rf.fit(x_rf[:,0:len(best_feature_indicies)+1],y)
            if rf.oob_score_ > max_oob_score:
                max_oob_score = rf.oob_score_
                iFtrBest = iFtrNew
            output_weights[iFtrNew]+=rf.oob_score_
            for iFtrOld in best_feature_indicies:
                output_weights[iFtrOld]+=rf.oob_score_
        x_rf[:,len(best_feature_indicies)] = x[:,iFtrBest]                   
        best_feature_indicies.append(iFtrBest)
    output_weights/=np.sum(output_weights)
    return output_weights, np.array(best_feature_indicies)

np.random.seed(111)
maxFeatures = 'auto'
n_features = 10
n_samples = 500
x = np.random.uniform(-10.0,10.0,n_samples*n_features)
x = x.reshape([n_samples,n_features])
xMean = np.random.uniform(-3.0,3.0,n_features)
#xMean = np.zeros(n_features)
xMean = xMean.reshape([1,n_features])
cov = np.zeros([n_features,n_features],dtype=np.float32)
cov[0,0] = 5.0
cov[1,1] = 3.0
#covInv = np.linalg.inv(cov)
covInv = np.zeros([n_features,n_features],dtype=np.float32)
covInv[0:2,0:2]=np.linalg.inv(cov[0:2,0:2])
xDemean = np.subtract(x,xMean)
#
pTrue = np.exp(-0.5*np.sum(xDemean*np.matmul(xDemean,covInv),axis=1))
yRand = np.random.uniform(0.0,1.0,n_samples)
whrTrue = np.where(yRand <= pTrue)[0]
whrFalse = np.where(yRand > pTrue)[0]
whrTF = yRand <= pTrue
y = ['True' if whrTF[indx] else 'False' for indx in range(n_samples)]
yf = np.zeros(n_samples)
yf2 = np.zeros([n_samples,2])
yf[whrTrue] = 1.0
yf2[:,0] = 1.0-yf
yf2[:,1] = yf
plt.figure(1)
plt.clf()
plt.plot(x[whrTrue,0],x[whrTrue,1],'ro')
plt.plot(x[whrFalse,0],x[whrFalse,1],'bx')

#weights,features = EvaluateFeatures(x,yf2,6,8,'gini',50)
#print(weights)
#print(features)
#stop


print('Number of true values = ',len(whrTrue))

skf = StratifiedKFold(n_splits=5, random_state=1331, shuffle=False)
print(skf.get_n_splits(x, yf))
y_pred_all = np.zeros(n_samples)
y_pred_all_sk = np.zeros(n_samples)
for train_index, test_index in skf.split(x, yf):
    print("TRAIN:", train_index[:10], "TEST:", test_index[:10])
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = yf[train_index], yf[test_index]  
    rf = RandomForest(n_estimators=100,min_features_considered=maxFeatures,oob_score=True)
    rf.fit(x_train,y_train)
    print('classes = ',rf.classes)
    y_pred = rf.predict(x_test)
    y_pred_all[test_index] = y_pred[:,1]
    print(y_pred[0:10,1])
    print(y_pred[-10:,1])        
    print('avg test error = {}'.format(np.mean((y_pred[:,1]-y_test)**2)))
    print('oob error = {}'.format(rf.oob_error))
    
    rfsk = RandomForestClassifier(n_estimators=100,max_features=maxFeatures,oob_score=True)
    #rfsk = RandomForestClassifier(n_estimators=100,criterion='entropy')    
    rfsk.fit(x_train,y_train)
    print('classes sk= ',rfsk.classes_)
    y_pred = rfsk.predict_proba(x_test)
    y_pred_all_sk[test_index] = y_pred[:,1]
    print(y_pred[0:10,1])
    print(y_pred[-10:,1])        
    print('sklearn avg test error = {}'.format(np.mean((y_pred[:,1]-y_test)**2)))
    print('sk oob score = {}'.format(rfsk.oob_score_))
    print('sk feature importance = ')
    print(rfsk.feature_importances_ )
    print('-------------------------------')

    
fpr, tpr, _ = roc_curve(yf, y_pred_all)
roc_auc = auc(fpr, tpr)    
    
fprsk, tprsk, _ = roc_curve(yf, y_pred_all_sk)
roc_aucsk = auc(fprsk, tprsk)    
plt.figure(2)
plt.clf()
lw = 2
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot(fprsk, tprsk, color='g',lw=lw, label='SK ROC curve (area = %0.2f)' % roc_aucsk)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()