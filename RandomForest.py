# -*- coding: utf-8 -*-
"""
Created on Wed Nov  13 09:43:52 2018

@author: gward
"""

import numpy as np
import scipy as sp
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
# Random Forest Classifier Algorithm
class RandomForest:
    
    def __init__(self,
                n_estimators=10,
                criterion='gini',
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_features_considered='auto',
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                bootstrap=True,
                oob_score=False,
                n_jobs=None,
                random_state=None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_features_considered = min_features_considered
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.tree_base_node_list = []
        self.classes = []  
        self.oob_error = 0.0
        
    class RFTreeNode:
        def __init__(self):
            self.split_feature_index = -1
            self.split_feature_value = None
            self.less_than_node = None
            self.greater_than_node = None
            self.prediction = None
            self.depth = 0
            
        def grow_tree(self,x,y,min_features_considered,criterion,min_samples_leaf,min_samples_split,max_depth,depth,min_impurity_decrease,n_samples_total,skrfparams,sk_rf_prev_feature_weights,sk_rf_weight_fade):
            
            self.depth = depth+1
            if max_depth is not None:
                if self.depth == max_depth:
                    self.prediction = np.mean(y,axis=0)
                    return
                
            ySum = np.sum(y,axis = 0)
            for iclass in range(y.shape[1]):
                if ySum[iclass] == y.shape[0]:
                    self.prediction = y[0,:]
                    return
            
            if x.shape[0] < min_samples_split or x.shape[0] < 2*min_samples_leaf:
                self.prediction = np.mean(y)
                return
            feature_indicies = np.arange(x.shape[1],dtype=np.int32)
            #np.random.shuffle(feature_indicies)
            #max_features_considered = np.max(np.array([min_features_considered,x.shape[1]+1-int(np.floor(0.5*np.log2(x.shape[0])+0.5*x.shape[0]))]))
            max_features_considered = np.min([x.shape[1],np.min([int(np.min(ySum)), int(np.ceil(np.log2(x.shape[0])))])])
            n_sk_rf = int(np.ceil(float(x.shape[1])/float(max_features_considered)))
            sk_rf_feature_weights = np.zeros(x.shape[1])
            sk_rf_feature_counts = np.zeros(x.shape[1])
            for irf in range(n_sk_rf):
                rf_features = np.random.choice(feature_indicies,size = max_features_considered)
                x_rf = x[:,rf_features]
                #rf = RandomForestClassifier(n_estimators=skrfparams['n_estimators'],oob_score=True)
                rf = RandomForestClassifier(n_estimators=skrfparams['n_estimators'])                
                rf.fit(x_rf,y)
                for iftr in range(rf_features.shape[0]):
                   # sk_rf_feature_weights[rf_features[iftr]]+=rf.oob_score_*rf.feature_importances_[iftr]
                    sk_rf_feature_weights[rf_features[iftr]]+=rf.feature_importances_[iftr]                    
                    sk_rf_feature_counts[rf_features[iftr]]+=1.0
            sk_rf_feature_counts = np.clip(sk_rf_feature_counts,1.0,float(n_sk_rf+1))
            sk_rf_feature_weights/=sk_rf_feature_counts
#            rf = RandomForestClassifier(n_estimators=skrfparams['n_estimators'])        
#            rf.fit(x,y)            
            sk_rf_feature_weights_sum = np.sum(sk_rf_feature_weights)
            if sk_rf_feature_weights_sum == 0.0:
                sk_rf_feature_weights = sk_rf_prev_feature_weights.copy()
            else:
                sk_rf_feature_weights/=sk_rf_feature_weights_sum
                sk_rf_feature_weights=(1.0-sk_rf_weight_fade)*sk_rf_feature_weights+sk_rf_weight_fade*sk_rf_prev_feature_weights
                sk_rf_feature_weights/=np.sum(sk_rf_feature_weights)
                
            sk_argsort = np.argsort(sk_rf_feature_weights)
            feature_indicies = feature_indicies[sk_argsort]
            min_impurity = float(x.shape[0]+2)
            iFtrCount = 0
            feature_indicies = np.array([np.random.choice(feature_indicies,p=sk_rf_feature_weights)])
            for iFtr in feature_indicies:
                xUniqueSorted, reverseIndex, counts = np.unique(x[:,iFtr],return_inverse=True,return_counts=True)
                if xUniqueSorted.shape[0] == 1:
                    continue
                yUniqueSum = np.zeros([xUniqueSorted.shape[0],y.shape[1]])
                for iy in range(y.shape[0]):
                    yUniqueSum[reverseIndex[iy],:]+=y[iy,:]
                #yUniqueSum[reverseIndex]+=y
                leftCumulativeCount = np.cumsum(counts)
                rightCumulativeCount = np.cumsum(counts[::-1])
                pLeft = np.cumsum(yUniqueSum,axis=0)
                pRight = np.cumsum(yUniqueSum[::-1,:],axis=0)                
                for iclass in range(y.shape[1]):
                    pLeft[:,iclass]/=leftCumulativeCount
                    pRight[:,iclass]/=rightCumulativeCount
                left_sample_start = np.argmax(leftCumulativeCount >= min_samples_leaf)                    
                right_sample_stop = rightCumulativeCount.shape[0]-1-np.argmax(rightCumulativeCount >= min_samples_leaf)
                pRight = pRight[::-1,:]
                rightCumulativeCount = rightCumulativeCount[::-1]
                if left_sample_start >= right_sample_stop:
                    continue
                if criterion == 'gini':
                    impurity_zero = 1.0-np.sum(pLeft[-1,:]**2)
                    impurity = leftCumulativeCount[:-1]*(1.0-np.sum(pLeft[:-1,:]**2,axis=1))+rightCumulativeCount[1:]*(1.0-np.sum(pRight[1:,:]**2,axis=1))
                elif criterion == 'entropy':
                    impurity_zero = np.sum(sp.special.entr(pLeft[-1,:]))          
                    impurity = leftCumulativeCount[:-1]*np.sum(sp.special.entr(pLeft[:-1,:]),axis=1)+rightCumulativeCount[1:]*np.sum(sp.special.entr(pRight[1:,:]),axis=1)
                else:
                    print('{} is not a supported impurity function'.format(criterion))
                    return                        
                indxArgMin = np.argmin(impurity[left_sample_start:right_sample_stop])+left_sample_start
                impurity_decrease = (float(rightCumulativeCount[0])/float(n_samples_total))*(impurity_zero - impurity[indxArgMin]/float(rightCumulativeCount[0]))
                if impurity_decrease < min_impurity_decrease:
                    continue
                if impurity[indxArgMin] < min_impurity:
                    self.split_feature_value = 0.5*(xUniqueSorted[indxArgMin]+xUniqueSorted[indxArgMin+1])
                    self.split_feature_index = iFtr
                    min_impurity = impurity[indxArgMin]
                iFtrCount+=1
                if iFtrCount >= min_features_considered:
                    break
            if iFtrCount == 0:
                self.prediction = np.mean(y,axis=0)
                return                    
            min_impurity_argsort=x[:,self.split_feature_index].argsort()
            x=x[min_impurity_argsort,:]
            y=y[min_impurity_argsort,:]
            min_impurityi_index = np.argmax(x[:,self.split_feature_index]>self.split_feature_value)
            self.less_than_node = RandomForest.RFTreeNode()
            self.less_than_node.grow_tree(x[:min_impurityi_index,:],y[:min_impurityi_index],min_features_considered,criterion,min_samples_leaf,min_samples_split,max_depth,self.depth,min_impurity_decrease,n_samples_total,skrfparams,sk_rf_feature_weights,sk_rf_weight_fade)
            self.greater_than_node = RandomForest.RFTreeNode()
            self.greater_than_node.grow_tree(x[min_impurityi_index:,:],y[min_impurityi_index:],min_features_considered,criterion,min_samples_leaf,min_samples_split,max_depth,self.depth,min_impurity_decrease,n_samples_total,skrfparams,sk_rf_feature_weights,sk_rf_weight_fade)
            
        def predict(self,x):
            #print(self.depth)
            if self.prediction is not None:
                return self.prediction
            else:
                if x[self.split_feature_index] > self.split_feature_value:
                    return self.greater_than_node.predict(x)
                else:
                    return self.less_than_node.predict(x)
        
    def fit(self,x,y,skrfparams=None):      
        if skrfparams is None:
            skrfparams = {'n_estimators':20}
        if isinstance(x,(np.ndarray)) == False:
            raise Exception('x must be a numpy ndarray')    
        if y.shape[0] != x.shape[0]:
            raise Exception('sample size doesnt match for x and y')              
        if isinstance(y,(list)):
            self.classes = list(set(y))
            y_encoded = np.zeros([x.shape[0], len(self.classes)])
            #probably a way to vectorize this....
            for iy in range(len(y)):
                for iclass in self.classes:
                    if self.classes[iclass] == y[iy]:
                        y_encoded[iy,iclass] = 1.0
                        break
        elif isinstance(y,(np.ndarray)):
            if len(y.shape) == 1:
                self.classes = np.unique(y)
                y_encoded = np.zeros([x.shape[0], len(self.classes)])
                #probably a way to vectorize this....
                for iy in range(len(y)):
                    for iclass in range(self.classes.shape[0]):
                        if self.classes[iclass] == y[iy]:
                            y_encoded[iy,iclass] = 1.0
                            break                        
            elif y.shape[0] == x.shape[0] and y.shape[1] > 1:
                self.classes = np.arange(y.shape[1],dtype=np.int32)                
                y_encoded = y
                y_sum = np.sum(y,axis = 1)
                if np.max(y_sum) != 1 or np.min(y_sum) != 1:
                    raise Exception('y should be of size n_samples x n_classes encoded by zeros and ones')    
            else:
                raise Exception('sample size doesnt match for x and y')   
        else:
            raise Exception('y must be a list or numpy ndarray')      
            
        min_features_considered = self.min_features_considered
        if type(self.min_features_considered) == str:
            min_features_considered = np.int32(np.ceil(np.sqrt(x.shape[1])))
        if min_features_considered > x.shape[1]:
            min_features_considered = x.shape[1]
        np.random.seed(self.random_state)
        oob_pred = 0
        oob_counts = 0
        sample_range = 0
        if self.oob_score == True:
            oob_pred = np.zeros([x.shape[0],y_encoded.shape[1]],dtype=np.float32)
            oob_counts = np.zeros(x.shape[0],dtype=np.float32)    
            sample_range = np.arange(x.shape[0],dtype=np.int32)
            
        sk_feature_weight_fade = 0.25
        for iEstimator in range(self.n_estimators):
            self.tree_base_node_list.append(self.RFTreeNode())
            if self.bootstrap == True:
                #bootstrapIndx = np.random.choice(x.shape[0],x.shape[0])
                bootstrapIndx = np.random.randint(0,x.shape[0],size=x.shape[0])
                xBootstrap = x[bootstrapIndx,:]
                yBootstrap = y_encoded[bootstrapIndx,:]
                #xBootstrap, yBootstrap = resample(x,y_encoded)
                sk_feature_weights = np.zeros(x.shape[1])+1.0/float(x.shape[1])
                self.tree_base_node_list[iEstimator].grow_tree(xBootstrap,yBootstrap,min_features_considered,self.criterion,self.min_samples_leaf,self.min_samples_split,self.max_depth,0,self.min_impurity_decrease,x.shape[0],skrfparams,sk_feature_weights,sk_feature_weight_fade)
                if self.oob_score == True:
                    oob_indicies = np.where(np.isin(sample_range,bootstrapIndx) == False)[0]
                    if len(oob_indicies) > 0:
                        oob_counts[oob_indicies] += 1.0
                        x_oob = x[oob_indicies,:]
                        for indx in range(x_oob.shape[0]):
                            oob_pred[oob_indicies[indx],:]+=self.tree_base_node_list[iEstimator].predict(x_oob[indx,:])         
            else:
                sk_feature_weights = np.zeros(x.shape[1])+1.0/float(x.shape[1])                
                self.tree_base_node_list[iEstimator].grow_tree(x.copy(),y_encoded.copy(),min_features_considered,self.criterion,self.min_samples_leaf,self.min_samples_split,self.max_depth,0,self.min_impurity_decrease,x.shape[0],skrfparams,sk_feature_weights,sk_feature_weight_fade)
        if self.oob_score == True:
            oob_indicies = np.where(oob_counts > 0)[0]
            if len(oob_indicies) > 0:
                oob_pred = oob_pred[oob_indicies,:]
                oob_counts = oob_counts[oob_indicies]
                oob_y_encoded = y_encoded[oob_indicies,:]
                for iclass in range(y_encoded.shape[1]):
                    oob_pred[:,iclass]/=oob_counts
                self.oob_error=np.mean((oob_y_encoded-oob_pred)**2)
            else:
                print('No samples out of bag')
                
    def predict(self,x):  
       # print('predict')
        if len(x.shape) == 1:
            prediction = np.zeros(len(self.classes))
            for iEstimator in range(self.n_estimators):
                #print('Estimator {}'.format(iEstimator))                
                prediction+=self.tree_base_node_list[iEstimator].predict(x)
            return prediction/float(self.n_estimators)
        else:
            prediction = np.zeros([x.shape[0],len(self.classes)])
            for iEstimator in range(self.n_estimators):
                #print('Estimator {}'.format(iEstimator))
                for indx in range(x.shape[0]):
                    prediction[indx,:]+=self.tree_base_node_list[iEstimator].predict(x[indx,:])
            return prediction/float(self.n_estimators)