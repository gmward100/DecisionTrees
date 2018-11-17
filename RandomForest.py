# -*- coding: utf-8 -*-
"""
Created on Wed Nov  13 09:43:52 2018

@author: gward
"""

import numpy as np
import scipy as sp
# Random Forest Classifier Algorithm
class RandomForest:
    
    def __init__(self,
                n_estimators=10,
                criterion='gini',
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='auto',
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
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.tree_base_node = []
        
    class RFTreeNode:
        def __init__(self):
            self.split_feature_index = -1
            self.split_feature_value = None
            self.less_than_node = None
            self.greater_than_node = None
            self.prediction_value = None
            self.depth = 0
            
        def grow_tree(self,x,y,max_features,criterion,min_samples_leaf,min_samples_split,max_depth,depth):
            
            self.depth = depth+1
            if max_depth is not None:
                if self.depth == max_depth:
                    self.prediction_value = np.mean(y)
                    return
                
            ySum = np.sum(y)
            if ySum == 0 or ySum == y.shape[0]:
                self.prediction_value = y[0]
                return
            
            if x.shape[0] < min_samples_split or x.shape[0] < 2*min_samples_leaf:
                self.prediction_value = np.mean(y)
                return
            feature_indicies = np.arange(x.shape[1])
            np.random.shuffle(feature_indicies)
            #feature_indicies = np.random.randint(0,x.shape[1],size=max_features)
            min_cost = float(x.shape[0]+2)
            iFtrCount = 0
            for iFtr in feature_indicies:
                xUniqueSorted, reverseIndex, counts = np.unique(x[:,iFtr],return_inverse=True,return_counts=True)
                if xUniqueSorted.shape[0] == 1:
                    continue
                yUniqueSum = np.zeros(xUniqueSorted.shape[0])
                for iy in range(y.shape[0]):
                    yUniqueSum[reverseIndex[iy]]+=y[iy]
                #yUniqueSum[reverseIndex]+=y
                leftCumulativeCount = np.cumsum(counts)
                rightCumulativeCount = np.cumsum(counts[::-1])
                pLeft = np.cumsum(yUniqueSum)/leftCumulativeCount
                pRight = np.cumsum(yUniqueSum[::-1])/rightCumulativeCount
                left_sample_start = np.argmax(leftCumulativeCount >= min_samples_leaf)                    
                right_sample_stop = rightCumulativeCount.shape[0]-1-np.argmax(rightCumulativeCount >= min_samples_leaf)
                pRight = pRight[::-1]
                rightCumulativeCount = rightCumulativeCount[::-1]
                if left_sample_start >= right_sample_stop:
                    continue
                if criterion == 'gini':
                    costFunction = 2.0*(leftCumulativeCount*pLeft*(1.0-pLeft)+rightCumulativeCount*pRight*(1.0-pRight))
                elif criterion == 'entropy':
                    costFunction = leftCumulativeCount*(sp.special.entr(pLeft)+sp.special.entr(1.0-pLeft))+rightCumulativeCount*(sp.special.entr(pRight)+sp.special.entr(1.0-pRight))
                else:
                    print('{} is not a supported cost function'.format(criterion))
                    return                        
                indxArgMin = np.argmin(costFunction[left_sample_start:right_sample_stop])+left_sample_start
                if costFunction[indxArgMin] < min_cost:
                    self.split_feature_value = xUniqueSorted[indxArgMin]
                    self.split_feature_index = iFtr
                    min_cost = costFunction[indxArgMin]
                iFtrCount+=1
                if iFtrCount >= max_features:
                    break
            if iFtrCount == 0:
                self.prediction_value = np.mean(y)
                return                    
            min_cost_argsort=x[:,self.split_feature_index].argsort()
            x=x[min_cost_argsort,:]
            y=y[min_cost_argsort]
            min_costi_index = np.argmax(x[:,self.split_feature_index]>self.split_feature_value)
            self.less_than_node = RandomForest.RFTreeNode()
            self.less_than_node.grow_tree(x[:min_costi_index,:],y[:min_costi_index],max_features,criterion,min_samples_leaf,min_samples_split,max_depth,self.depth)
            self.greater_than_node = RandomForest.RFTreeNode()
            self.greater_than_node.grow_tree(x[min_costi_index:,:],y[min_costi_index:],max_features,criterion,min_samples_leaf,min_samples_split,max_depth,self.depth)
        
        def predict(self,x):
            #print(self.depth)
            if self.prediction_value is not None:
                return self.prediction_value
            else:
                if x[self.split_feature_index] > self.split_feature_value:
                    return self.greater_than_node.predict(x)
                else:
                    return self.less_than_node.predict(x)           
                
        
    def fit(self,x,y):
        max_features = self.max_features
        if type(self.max_features) == str:
            max_features = np.int32(np.ceil(np.sqrt(x.shape[1])))
            
        np.random.seed(self.random_state)
        
        for iEstimator in range(self.n_estimators):
            self.tree_base_node.append(self.RFTreeNode())
            if self.bootstrap == True:
                bootstrapIndx = np.random.randint(0,x.shape[0],size=x.shape[0],dtype=np.int32)
                xBootstrap = x[bootstrapIndx,:]
                yBootstrap = y[bootstrapIndx]
                self.tree_base_node[iEstimator].grow_tree(xBootstrap,yBootstrap,max_features,self.criterion,self.min_samples_leaf,self.min_samples_split,self.max_depth,0)
            else:
                self.tree_base_node[iEstimator].grow_tree(x.copy(),y.copy(),max_features,self.criterion,self.min_samples_leaf,self.min_samples_split,self.max_depth,0)
            
    def predict(self,x):
       # print('predict')
        if len(x.shape) == 1:
            prediction = 0.0
            for iEstimator in range(self.n_estimators):
                #print('Estimator {}'.format(iEstimator))                
                prediction+=self.tree_base_node[iEstimator].predict(x)
            return prediction/float(self.n_estimators)
        else:
            prediction = np.zeros(x.shape[0])
            for iEstimator in range(self.n_estimators):
                #print('Estimator {}'.format(iEstimator))
                prediction+=np.array([self.tree_base_node[iEstimator].predict(x[indx,:]) for indx in range(x.shape[0])])
            return prediction/float(self.n_estimators)