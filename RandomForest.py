# -*- coding: utf-8 -*-
"""
Created on Wed Nov  13 09:43:52 2018

@author: gward
"""

import numpy as np

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
                min_impurity_split=None,
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
            self.less_than_node = 0
            self.greater_than_node = 0
            self.prediction_value = None
            self.depth = 0
            
        def grow_tree(self,x,y,max_features,criterion,min_samples_leaf,min_samples_split,max_depth,depth):
            
            self.depth = depth+1
            if self.depth == max_depth:
                self.prediction_value = np.mean(y)
                return
                
            ySum = np.sum(y)
            if ySum == 0 or ySum == y.shape[0]:
                self.prediction_value = y[0]
                return
            
            if x.shape[1] <= min_samples_split:
                self.prediction_value = np.mean(y)
                return
            
            feature_indicies = np.random.randint(0,x.shape[0],size=max_features)
            if criterion == 'gini':
                min_gini_index = -1
                min_gini_argsort = np.zeros(x.shape[1])
                min_gini_impurity = float(x.shape[1]+2)
                count = np.arange(1.0,float(x.shape[1]+1))
                for iFtr in feature_indicies:
                    xArgSort=x[iFtr,:].argsort()
                    xSorted = x[xArgSort]
                    ySorted = y[xArgSort]
                    pLeft = np.cumsum(ySorted)/count
                    pRight = np.cumsum(ySorted[::-1]/count)[::-1]      
                    giniImpurity = count*pLeft*(1.0-pLeft)+(float(xSorted.shape[1]+1)-count)*pRight*(1.0-pRight)
                    indxArgMin = np.argmin(giniImpurity[min_samples_leaf-1:giniImpurity.shape[0]-min_samples_leaf-1])+min_samples_leaf-1
                    if giniImpurity[indxArgMin] < min_gini_impurity:
                        self.split_feature_value = xSorted[indxArgMin]
                        self.split_feature_index = iFtr
                        min_gini_impurity = giniImpurity[indxArgMin]
                        min_gini_index = indxArgMin
                        min_gini_argsort = xArgSort
                x=x[:,min_gini_argsort]
                y=y[min_gini_argsort]
                self.less_than_node = self.RFTreeNode()
                self.less_than_node.grow_tree(x[:,:min_gini_index+1],y[:min_gini_index+1],max_features,criterion,min_samples_leaf,min_samples_split,max_depth,self.depth)
                self.greater_than_node = self.RFTreeNode()
                self.less_than_node.grow_tree(x[:,min_gini_index+1:],y[min_gini_index+1:],max_features,criterion,min_samples_leaf,min_samples_split,max_depth,self.depth)     
            else:
                print('ENTROPY NOT IMPLEMENTED')
            
        def predict(self,x):
            if self.prediction_value is not None:
                return self.prediction_value
            else:
                if x[self.split_feature_index] <= self.split_feature_value:
                    return self.less_than_node.predict(x)
                else:
                    return self.greater_than_node.predict(x)           
                
        
    def fit(self,x,y):
        max_features = self.max_features
        if type(self.max_features) == str:
            max_features = np.int32(np.ceil(np.sqrt(x.shape[0])))
            
        np.random.seed(self.random_state)

        for iEstimator in range(self.n_estimators):
            self.tree_base_node.append(self.RFTreeNode())
            self.tree_base_node[-1].grow_tree(x,y,max_features,self.criterion,self.min_samples_leaf,self.min_samples_split,self.max_depth,0)
            
    def predict(self,x):
        if len(x.shape) == 1:
            prediction = 0.0
            for iEstimator in range(self.n_estimators):
                prediction+=self.tree_base_node[iEstimator].predict(x)
            return prediction/float(self.n_estimators)
        else:
            prediction = np.zeros(x.shape[1])
            for iEstimator in range(self.n_estimators):
                prediction+=np.array([self.tree_base_node[iEstimator].predict(x[:,indx]) for indx in range(x.shape[1])])
            return prediction/float(self.n_estimators)