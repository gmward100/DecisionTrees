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
        self.split_feature_indicies = []
        self.split_feature_values = []
        
    def grow_tree(self,x,y,max_features):
        print('growing')
        
    def fit(self,x,y):
        max_features = self.max_features
        if type(self.max_features) == str:
            max_features = np.int32(np.ceil(np.sqrt(x.shape[1])))
            
        np.random.seed(self.random_state)

        for iEstimator in range(self.n_estimators):
            self.split_feature_indicies.append([])
            self.split_feature_values.append([])
            self.grow_tree(x,y,max_features)