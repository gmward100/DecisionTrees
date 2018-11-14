# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 10:25:39 2018

@author: gward
"""

import numpy as np
from RandomForest import RandomForest
import pandas as pd

iris_data = pd.read_csv('Iris.csv', keep_default_na=False)
iris_data['species'] = iris_data['species'].str.replace('setosa','1')
iris_data['species'] = iris_data['species'].str.replace('versicolor','0')
iris_data['species'] = iris_data['species'].str.replace('virginica','0')

#x = iris_data.as_matrix(columns=['sepal_length','sepal_width','petal_length','petal_width'])
x = np.array(iris_data[iris_data.columns[0:4]].values,dtype=np.float32)
y = np.array(iris_data['species'].astype(np.float32))

print(y)
print(x)
print(x.shape)
#
rf = RandomForest(criterion='entropy')
#rf = RandomForest()
rf.fit(x,y)