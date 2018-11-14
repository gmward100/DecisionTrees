# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 10:25:39 2018

@author: gward
"""

import numpy as np
from RandomForest import RandomForest


x = np.zeros([100,100])
y = np.zeros(100)

#rf = RandomForest(criterion='entropy')
rf = RandomForest()
rf.fit(x,y)