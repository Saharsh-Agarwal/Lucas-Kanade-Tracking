# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 17:53:04 2022

@author: sahar
"""

import numpy as np 

a = np.load(r'C:\Users\sahar\Desktop\carseqrects.npy')
b = np.load(r'C:\Users\sahar\Desktop\carseqrects-wcrt.npy')
for i in b :
    print(i)