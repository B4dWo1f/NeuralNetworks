#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import pickle
import numpy as np
from sklearn.neural_network import MLPRegressor as MLP
import matplotlib.pyplot as plt


## Load training data
X,Y = np.loadtxt('../example_data/sin.dat',unpack=True)

X0 = X/np.max(np.abs(X))
Y0 = Y/np.max(np.abs(Y))

if len(X0.shape) == 1: X0 = X0.reshape((X0.shape[0],1))


with open('my_dumped_classifier.pkl', 'rb') as fid:
   model = pickle.load(fid)

x = np.linspace(min(X0),max(X0),50).reshape(50,1)
y = model.predict(x)
fig, ax = plt.subplots()
ax.scatter(X0,Y0,c='C0')
ax.plot(x,y,'C1')
plt.show()
