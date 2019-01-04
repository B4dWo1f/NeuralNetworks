#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import pickle
import numpy as np
from sklearn.neural_network import MLPRegressor as MLP
import matplotlib.pyplot as plt


## Load training data
X,Y,Z = np.loadtxt('../example_data/sin2d.dat',unpack=True)

X0 = X/np.max(np.abs(X))
Y0 = Y/np.max(np.abs(Y))
Z0 = Z/np.max(np.abs(Z))

if len(X0.shape) == 1: X0 = X0.reshape((X0.shape[0],1))


with open('my_dumped_classifier.pkl', 'rb') as fid:
   model = pickle.load(fid)

Z_model = []
for x,y in zip(X0,Y0):
   Z_model.append( model.predict(x,y) )

fig, ax = plt.subplots()
ax.scatter(X0,Y0,c=Z0)
#ax.plot(x,y,'C1')

fig, ax = plt.subplots()
ax.scatter(X0,Y0,c=Z_model)
plt.show()
