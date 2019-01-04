#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import pickle
import numpy as np
from sklearn.neural_network import MLPRegressor as MLP
import matplotlib.pyplot as plt


## Load training data
X,Y,Z = np.loadtxt('../example_data/sin2d.dat',unpack=True)
M = np.loadtxt('../example_data/sin2d.dat')
print(M.shape)
Z = M[:,2]
M = M[:,0:2]

#X0 = X/np.max(np.abs(X))
#Y0 = Y/np.max(np.abs(Y))

M0 = M/np.max(np.abs(M))
Z0 = Y/np.max(np.abs(Z))

#if len(Z0.shape) == 1: Z0 = Z0.reshape((Z0.shape[0],1))



## Try to load model
try:
   with open('my_dumped_classifier.pkl', 'rb') as fid:
      model = pickle.load(fid)
   print('Model loaded')
except FileNotFoundError:
   print('Starting new model')
   model = MLP(activation='logistic',
               solver='adam',
               learning_rate='adaptive',
               learning_rate_init=0.01,
               alpha=0.01,
               max_iter=1000,
               hidden_layer_sizes=(3,5),
               warm_start=True)


model.fit(M, Z0)
#Err = []
#for i in range(10000):
#   model.partial_fit(X0, Y0)
#   Err.append(1-model.score(X0,Y0))

#import matplotlib.pyplot as plt
#fig, ax = plt.subplots()
#ax.plot(Err)
#plt.show()

# save the classifier
with open('my_dumped_classifier.pkl', 'wb') as fid:
    pickle.dump(model, fid)    

print('score:', model.score(M0, Z0)) # outputs 0.5
exit()
x = np.linspace(min(X0),max(X0),50).reshape(50,1)
y = model.predict(x)
#print('predictions:', model.predict(xs)) # outputs [0, 0, 0, 0]
#print('expected:', np.array([0, 1, 1, 0]))
fig, ax = plt.subplots()
ax.scatter(X0,Y0,c='C0')
ax.plot(x,y,'C1')
#plt.show()
