#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

xs = np.array([
     0, 0,
     0, 1,
     1, 0,
     1, 1]).reshape(4, 2)


ys = np.array([0, 1, 1, 0]).reshape(4,)

xs = np.random.uniform(0,41,size=(100,1))
ys = []
for x in xs:
   if 0<= x < 10: ys.append(0.)
   elif 10<= x < 25: ys.append(1.)
   elif 25<= x: ys.append(2.)
ys = np.array(ys)


model = MLPClassifier(
             activation='tanh', max_iter=10000, hidden_layer_sizes=(4,2))

model.fit(xs, ys)

print('score:', model.score(xs, ys)) # outputs 0.5
x = np.linspace(0,42,50).reshape(50,1)
y = model.predict(x)
print(y)
#print('predictions:', model.predict(xs)) # outputs [0, 0, 0, 0]
#print('expected:', np.array([0, 1, 1, 0]))
fig, ax = plt.subplots()
ax.plot(x,y)
ax.scatter(xs,ys)
plt.show()

import pickle
# save the classifier
with open('my_dumped_classifier.pkl', 'wb') as fid:
    pickle.dump(model, fid)    

print('*'*80)
print('Loading model:')
# load it again
with open('my_dumped_classifier.pkl', 'rb') as fid:
    model_loaded = pickle.load(fid)

x = np.linspace(0,42,50).reshape(50,1)
y = model_loaded.predict(x)
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(x,y)
plt.show()
