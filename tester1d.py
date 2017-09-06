#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np
import os
from evaluate import evaluate_net

f_net = 'my_network.xml'
f_data = 'sin.dat'

X,Y = np.loadtxt(f_data,unpack=True)

d = 0.1*(max(X)-min(X))
x = np.linspace(min(X)-d,max(X)+d,100)
Xx,Yy = [],[]
for ix in x:
   Xx.append(ix)
   Yy.append(evaluate_net(f_net,(ix,)))


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(X,Y,c='b',s=30,edgecolor='none',label='Data',alpha=0.5)
ax.plot(Xx,Yy,'r',label='NN')
ax.grid()
ax.legend()
plt.show()
