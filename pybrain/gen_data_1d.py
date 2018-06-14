#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt

x = np.random.uniform(0,2*np.pi,size=(100,))
y = np.sin(x) + np.random.uniform(-0.1,0.1,size=x.shape)

with open('sin.dat','w') as f:
   for ix,iy in zip(x,y):
      f.write(str(ix)+'   '+str(iy)+'\n')

fig, ax = plt.subplots()
ax.scatter(x,y)
plt.show()
