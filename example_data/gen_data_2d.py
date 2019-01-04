#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt


X,Y,Z = [],[],[]
for x in np.linspace(-np.pi,np.pi,100):
   for y in np.linspace(-np.pi,np.pi,100):
      X.append(x)
      Y.append(y)
      Z.append(np.sin(x*y))

with open('sin2d.dat','w') as f:
   for ix,iy,iz in zip(X,Y,Z):
      f.write(str(ix)+'   '+str(iy)+'   '+str(iz)+'\n')

#X,Y,Z = [],[],[]
#for i in range(x.shape[0]):
#   for j in range(x.shape[1]):
#      X.append(x[i,j])
#      Y.append(y[i,j])
#      Z.append(z[i,j])

fig, ax = plt.subplots()
ax.scatter(X,Y,c=Z)
plt.show()
