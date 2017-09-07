#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import os
import numpy as np
from random import uniform as rand


#TODO read from files
W1 = np.random.uniform(0,5,(2,3))
W2 = np.random.uniform(0,5,(3,1))
Ws = [W1, W2]


#fol = '/home/ngarcia/Dropbox/Diana_summer_INL/src/samples/lutchin_wire_matrix/4/the_hybrid'
#files = os.popen('ls %s/W*.npy'%(fol)).read().splitlines()
#files = sorted(files,key=lambda x: int(x.split('/')[-1].replace('W','').replace('.npy','')))
#Ws = []
#for f in files:
#   Ws.append(np.load(f))

dims = [Ws[0].shape[0]]
for w in Ws:
   dims += [w.shape[1]]

X,Y = [],[]
layers = []
for i in range(len(dims)):
   if dims[i]==1: ym = 0  #TODO do it more elegantly
   else: ym = dims[i]/2
   aux = []
   for y in np.linspace(-ym,ym,dims[i]):
      X.append(i)
      Y.append(y)
      aux.append( (i,y) )
   layers.append(aux)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(X,Y,s=500,edgecolors='none',zorder=10)

links = []
for il in range(1,len(layers)):
   l1 = layers[il-1]
   l2 = layers[il]
   print('Connections between',il-1,'and',il)
   print('L1:',l1)
   print('L2:',l2)
   w = Ws[il-1]
   #print(il,'',np.max(w),np.min(w))
   print(w)
   for i in range(len(l1)):
      for j in range(len(l2)):
         #print(i,j,'',l1[i],l2[j],'  ',w[i,j])
         x = [l1[i][0],l2[j][0]]
         y = [l1[i][1],l2[j][1]]
         ax.plot(x,y,'k-',lw=abs(w[i,j]),zorder=0)
plt.show()
