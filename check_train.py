#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt

fname = 'Err.npy'
try: M = np.load(fname)
except FileNotFoundError: exit('No %s file'%(fname))


print('Error:',np.mean(M[-5:]))
dolog = False

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter([i for i in range(len(M))],M, c='blue', edgecolors='none')
if dolog:
   ax.set_yscale('log')
   ax.semilogy(M,'b-')
else:
   ax.plot(M,'b-')
   ax.set_ylim([0,1.05*np.max(M)])
ax.set_xlim([-5,len(M)+5])
ax.set_ylabel('Error (a.u.)')
ax.set_xlabel('Iteration')
ax.axhline(10**-5,c='k',ls='--')
ax.grid()
fig.tight_layout()
plt.show()
