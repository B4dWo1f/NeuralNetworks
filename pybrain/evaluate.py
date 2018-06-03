#!/usr/bin/python3
# -*- coding: UTF-8 -*-

"""
  This code evaluates the given conditions returning the flying level requierd.
"""

### LOG #########################################################################
#import logging
#logging.basicConfig(level=logging.DEBUG,
#                  format='%(asctime)s %(name)s:%(levelname)s - %(message)s',
#                  datefmt='%Y/%m/%d-%H:%M:%S',
#                  filename='eval.log', filemode='w')
### Console Handler
#sh = logging.StreamHandler()
#sh.setLevel(logging.INFO)
#fmt = logging.Formatter('%(name)s -%(levelname)s- %(message)s')
#sh.setFormatter(fmt)
##logging.getLogger('').addHandler(console)
#LG = logging.getLogger('main')
#LG.addHandler(sh)
#################################################################################

## PyBrain
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader
## General Libraries
from scipy.interpolate import interp1d  # For interpolations
import numpy as np
import sys

## Options #####################################################################
import argparse
parser = argparse.ArgumentParser(description='Main parser')
parser.add_argument('-f',nargs=1,type=str,default='my_network.xml',
                    help='Network file')
parser.add_argument('rest', nargs='*')
args = parser.parse_args()
################################################################################

def evaluate_net(f_net,inputs):
   try: iter(inputs)
   except: inputs = [inputs]
   ## Read limits for inputs/outputs
   f_lim = f_net.replace('.xml','.lim')
   lims = np.loadtxt(f_lim)
   
   #LG.info('Neural Network file: %s'%(f_net))
   msg = 'Inputs for the neural network: '
   for i in inputs:
      msg += str(i)+', '
   
   ## Build interpolators
   net = NetworkReader.readFrom(f_net) # Re-use existing network
   inL = net['in'].indim
   outL = net['out'].outdim
   interpols,cont = [],0
   for _ in range(inL):
      interpols.append( interp1d([lims[cont,0],lims[cont,1]],
                                 [-1,1],
                                 fill_value='extrapolate') )
      cont += 1
   for _ in range(outL):
      interpols.append( interp1d([-1,1],
                                 [lims[cont,0],lims[cont,1]],
                                 fill_value='extrapolate') )
      cont += 1
   
   
   ## Intputs to correct format
   inputs = [[i] for i in inputs]
   
   
   ## Scale the inputs
   cont = 0
   for i in range(len(inputs)):
      inputs[i] = interpols[cont]( inputs[i] )
      cont += 1
   
   
   ## Calculate the neural network output
   outputs = net.activate(tuple(inputs))
   
   
   ## Try to scale back the outputs
   outputs = list(outputs)
   for i in range(len(outputs)):
      outputs[i] = interpols[cont]( outputs[i] )
      cont += 1
   return outputs


if __name__ == '__main__':
   ## Read inputs
   f_net = args.f[0]
   inputs = list(map(float,args.rest))

   print(evaluate_net(f_net,inputs))

