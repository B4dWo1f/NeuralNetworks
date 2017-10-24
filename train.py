#!/usr/bin/python3
# -*- coding: UTF-8 -*-

"""
  This code is a general feed forward neural network trainer. By default it
  takes a data file 'datos.dat' from which considers the last n columns as
  outputs and all the previous ones as inputs.
  Options:
   -i Input data file
   -n Number of nodes in the output layer
   -o File to store the neural network
   -ar Architecture of the neural network
   -nb Batch size
 TODO:
  * Add option for N_train
  * Add option for minimal error (bool + eps)
  * Add option for overwrite file
  * Add options for Lrate, Ldecay
"""

############################### LOGGING #####################################
import logging
import log_help
logging.basicConfig(level=logging.DEBUG,
                 format='%(asctime)s %(name)s:%(levelname)s - %(message)s',
                 datefmt='%Y/%m/%d-%H:%M:%S',
                 filename='trainer.log', filemode='w')
LG = logging.getLogger('main')
log_help.screen_handler(LG)
#############################################################################


## Standard libraries
import numpy as np
from scipy.interpolate import interp1d  # For interpolations
#import sys
import os

## PyBrain library
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet,ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader


## Options #####################################################################
import argparse
parser = argparse.ArgumentParser(description='Main parser')
parser.add_argument('-i',nargs=1,type=str, #default='datos.dat',
                    help='Name of the input data file')
parser.add_argument('-o',nargs=1,type=str,default=['my_network.xml'],
                    help='Name of the file to save the network')
parser.add_argument('-n',nargs=1,type=int,default=1,
                    help='Number of columns to be considered as output')
parser.add_argument('-ar',nargs="+",type=int,default=(1,10,10,1),
                    help='Architecture of the Neural Network (tuple)')
parser.add_argument('-m',action='store_true',default=False,
                    help='Randomize the order of input data')
#parser.add_argument('-nb',nargs=1,type=int,default=5000,
#                    help='Batch size')
parser.add_argument('rest', nargs='*')
args = parser.parse_args()
################################################################################
def usage():
   msg  = 'This code is a general (feed-forward) neural network trainer '
   msg += 'prepared to read a data file with M columns. The first (M-n) '
   msg += 'columns will be consider as inputs and the n last columns as'
   msg += ' outputs.\nThe possible arguments are:\n'
   msg += '  -i: Specify the input data file\n'
   msg += '  -n: Specify number of nodes in the output layer [default: 1]\n'
   msg += '  -ar: Specify the architecture of the NN. example -ar 1 5 10 1'
   msg += ' it is important to not have spaces in this argument'
   msg += '  -o: Specify the file in which to store the neural network '
   msg += '[default: my_network.xml]\n'
   msg += '**Note that the program will create an extra file '
   msg += '\'my_network.lim\' containing the information for the data treatment'
   print(msg)
   exit()


## Parse and check filename
if args.i == None: usage()
f_data = args.i[0]
if len(args.i) > 1: LG.warning('Too many input files. Using only: %s'%(f_data))
if not os.path.isfile(f_data):
   LG.critical('Could not find file: %s'%(f_data))
   exit()
else: LG.info('Input data file: %s'%(f_data))
## Parse and check type of number of input/output
N = int(args.n[0])
LG.info('Using %s last coloumns as output'%(N))
## Parse architecture
arq = args.ar
LG.info('Architecture: (%s)'%(str(arq)))


## Parse output files
f_net = args.o[0]
f_lim = f_net.replace('.xml','.lim')
LG.info('NN will be saved in file: %s'%(f_net))
LG.debug('extra information will be saved in file: %s'%(f_lim))

mix = args.m
mix = False #TODO test this



## Read the data
try:
   M = np.loadtxt(f_data) #,unpack=True)
   inputs = M[:,:-N]
   outputs = M[:,-N:]
   if inputs.shape[0] != outputs.shape[0]:
      LG.warning('Different amount of input/outputs samples')
   else: LG.debug('Loaded %s samples of inputs/outputs'%(len(inputs)))
   if inputs.shape[1] != arq[0] or outputs.shape[1] != arq[-1]:
      LG.critical('Incompatible size of input data and requested architecture')
      msg = 'Expected %s inputs and %s outputs'%(arq[0],arq[-1])
      msg += ' data seems to have %s inputs'%(inputs.shape[1])
      msg += ' and %s outputs'%(outputs.shape[1])
      LG.debug(msg)
   else: LG.debug('NN architecture and data shape match')
except FileNotFoundError:
   LG.critical('Data file \'%s\' not found'%(f_data))
   exit()


lims = []
for i in range(inputs.shape[1]):
   # XXX shame on you!!! do this better
   mini = min([0.95*np.min(inputs[:,i]),1.05*np.min(inputs[:,i])])
   maxi = max([0.95*np.max(inputs[:,i]),1.05*np.max(inputs[:,i])])
   lims.append( (mini,maxi) )
for i in range(outputs.shape[1]):
   # XXX shame on you!!! do this better
   mini = min([0.95*np.min(outputs[:,i]),1.05*np.min(outputs[:,i])])
   maxi = max([0.95*np.max(outputs[:,i]),1.05*np.max(outputs[:,i])])
   lims.append( (mini,maxi) )
lims = np.array(lims)
np.savetxt(f_lim,lims,fmt='%.5f')
LG.info('Saved scaling data parameters in %s'%(f_lim))


## mix data if requested
if mix:
   #XXX to be tested
   inds = np.array([range(inputs.shape[0])])
   from random import shuffle
   shuffle(inds)
   inputs = inputs[inds]
   outputs = outputs[inds]


## Build Interpolators for input/output
# All data should be scaled aprox to the interval [-1,1]
interpols = []
cont = 0
for _ in range(inputs.shape[1]):
   interpols.append( interp1d([lims[cont,0],lims[cont,1]],
                               [-1,1],
                               fill_value='extrapolate') )
   cont += 1
for _ in range(outputs.shape[1]):
   interpols.append( interp1d([lims[cont,0],lims[cont,1]],
                              [-1,1],
                              fill_value='extrapolate') )
   cont += 1

# Re-scale data
cont = 0
for i in range(inputs.shape[1]):
   inputs[:,i] = interpols[cont]( inputs[:,i] )
   cont += 1
for i in range(outputs.shape[1]):
   outputs[:,i] = interpols[cont]( outputs[:,i] )
   cont += 1


## Build Neural Network object
try:
   net = NetworkReader.readFrom(f_net) # Re-use existing network
   inL = net['in'].indim
   outL = net['out'].outdim
   LG.info('Resuming training of network stored in %s'%(f_net))
   if inputs.shape[1] != inL or outputs.shape[1] != outL:
      LG.critical('Loaded neural network from %s is not compatible with the data dimensions in %s'%(f_net,f_data))
      exit()
   # TODO add architecture information
except: 
   LG.info('Starting a new NN with shape: (%s)'%(str(arq)))
   inL = arq[0]
   outL = arq[-1]
   net = buildNetwork(*arq,bias=True)


## Prepare DataSet
# Supervised DataSet
ds = SupervisedDataSet(inL,outL)  # Dimensions of the inputs,outputs
# Add data to the dataset
for i in range(inputs.shape[0]):
   ds.addSample(tuple(inputs[i,:]), tuple(outputs[i,:]))


## Trainer
Lrate = 0.01
Ldecay = 1.0
trainer = BackpropTrainer(net, ds,lrdecay=Ldecay) #,learningrate=Lrate)
Err = []
LG.info('Initial error: %.5e'%(trainer.train()))
cont = 10
error,eps = 1e5, 1e-5
while error>eps and not os.path.isfile('STOP'):
   error = trainer.train()
   cont += 1
   Err.append( error )
   #if os.path.isfile('STOP'): break
   if cont%(10**int(np.log10(cont))//2) ==0: 
      LG.info('%s iterations, error: %s'%(cont-10,Err[-1]))
      np.save('Err.npy',Err)
      ## Save network to file (for further training...)
      NetworkWriter.writeToFile(net, f_net)
      LG.debug('Neural network saved to %s'%(f_net))
cont -= 10
LG.info('Error after %s iterations: %s'%(cont,error))
## Save network to file (for further training...)
NetworkWriter.writeToFile(net, f_net)
LG.info('Neural network saved to %s'%(f_net))


## Save network to file (for further training...)
NetworkWriter.writeToFile(net, f_net)
LG.info('Neural network saved to %s'%(f_net))
if error < eps:
   msg = 'Low error achived (<%s)'%(eps)
   LG.info(msg)
elif error > 1e-3: LG.warning('NN probably not fully converged')


LG.info('Training complete. %s times. Error: %s'%(cont, error))
