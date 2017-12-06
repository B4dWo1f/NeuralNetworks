# General Feed Forward generator

This repo is a wrapper to some pyBrain ( http://pybrain.org/ ) functions.

This repo provides two general, and hopefully useful, scripts to create, train and evaluate general Feed Forward Neural Networks taking care of all the details (dimension, normalization of the data and so on).

# Options
  -h, --help       show this help message and exit
  
  -i I             Name of the input data file
  
  -o O             Name of the file to save the network
  
  -n N             Number of columns to be considered as output
  
  -ar AR [AR ...]  Architecture of the Neural Network (tuple)
  
  -m               Randomize the order of input data  [Not implemented]


# Usage:
Consider that we have a data file "datos.dat" which contains our training data with format:

input1   input2  input3.......output1   output2   outputN

The training data set will have M rows (each row is a sample of input-output) and N columns, the first (N-n) columns are inputs, and the last (n) columns are outputs.
Let's Consider that datos.dat contains 3 columns (2 input, 1 output), and we want an architecture 2,5,1
To build the model and train it:

./train.py -i datos.dat -n 1 -ar 2 5 1 -o "my_cool_NN.xml"

You can check the training process by running: check_train.py

Once the training is done, you should have the files my_cool_NN.xml and my_cool_NN.lim, which contains the NN parameters.

To use the saved NN I provide the evaluate.py library:
./evaluate -f my_cool_NN.xml 0.5

# TO-DO
- [ ] Previous check of size/architecture of the training data
- [ ] Include training/validation data
- [ ] Fragmentation of training data
- [ ] Randomize order for training data
- [ ] Smart guess for architecture
