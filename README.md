# simpleNN
simpleNN is a library that implements a multilayer perceptron (MLP), allowing users to train models with several modes and loss functions. It provides functions for initializing the MLP with given hyperparameters, training from a training set, and running inference on a testing set. 

Additionally, this repository includes a command line tool which allows users to train and test a multilayer perceptron model through csv files.

# features
- C99 with no dependencies.
- Supports multiple activation functions: identity, ReLU, sigmoid, tanh, and softmax.
- Implements backpropagation for training.
- Supports various training modes: L2 regression, L1 regression, binary classification, and multi-class classification.
- Easily extendable.
- Contains functions for reading from and writing to CSV files.
- Examples included for training and testing models.

# Building

simpleNN is contained in a few source files. To use simpleNN, simply add the following files to your project:
- `simpleNN.c`
- `simpleNN.h`
- ‘main.c’ (if not modifying - can train directly from command line)


# Example

To compile and run the simpleNN library from the command line:


gcc -g -o runSimpleNN simpleNN.c readwrite_csv.c main.c
./runSimpleNN MODE_BINARY_CLASSIFICATION 4,5,3,1 ACT_RELU,ACT_TANH,ACT_SIGMOID 0.0005 10000 ./data_train.csv,./data_test.csv




