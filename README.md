# simpleNN
simpleNN is a library that implements a multilayer perceptron (MLP), allowing users to train models with several modes and loss functions. It provides functions for initializing the MLP with given hyperparameters, training from a training set, and running inference on a testing set. It also includes functions for saving a trained model into a file and loading it back for subsequent inference.

Additionally, this repository includes a command line tool which allows users to train and test a multilayer perceptron model through csv files using the sinpleNN library.

# Features
- C99 with no dependencies.
- Supports multiple activation functions: identity, ReLU, sigmoid, tanh, and softmax.
- Implements backpropagation for training.
- Supports various training modes: L2 regression, L1 regression, binary classification, and multi-class classification.
- Easily extendable.
- Contains functions for reading from and writing to CSV files.
- Contains functions for saving trained models to files and loading them back for further training or inference.
- Examples included for training and testing models.

# Building

simpleNN is contained in a few source files. To use simpleNN, simply add the following files to your project:
- `simpleNN.c`
- `simpleNN.h`

To build the driver commandline tool (runSimpleNN), you need the following additional files:
- `readwrite_csv.c`
- `readwrite_csv.h`
- `main.c` (driver code for command line tool)
  - To compile runSimpleNN:
  -   gcc -g -o  runSimpleNN simpleNN.c readwrite_csv.c main.c
  
  
  ./runSimpleNN MODE_REGRESSION_L2 6,5,3,1 ACT_RELU,ACT_RELU,ACT_IDENTITY 0.000001 20000 ./Realestate_train.csv,./Realestate_test.csv






