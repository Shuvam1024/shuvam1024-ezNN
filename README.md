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
  
# Usage of runSimpleNN command-line tool

Usage
>runSimpleNN
>
&lt;<mode>&gt;
>
list_of_layers
>
list_of_activations
>
learning_rate
>
num_epochs
>
training_csv[,testing_csv]
>
model_filename
>
[map_of_columns_in_csv]
>


`mode` is one of four strings: "MODE_REGRESSION_L2", "MODE_REGRESSION_L1", "MODE_BINARY_CLASSIFICATION", "MODE_MULTICAT_CLASSIFICATION", representing the type of network to be trained.

`list_of_layers` is a comma-separated list of integers representing the number of nodes in each layer starting from the input layer to the output layer. For example if the list has 4 numbers the first represents the number of nodes ij the input layer, the last represents the number of nodes in the output layer, and the 2 in the middle present the number of nodes in the 2 hidden layers. 

- If the given mode is "MODE_MULTICAT_CLASSIFICATION", the number of output nodes should be the same as the number of categories. However, the training and testing set should only have 1 output column corresponding to the index of the category. If the given mode is "MODE_BINARY_CLASSIFICATION", the number of output nodes represents the number of independant binary classifiers the network is supposed to be trained for. For regression, the number of ouput nodes represents the dimensions of the output vector. Note that the training and testing set in multicategory classification is expected to only have 1 output column corresponding to the index of the category.

`list_of_activations` is a list of actTypes which should be one number shorter than the <list_of_layers> as the input features layer will not have any activation.

  The actTypes are, ACT_IDENTITY, ACT_RELU, ACT_TANH,ACT_SIGMOID,ACT_SOFTMAX
  
  - If the given mode is "MODE_MULTICAT_CLASSIFICATION", the last activation function should be ACT_SOFTMAX
  
  - If the given mode is "MODE_MULTICAT_CLASSIFICATION", the last activation function should be ACT_SIGMOID

`learning_rate` is the learning rate for training

`num_epochs` is the number epochs

`training_csv`[,`testing_csv`] is a list of the training csv file and the testing csv file seperated by a comma. 

  - Note that the testing csv file is optional if just training.

`model_filename` is the name of the file in which to save the trained model; 

  - If a testing csv file is provided, it will load the model back from this file to run inference on the testing set.

[`map_of_columns_in_csv`] is an optional parameters which provides an ordered list of collumn indices in the csv files which represent the input features followed by output features.
  - Note that the given mode is "MODE_MULTICAT_CLASSIFICATION", the number of output features is always 1, where as in other cases it is the same as the number of noes in the output layer.


  ./runSimpleNN MODE_REGRESSION_L2 6,5,3,1 ACT_RELU,ACT_RELU,ACT_IDENTITY 0.000001 20000 ./Realestate_train.csv,./Realestate_test.csv






