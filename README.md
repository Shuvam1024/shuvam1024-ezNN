# simpleNN
simpleNN is a library that implements a multilayer perceptron (MLP), allowing users to train models for various regression and classification tasks. It provides functions for creating an MLP with given hyperparameters and objective, training the parameters of the MLP using a training set, and running inference on the trained model a testing set. It also includes functions for saving a trained model into a file and loading it back for subsequent inference or further fine-tuning.

Additionally, this repository includes a command line tool which allows users to train and test a multilayer perceptron model from csv files using the sinpleNN library.

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

To build the driver commandline tool `runSimpleNN`, you need the following additional files:
- `readwrite_csv.c`
- `readwrite_csv.h`
- `main.c` (driver code for command line tool)
- To compile runSimpleNN:
  -   gcc -o  runSimpleNN simpleNN.c readwrite_csv.c main.c -lm
  
# Usage of `runSimpleNN` command-line tool

Usage:

`runSimpleNN`  
> &lt;mode&gt;  
> &lt;list_of_layers&gt;  
> &lt;list_of_activations&gt;  
> &lt;learning_rate&gt;  
> &lt;num_epochs&gt;  
> &lt;training_csv[,testing_csv]&gt;  
> &lt;model_filename&gt;  
> [&lt;map_of_columns_in_csv&gt;]  


> &lt;mode&gt; is one of four strings: "MODE_REGRESSION_L2", "MODE_REGRESSION_L1", "MODE_BINARY_CLASSIFICATION", "MODE_MULTICAT_CLASSIFICATION", representing the type of network to be trained.

> &lt;list_of_layers&gt; is a comma-separated list of integers representing the number of nodes in each layer starting from the input layer to the output layer. For example if the list has four numbers the first represents the number of nodes in the input layer, the last represents the number of nodes in the output layer, and the two in the middle present the number of nodes in two hidden layers.  
> If the given mode is "MODE_MULTICAT_CLASSIFICATION", the number of output nodes should be the same as the number of categories. However, the training and testing set is expected to have only a single output column corresponding to the index of the given category. If the  given mode is "MODE_BINARY_CLASSIFICATION", the number of output nodes is the number of independant binary classifiers the network is supposed to be trained for. For regression, the number of ouput nodes represents the dimensions of the output vector.  

> &lt;list_of_activations&gt; is a list of actTypes which should be one less than the <list_of_layers> as the input layer will not have any activation. The activation types are indicated by a comma-separated list of strings, where each is one of "ACT_IDENTITY", "ACT_RELU", "ACT_TANH", "ACT_SIGMOID", or "ACT_SOFTMAX". Note  
>  - If the given mode is "MODE_MULTICAT_CLASSIFICATION", the last activation function should be "ACT_SOFTMAX"
>  - If the given mode is "MODE_MULTICAT_CLASSIFICATION", the last activation function should be "ACT_SIGMOID"

> &lt;learning_rate&gt; is the learning rate for training

> &lt;num_epochs&gt; is the number epochs for training

> &lt;training_csv[,testing_csv]&gt; is a list of the training csv file and the testing csv file seperated by a comma. 
>  - Note that the testing csv file is optional if just training.

> &lt;model_filename&gt; is the name of the file in which to save the trained model; 
>  - If a testing csv file is provided, it will load the model back from this file to run inference on the testing set.

> [&lt;map_of_columns_in_csv&gt;] is an optional parameters which provides a comma-separated ordered list of collumn indices in the csv files representing the input features followed by output features in order.
>  - Note that if the given mode is "MODE_MULTICAT_CLASSIFICATION", the number of output features is always 1, whereas in other cases it is the same as the number of nodes in the output layer.
>  - If the map is not provided, it is assumed that the csv files have input features followed by output features in order as their only columns.

# Example `runSimpleNN` command lines

Regression L2 command line

> runSimpleNN  MODE_REGRESSION_L2  6,5,3,1  ACT_RELU,ACT_RELU,ACT_IDENTITY  0.000001  20000  train.csv,test.csv  mymodel.dat

Binary Classification command line

> runSimpleNN  MODE_BINARY_CLASSIFICATION  4,5,3,1  ACT_RELU,ACT_TANH,ACT_SIGMOID  0.0005  10000  train.csv,test.csv  mymodel.dat

Multicategory Classification command line

> runSimpleNN  MODE_MULTICAT_CLASSIFICATION  4,5,3,1  ACT_RELU,ACT_TANH,ACT_SOFTMAX  0.0005  10000  ./iris.csv  mymodel.dat  


# The core simpleNN library API

These are the core public functions in the simpleNN library API

- `void init_simpleNN(simpleNNType *nn, modeType mode, int nlayers, int *sizes, actType *acts);` 
> >Initializes the simpleNN structure with mode, number of layers, a list of sizes for each layer and a list of activations. Note if there are n layers, there needs to be n sizes and n-1 activations provided.
> >
> >Parameters:
> - `nn`: Pointer to simpleNN structure to be initialized
> - `mode`: type of network i.e. classification/regression mode
> - `nlayers`: Number of layers in the neural network   
> - `sizes`: Array with sizes for each layer starting from the input layer and up to the output layer. Note number of sizes is one less than the number of layers.      
> - `acts`: Array with list of activation types for each layer other than the input layer. Note number of activation types is one less than the number of layers.        
     
- `void free_simpleNN(simpleNNType *nn);`
> >Frees the simpleNN structure
> >  
> >Parameters:
> - `nn`: Pointer to simpleNN structure to be freed
  
- `void do_inference(simpleNNType *nn, float **inputs, int n, float **outputs);`
> >Runs inference on a list of inputs
> >
> >Parameters:
> - `nn`: Pointer to simpleNN structure to be used for inference
> - `inputs`: 2D array where each row contains the list of input features corresponding to a sample to run inference on.
> - `n`: Number of input sample vectors i.e. number of rows in the `inputs` 2D array
> - `outputs`: 2D array where each row contains the list of outputs generated for each sample. Space must be pre-allocated for the outputs. Note that in multicategory classification mode, the number of outputs is 1. In other cases, the number of outputs for each sample is the same as the number of nodes in the output layer of the network.
   
- `void do_training(simpleNNType *nn, float **train_data, int train_sample, float learning_rate, int max_epochs, int reset);`
> >Trains a network based on a training set
> >
> >Parameters:
> - `nn`: Pointer to simpleNN structure to be used for inference
> - `train_data`: 2D array where each row is a training sample comprising the input features for a training sample followed by its expected output vector. Note the number of input features is the same as the number of nodes in the first (input) layer of the network. The number of outputs for each sample is 1 in multicategory classification mode, and is the same as the number of nodes in the output layer of the network in all other modes.
> - `train_sample`: Number of trainig samples
> - `learning_rate`: Learning rate for training
> - `max_epochs`: Number of epochs to run training on
> - `reset`: Flag to indicate whether to reset and randomly initialize the parameters of the netowrk prior to training. Set to 1 for new training starting from random initialization. When set as 0, this function can be used to continue or fine-tune training on a given training set, starting from an already pre-trained network.
   
- `void save_model_to_file(simpleNNType *nn, char * model_filename);`
> >Saves trained model architecture and parameters to a file
> >
> >Parameters:
> - `nn`: Pointer to simpleNN structure to be used to export model from
> - `model_filename`: Filename for model file to save   
  
- `void load_model_from_file(simpleNNType *nn, char * model_filename);`
> >Loads pre-trained model architecture and parameters from a file
> >
> >Parameters:
> - `nn`: Pointer to simpleNN structure to be used to import model into
> - `model_filename`: Filename for model file to load





