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

`Usage`

runSimpleNN

>&lt;mode&gt;

>&lt;list_of_layers&gt;

>&lt;list_of_activations&gt;

>&lt;learning_rate&gt;

>&lt;num_epochs&gt;

>&lt;training_csv[,testing_csv]&gt;

>&lt;model_filename&gt;

>&lt;[map_of_columns_in_csv]&gt;



&lt;mode&gt; is one of four strings: "MODE_REGRESSION_L2", "MODE_REGRESSION_L1", "MODE_BINARY_CLASSIFICATION", "MODE_MULTICAT_CLASSIFICATION", representing the type of network to be trained.

&lt;list_of_layers&gt; is a comma-separated list of integers representing the number of nodes in each layer starting from the input layer to the output layer. For example if the list has four numbers the first represents the number of nodes in the input layer, the last represents the number of nodes in the output layer, and the two in the middle present the number of nodes in two hidden layers. 

- If the given mode is "MODE_MULTICAT_CLASSIFICATION", the number of output nodes should be the same as the number of categories. However, the training and testing set should only have 1 output column corresponding to the index of the category. If the given mode is "MODE_BINARY_CLASSIFICATION", the number of output nodes represents the number of independant binary classifiers the network is supposed to be trained for. For regression, the number of ouput nodes represents the dimensions of the output vector. Note that the training and testing set in multicategory classification is expected to only have 1 output column corresponding to the index of the category.

&lt;list_of_activations&gt; is a list of actTypes which should be one number shorter than the <list_of_layers> as the input features layer will not have any activation.

  The actTypes are, ACT_IDENTITY, ACT_RELU, ACT_TANH,ACT_SIGMOID,ACT_SOFTMAX
  
  - If the given mode is "MODE_MULTICAT_CLASSIFICATION", the last activation function should be ACT_SOFTMAX
  
  - If the given mode is "MODE_MULTICAT_CLASSIFICATION", the last activation function should be ACT_SIGMOID

&lt;learning_rate&gt; is the learning rate for training

&lt;num_epochs&gt; is the number epochs for training

&lt;training_csv[,testing_csv]&gt; is a list of the training csv file and the testing csv file seperated by a comma. 

  - Note that the testing csv file is optional if just training.

&lt;model_filename&gt; is the name of the file in which to save the trained model; 

  - If a testing csv file is provided, it will load the model back from this file to run inference on the testing set.

&lt;[map_of_columns_in_csv]&gt; is an optional parameters which provides a comma-separated ordered list of collumn indices in the csv files representing the input features followed by output features in order.
  - Note that if the given mode is "MODE_MULTICAT_CLASSIFICATION", the number of output features is always 1, where as in other cases it is the same as the number of noes in the output layer.
  - If this is not provided, it assumes the csv files has input features followed by output features as their only colllumns.

# Example command lines

Regression L2 command line

> runSimpleNN MODE_REGRESSION_L2 6,5,3,1 ACT_RELU,ACT_RELU,ACT_IDENTITY 0.000001 20000 train.csv,test.csv mymodel.dat

Binary Classification command line

> runSimpleNN MODE_BINARY_CLASSIFICATION 4,5,3,1 ACT_RELU,ACT_TANH,ACT_SIGMOID 0.0005 10000 train.csv,test.csv mymodel.dat

Multicategory Classification command line

> runSimpleNN MODE_MULTICAT_CLASSIFICATION 4,5,3,1 ACT_RELU,ACT_TANH,ACT_SOFTMAX 0.0005 10000 ./iris.csv mymodel.dat  



# The core simpleNN library API

These are the core public functions in the simpleNN library API

- `void init_simpleNN(simpleNNType *nn, modeType mode, int nlayers, int *sizes, actType *acts);` 
>Initializes the simpleNN structure with mode, number of layers, a list of sizes for each layer and a list of activations. Note if there are n layers, there needs to be n sizes and n-1 activations provided.
>Parameters: 
> - `nn`: Pointer to simpleNN structure to be initialized
> - `mode`: type of network i.e. classification/regression mode
> - `nlayers`: Number of layers in the neural network   
> - `sizes`: Array with sizes for each layer starting from the input layer and up to the output layer. Note number of sizes is one less than the number of layers.      
> - `acts`: Array with list of activation types for each layer other than the input layer. Note number of activation types is one less than the number of layers.        
     

/// @brief Free the simpleNN structure    
/// @param nn Pointer to simpleNN structure to be freed    
`void free_simpleNN(simpleNNType *nn);`   

/// @brief Run inference on a list of inputs   
/// @param nn Pointer to simpleNN structure      
/// @param inputs list of input samples as a 2D array     
/// @param n number of input samples       
/// @param outputs list of outputs generated for each sample as a 2D array. Space must be pre-allocated for the outputs.   
`void do_inference(simpleNNType *nn, float **inputs, int n, float **outputs);`    

/// @brief Train a sample   
/// @param nn Pointer to simpleNN structure   
/// @param train_data training data   
/// @param train_sample number of traning samples   
/// @param learning_rate learning rate for training    
/// @param max_epochs number of epochs    
/// @param reset whether to reset to random model in the beginning   
`void do_training(simpleNNType *nn, float **train_data, int train_sample, float learning_rate, int max_epochs, int reset);`   

/// @brief    
/// @param nn Pointer to simpleNN structure   
/// @param model_filename filename to save model to    
`void save_model_to_file(simpleNNType *nn, char * model_filename);`   

/// @brief     
/// @param nn Pointer to simpleNN structure   
/// @param model_filename filename to load model from   
`void load_model_from_file(simpleNNType *nn, char * model_filename);`   





