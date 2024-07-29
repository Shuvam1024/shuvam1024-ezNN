#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_LAYERS 16
typedef enum {
    ACT_IDENTITY = 0,
    ACT_RELU,
    ACT_TANH,
    ACT_SIGMOID,
    ACT_SOFTMAX
} actType;

typedef enum {
    MODE_REGRESSION_L2 = 0,
    MODE_REGRESSION_L1,
    MODE_BINARY_CLASSIFICATION,
    MODE_MULTICAT_CLASSIFICATION
} modeType;

typedef struct {
    modeType mode;
    int nlayers;
    int layer_sizes[MAX_LAYERS];
    float **weights[MAX_LAYERS - 1];
    float *biases[MAX_LAYERS - 1];
    float **weights_grad[MAX_LAYERS - 1];
    float *biases_grad[MAX_LAYERS - 1];
    actType activations[MAX_LAYERS - 1];
    float *act_inputs[MAX_LAYERS];
    float *act_outputs[MAX_LAYERS];
    float *error[MAX_LAYERS];
    float *grad[MAX_LAYERS];

} ezNNType;


/// @brief Initializes the ezNN structure with mode, number of layers, a list of sizes for each layer and a list of activations. Note if there are n layers, there needs to be n sizes and n-1 activations provided.
/// @param nn Pointer to ezNN structure to be initialized
/// @param nlayers Number of layers in the neural network
/// @param sizes List of sizes for each layer starting from the input layer and up to the output layer
/// @param acts List of activation types for each layer other than the input layer
void init_ezNN(ezNNType *nn, modeType mode, int nlayers, int *sizes, actType *acts);

/// @brief Free the ezNN structure
/// @param nn Pointer to ezNN structure to be freed
void free_ezNN(ezNNType *nn);

/// @brief Run inference on a list of inputs
/// @param nn Pointer to ezNN structure
/// @param inputs list of input samples as a 2D array
/// @param n number of input samples
/// @param outputs list of outputs generated for each sample as a 2D array. Space must be pre-allocated for the outputs.
void do_inference(ezNNType *nn, float **inputs, int n, float **outputs);

/// @brief 
/// @param nn Pointer to ezNN structure
/// @param inputs list of input samples as a 2D array
/// @param n number of input samples
/// @param outputs list of outputs generated for each sample as a 2D array. Space must be pre-allocated for the outputs.
void do_classification_hard(ezNNType *nn, float **inputs, int n, int **outputs);

/// @brief 
/// @param nn Pointer to ezNN structure
/// @param inputs list of input samples as a 2D array
/// @param n number of input samples
/// @param outputs list of outputs generated for each sample as a 2D array. Space must be pre-allocated for the outputs.
void do_regression_hard(ezNNType *nn, float **inputs, int n, float **outputs);

/// @brief Train a sample
/// @param nn Pointer to ezNN structure
/// @param train_data training data
/// @param train_sample number of traning samples
/// @param learning_rate learning rate for training
/// @param max_epochs number of epochs
/// @param reset whether to reset to random model in the beginning
void do_training(ezNNType *nn, float **train_data, int train_sample, float learning_rate, int max_epochs, int reset);

/// @brief returns number of total feautures in a sample
/// @param nn Pointer to ezNN structure
/// @return integer number of features
int get_num_features(ezNNType *nn);

/// @brief returns number of output feautures in a sample
/// @param nn Pointer to ezNN structure
/// @return integer number of output features
int get_num_out_features(ezNNType *nn);

/// @brief Calculates the loss for each sample
/// @param nn Pointer to ezNN structure
/// @param train_data training data
/// @param train_samples number of traning samples
/// @param outputs the list of outputs
/// @return a float number for loss
float get_multicat_classification_loss(ezNNType *nn, float **train_data, int train_samples, float **outputs);

/// @brief Calculates the loss for each sample
/// @param nn Pointer to ezNN structure
/// @param train_data training data
/// @param train_samples number of traning samples
/// @param outputs the list of outputs
/// @return a float number for loss
float get_binary_classification_loss(ezNNType *nn, float **train_data, int train_samples, float **outputs);

/// @brief Calculates the loss for each sample
/// @param nn Pointer to ezNN structure
/// @param train_data training data
/// @param train_samples number of traning samples
/// @param outputs the list of outputs
/// @return a float number for loss
float get_regression_l1_loss(ezNNType *nn, float **train_data, int train_samples, float **outputs);

/// @brief Calculates the loss for each sample
/// @param nn Pointer to ezNN structure
/// @param train_data training data
/// @param train_samples number of traning samples
/// @param outputs the list of outputs
/// @return a float number for loss
float get_regression_l2_loss(ezNNType *nn, float **train_data, int train_samples, float **outputs);

/// @brief Calculates the loss for each sample
/// @param nn Pointer to ezNN structure
/// @param train_data training data
/// @param train_samples number of traning samples
/// @param outputs the list of outputs
/// @return a float number for loss
float get_binary_classification_accuracy(ezNNType *nn, float **train_data, int train_samples, float **outputs);

/// @brief Calculates the loss for each sample
/// @param nn Pointer to ezNN structure
/// @param train_data training data
/// @param train_samples number of traning samples
/// @param outputs the list of outputs
/// @return a float number for loss
float get_multicat_classification_accuracy(ezNNType *nn, float **train_data, int train_samples, float **outputs);

/// @brief 
/// @param nn Pointer to ezNN structure
/// @param model_filename filename to save model to
void save_model_to_file(ezNNType *nn, char * model_filename);

/// @brief 
/// @param nn Pointer to ezNN structure
/// @param model_filename filename to load model from
void load_model_from_file(ezNNType *nn, char * model_filename);