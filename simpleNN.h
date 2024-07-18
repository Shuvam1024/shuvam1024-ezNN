#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_LAYERS 16
typedef enum {
    ACT_IDENTITY,
    ACT_RELU,
    ACT_TANH,
    ACT_SIGMOID,
    ACT_SOFTMAX
} actType;

typedef enum {
    MODE_REGRESSION_L2,
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

} simpleNNType;


/// @brief Initializes the simpleNN structure with mode, number of layers, a list of sizes for each layer and a list of activations. Note if there are n layers, there needs to be n sizes and n-1 activations provided.
/// @param nn Pointer to simpleNN structure to be initialized
/// @param nlayers Number of layers in the neural network
/// @param sizes List of sizes for each layer starting from the input layer and up to the output layer
/// @param acts List of activation types for each layer other than the input layer
void init_simpleNN(simpleNNType *nn, modeType mode, int nlayers, int *sizes, actType *acts);

/// @brief Free the simpleNN structure
/// @param nn Pointer to simpleNN structure to be freed
void free_simpleNN(simpleNNType *nn);

/// @brief Run inference on a list of inputs
/// @param nn Pointer to simpleNN structure
/// @param inputs list of input samples as a 2D array
/// @param n number of input samples
/// @param outputs list of outputs generated for each sample as a 2D array. Space must be pre-allocated for the outputs.
void do_inference(simpleNNType *nn, float **inputs, int n, float **outputs);

/// @brief 
/// @param nn 
/// @param inputs 
/// @param n 
/// @param outputs 
void do_classification_hard(simpleNNType *nn, float **inputs, int n, int **outputs);

void do_regression_hard(simpleNNType *nn, float **inputs, int n, float **outputs);

void do_training(simpleNNType *nn, float **train_data, int train_sample, float learning_rate, int max_epochs);

int get_num_features(simpleNNType *nn);
int get_num_out_features(simpleNNType *nn);

float get_multicat_classification_loss(simpleNNType *nn, float **train_data, int train_samples, float **outputs);
float get_binary_classification_loss(simpleNNType *nn, float **train_data, int train_samples, float **outputs);
float get_regression_l1_loss(simpleNNType *nn, float **train_data, int train_samples, float **outputs);
float get_regression_l2_loss(simpleNNType *nn, float **train_data, int train_samples, float **outputs);
float get_binary_classification_accuracy(simpleNNType *nn, float **train_data, int train_samples, float **outputs);
float get_multicat_classification_accuracy(simpleNNType *nn, float **train_data, int train_samples, float **outputs);
