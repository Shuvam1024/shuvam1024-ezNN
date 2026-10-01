#ifndef EZNN_H
#define EZNN_H

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

/* One network. Weights are stored as weights[layer][in][out]:
   z[out] = bias[out] + sum_in activation[in] * weights[layer][in][out].
   The struct layout is part of the ABI; new controls live in functions
   below rather than in new fields. Internal buffers (act_*, error, grad)
   hold a single sample, so one network is not reentrant. */
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

/* Initializes nn. nlayers is the number of layers including input and
   output, so there are nlayers sizes and nlayers-1 activations.
   Weights and biases start at zero. On invalid arguments nn is left
   unchanged and a message is written to stderr. */
void init_ezNN(ezNNType *nn, modeType mode, int nlayers, int *sizes, actType *acts);

/* Releases every allocation owned by nn and zeroes the struct.
   Safe on a zeroed struct and safe to call twice. */
void free_ezNN(ezNNType *nn);

/* Writes the output-layer activation for each of the n samples.
   outputs[i] must hold layer_sizes[nlayers-1] floats, including in
   multiclass mode (those are class scores, not a class index). */
void do_inference(ezNNType *nn, float **inputs, int n, float **outputs);

/* Hard decisions. Binary: one 0/1 per output unit, threshold 0.5
   (ties go to 1). Multiclass: outputs[i][0] is the argmax class.
   outputs[i] must hold one int per output unit in binary mode, and
   at least one int in multiclass mode. */
void do_classification_hard(ezNNType *nn, float **inputs, int n, int **outputs);

/* Same numeric result as do_inference. Kept for existing callers. */
void do_regression_hard(ezNNType *nn, float **inputs, int n, float **outputs);

/* Online SGD for max_epochs passes. Each row of train_data is
   inputs followed by targets: one target per output unit, or a single
   class index in multiclass mode. reset != 0 draws a fresh init.
   See ezNN_seed and ezNN_set_verbose. */
void do_training(ezNNType *nn, float **train_data, int train_sample, float learning_rate, int max_epochs, int reset);

/* Input columns plus target columns in one training row. */
int get_num_features(ezNNType *nn);

/* Target columns in one training row (1 in multiclass mode). */
int get_num_out_features(ezNNType *nn);

/* Mean over samples of the per-sample losses described in ezNN.c.
   Multiclass rows store a class index. Accuracy is a percentage in [0, 100]. */
float get_multicat_classification_loss(ezNNType *nn, float **train_data, int train_samples, float **outputs);
float get_binary_classification_loss(ezNNType *nn, float **train_data, int train_samples, float **outputs);
float get_regression_l1_loss(ezNNType *nn, float **train_data, int train_samples, float **outputs);
float get_regression_l2_loss(ezNNType *nn, float **train_data, int train_samples, float **outputs);
float get_binary_classification_accuracy(ezNNType *nn, float **train_data, int train_samples, float **outputs);
float get_multicat_classification_accuracy(ezNNType *nn, float **train_data, int train_samples, float **outputs);

/* Binary model file. Layer widths must fit in 16 bits. Native endian floats. */
void save_model_to_file(ezNNType *nn, char *model_filename);

/* Fills a zeroed or uninitialized nn. On failure nn is left zeroed.
   Do not pass a live network; its allocations would leak. */
void load_model_from_file(ezNNType *nn, char *model_filename);

/* Process-global seed for weight init and the training shuffle.
   A later do_training(..., reset != 0) replays this seed. If this was
   never called, that path still uses srand((unsigned)time(NULL)). */
void ezNN_seed(unsigned int seed);

/* Nonzero (the default) prints an epoch line to stdout from do_training.
   Zero skips the extra full-set forward pass used only for that line.
   Parameter updates do not depend on the flag. */
void ezNN_set_verbose(int enabled);

#endif
