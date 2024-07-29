#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ezNN.h"
#include "time.h"

// Initialize the neural network
void init_ezNN(ezNNType *nn, modeType mode, int nlayers, int *sizes, actType *acts) {
    // Set the number of layers and the mode
    nn->nlayers = nlayers;
    nn->mode = mode;

    // Copy layer sizes
    memcpy(nn->layer_sizes, sizes, nlayers * sizeof(int));

    // Set activation functions
    memcpy(nn->activations, acts, (nlayers - 1) * sizeof(actType));

    // Allocate memory for weights, biases, and their gradients
    for (int i = 0; i < nlayers - 1; i++) {
        nn->weights[i] = (float **)malloc(sizes[i] * sizeof(float *));
        nn->weights_grad[i] = (float **)malloc(sizes[i] * sizeof(float *));
        for (int j = 0; j < sizes[i]; j++) {
            nn->weights[i][j] = (float *)malloc(sizes[i+1] * sizeof(float));
            nn->weights_grad[i][j] = (float *)malloc(sizes[i+1] * sizeof(float));
        }
        nn->biases[i] = (float *)malloc(sizes[i+1] * sizeof(float));
        nn->biases_grad[i] = (float *)malloc(sizes[i+1] * sizeof(float));
    }

    // Allocate memory for activations, errors, and gradients
    for (int i = 0; i < nlayers; i++) {
        nn->act_inputs[i] = (float *)malloc(sizes[i] * sizeof(float));
        nn->act_outputs[i] = (float *)malloc(sizes[i] * sizeof(float));
        nn->error[i] = (float *)malloc(sizes[i] * sizeof(float));
        nn->grad[i] = (float *)malloc(sizes[i] * sizeof(float));
    }
}

// Free the allocated memory for the neural network
void free_ezNN(ezNNType *nn) {
    for (int i = 0; i < nn->nlayers - 1; i++) {
        for (int j = 0; j < nn->layer_sizes[i]; j++) {
            free(nn->weights_grad[i][j]);
            free(nn->weights[i][j]);
        }
        free(nn->weights_grad[i]);
        free(nn->biases_grad[i]);
    }
    for (int i = 0; i < nn->nlayers; i++) {
        free(nn->error[i]);
        free(nn->grad[i]);
        free(nn->act_inputs[i]);
        free(nn->act_outputs[i]);
    }
    // Zero out the structure to avoid dangling pointers
    memset(nn, 0, sizeof(*nn));
}

// Matrix multiplication
static void matmul(float *x, int inlen, int outlen, float **W, float *b, float *y) {
    // Perform the matrix multiplication y = Wx + b
    for (int i = 0; i < outlen; i++) {
        y[i] = b[i];
        for (int j = 0; j < inlen; ++j)
            y[i] += x[j] * W[j][i];
    }
}

// Activation functions
static void identity(float *x, int n, float *y) {
    // Identity activation function: y = x
    for (int k = 0; k < n; ++k) {
        y[k] = x[k];
    }
}

static void relu(float *x, int n, float *y) {
    // ReLU activation function: y = max(0, x)
    for (int k = 0; k < n; ++k) {
        y[k] = x[k] < 0.0 ? 0.0 : x[k];
    }
}

static void tan_h(float *x, int n, float *y) {
    // Tanh activation function: y = tanh(x)
    for (int k = 0; k < n; ++k) {
        y[k] = tanh(x[k]);
    }
}

static void sigmoid(float *x, int n, float *y) {
    // Sigmoid activation function: y = 1 / (1 + exp(-x))
    for (int k = 0; k < n; ++k) {
        y[k] = 1.0 / (1.0 + exp(-x[k]));
    }
}

static void softmax(float *x, int n, float *y) {
    // Softmax activation function: y_i = exp(x_i) / sum(exp(x_j))
    float denom = 0.0;
    for (int i = 0; i < n; ++i) {
        y[i] = exp(x[i]);
        denom += y[i];
    }
    float onebydenom = 1.0 / denom;
    for (int i = 0; i < n; ++i) y[i] *= onebydenom;
}

// Forward pass through the network
static void forward_pass(ezNNType *nn, float *x, float *y) {
    // Set the input layer activations
    for (int i = 0; i < nn->layer_sizes[0]; ++i) {
        nn->act_inputs[0][i] = nn->act_outputs[0][i] = x[i];
    }
    // Propagate through the hidden layers
    for (int i = 0; i < nn->nlayers - 1; ++i) {
        matmul(nn->act_outputs[i], nn->layer_sizes[i], nn->layer_sizes[i+1], nn->weights[i], nn->biases[i], nn->act_inputs[i+1]);
        switch (nn->activations[i]) {
            case ACT_IDENTITY:
                identity(nn->act_inputs[i+1], nn->layer_sizes[i+1], nn->act_outputs[i+1]);
                break;
            case ACT_RELU:
                relu(nn->act_inputs[i+1], nn->layer_sizes[i+1], nn->act_outputs[i+1]);
                break;
            case ACT_SIGMOID:
                sigmoid(nn->act_inputs[i+1], nn->layer_sizes[i+1], nn->act_outputs[i+1]);
                break;
            case ACT_TANH:
                tan_h(nn->act_inputs[i+1], nn->layer_sizes[i+1], nn->act_outputs[i+1]);
                break;
            case ACT_SOFTMAX:
                softmax(nn->act_inputs[i+1], nn->layer_sizes[i+1], nn->act_outputs[i+1]);
        }
    }
    // Copy the final output layer activations to the output array if provided
    if (y) {
        memcpy(y, nn->act_outputs[nn->nlayers-1], nn->layer_sizes[nn->nlayers-1] * sizeof(float));
    }
}

// Gradients of activation functions
static void gradient_identity(float *in, float *out, int n, float *error, float *grad) {
    // Gradient of identity activation function: 1
    for (int k = 0; k < n; ++k) {
        grad[k] = error[k];
    }
}

static void gradient_relu(float *in, float *out, int n, float *error, float *grad) {
    // Gradient of ReLU activation function: 0 if x < 0, 1 if x > 0
    for (int k = 0; k < n; ++k) {
        float deriv = 0.5;
        if (in[k] < 0.0) 
            deriv = 0.0;
        else if (in[k] > 0.0) 
            deriv = 1.0;
        grad[k] = error[k] * deriv;
    }
}

static void gradient_tanh(float *in, float *out, int n, float *error, float *grad) {
    // Gradient of tanh activation function: 1 - y^2
    for (int k = 0; k < n; ++k) {
        grad[k] = (1.0 - out[k] * out[k]) * error[k];
    }
}

static void gradient_sigmoid(float *in, float *out, int n, float *error, float *grad) {
    // Gradient of sigmoid activation function: y * (1 - y)
    for (int k = 0; k < n; ++k) {
        grad[k] = (1.0 - out[k]) * out[k] * error[k];
    }
}

static void gradient_softmax(float *in, float *out, int n, float *error, float *grad) {
    // Gradient of softmax activation function: y_i * (1 - y_i) for i == j, -y_i * y_j for i != j
    for (int k = 0; k < n; ++k) {
        grad[k] = 0.0;
        for (int j = 0; j < n; ++j) {
            grad[k] += error[j] * ((j == k) ? out[k] * (1.0 - out[k]) : -out[k] * out[j]);
        }
    }
}
#define USE_CLASSIFICATION_SHORTCUT 0
// Calculate partial derivative of the loss function w.r.t. the output activations of each layer.
// For the last layer, it depends directly on the specific loss function. 
// For other layers it is calculated by propagation through the next linear layer
static void calc_error(ezNNType *nn, int layer_number, float *expected_out) {
    int layer_size = nn->layer_sizes[layer_number];
    if (layer_number == nn->nlayers-1) {  // last layer
#if USE_CLASSIFICATION_SHORTCUT
        for (int i = 0; i < layer_size; i++) {
            nn->error[layer_number][i] = nn->act_outputs[layer_number][i] - expected_out[i];
        }
#else
        switch (nn->mode) {
            case MODE_MULTICAT_CLASSIFICATION:
                for (int i = 0; i < layer_size; i++) {
                    nn->error[layer_number][i] =  -expected_out[i] / nn->act_outputs[layer_number][i];
                }
                break;
            case MODE_BINARY_CLASSIFICATION:
                for (int i = 0; i < layer_size; i++) {
                    nn->error[layer_number][i] = (nn->act_outputs[layer_number][i] - expected_out[i]) / nn->act_outputs[layer_number][i] / (1.0 - nn->act_outputs[layer_number][i]);
                }
            case MODE_REGRESSION_L2:
            default:
                for (int i = 0; i < layer_size; i++) {
                    nn->error[layer_number][i] = nn->act_outputs[layer_number][i] - expected_out[i];
                }
                break;
            case MODE_REGRESSION_L1:
                for (int i = 0; i < layer_size; i++) {
                    nn->error[layer_number][i] = (nn->act_outputs[layer_number][i] - expected_out[i]) >= 0 ? 1.0 : -1.0;
                }
                break;
        };
#endif
    } else {  // non-last layers
        for (int i = 0; i < layer_size; ++i) {
            nn->error[layer_number][i] = 0.0;
            for (int k = 0; k < nn->layer_sizes[layer_number + 1]; ++k) {
                nn->error[layer_number][i] += nn->weights[layer_number][i][k] * nn->grad[layer_number + 1][k];
            }
        }

    }
}

// Calculate partial derivative of the loss function w.r.t. the input activations of each layer, given the partial derivatives w.r.t. the output activtions.
static void calc_grad(ezNNType *nn, int i) {
    int layer_size = nn->layer_sizes[i];
    actType activation = nn->activations[i - 1];
#if USE_CLASSIFICATION_SHORTCUT
    if (i == nn->nlayers-1 && nn->mode == MODE_MULTICAT_CLASSIFICATION && activation == ACT_SOFTMAX) {
        activation = ACT_IDENTITY;
    }
    
    if (i == nn->nlayers-1 && nn->mode == MODE_BINARY_CLASSIFICATION && activation == ACT_SIGMOID) {
        activation = ACT_IDENTITY;
    }
#endif
    switch (activation) {
        case ACT_IDENTITY:
            gradient_identity(nn->act_inputs[i], nn->act_outputs[i], layer_size, nn->error[i], nn->grad[i]);
            break;
        case ACT_RELU:
            gradient_relu(nn->act_inputs[i], nn->act_outputs[i], layer_size, nn->error[i], nn->grad[i]);
            break;
        case ACT_SIGMOID:
            gradient_sigmoid(nn->act_inputs[i], nn->act_outputs[i], layer_size, nn->error[i], nn->grad[i]);
            break;
        case ACT_TANH:
            gradient_tanh(nn->act_inputs[i], nn->act_outputs[i], layer_size, nn->error[i], nn->grad[i]);
            break;
        case ACT_SOFTMAX:
            gradient_softmax(nn->act_inputs[i], nn->act_outputs[i], layer_size, nn->error[i], nn->grad[i]);
            break;
    }
}

// Calculate partial derivatives of the loss function w.r.t. the weights and biases of each layer
static void calc_param_grads(ezNNType *nn, int i) {
    for (int j = 0; j < nn->layer_sizes[i]; ++j) {
        for (int k = 0; k < nn->layer_sizes[i-1]; ++k) {
            nn->weights_grad[i-1][k][j] = nn->grad[i][j] * nn->act_outputs[i-1][k];
        }
        nn->biases_grad[i-1][j] = nn->grad[i][j];
    }
}

// Update weights and biases using gradients and learning rate
static void update_params(ezNNType *nn, float learning_rate) {
    int nlayers = nn->nlayers;
    for (int i = 0; i < nlayers - 1; i++) {
        for (int j = 0; j < nn->layer_sizes[i]; j++) {
            for (int k = 0; k < nn->layer_sizes[i+1]; k++) {
                nn->weights[i][j][k] -= nn->weights_grad[i][j][k] * learning_rate;
            }

        }
        for (int k = 0; k < nn->layer_sizes[i+1]; k++) {
            nn->biases[i][k] -= nn->biases_grad[i][k] * learning_rate;
        }
    } 
}

void do_inference(ezNNType *nn, float **inputs, int n, float **outputs) {
    for (int i = 0; i < n; ++i) {
        forward_pass(nn, inputs[i], outputs[i]);
    }
}

static void hard_binary_classify(float *p, int n, int *hard) {
    for (int i = 0; i < n; ++i) hard[i] = (p[i] < 0.5 ? 0 : 1);
}

static int hard_multicat_classify(float *p, int n) {
    int max_index = 0;
    float max = p[0];
    for (int i = 0; i < n; ++i) {
        if (p[i] > max) {
            max = p[i];
            max_index = i;
        }
    }
    return max_index;
}

void do_classification_hard(ezNNType *nn, float **inputs, int n, int **outputs) {
    float *outs = malloc(nn->layer_sizes[nn->nlayers-1] * sizeof(float));
    for (int i = 0; i < n; ++i) {
        forward_pass(nn, inputs[i], outs);
        if (nn->mode == MODE_BINARY_CLASSIFICATION) {
            hard_binary_classify(outs, n, outputs[i]);
        }
        if (nn->mode == MODE_MULTICAT_CLASSIFICATION) {
            outputs[i][0] = hard_multicat_classify(outs, n);
        }
    }
    free(outs);
}

void do_regression_hard(ezNNType *nn, float **inputs, int n, float **outputs) {
    for (int i = 0; i < n; ++i) {
        forward_pass(nn, inputs[i], outputs[i]);
    }
}

static void back_propogation(ezNNType *nn, float * data_train, float *expected_out, float learning_rate) {
    for (int i = nn->nlayers - 1; i >= 1; --i) {
        calc_error(nn, i, expected_out);
        calc_grad(nn, i);
        calc_param_grads(nn, i);
    }
    update_params(nn, learning_rate);
}

static void one_hot_encode(int cl, int num_classes, float *hot) {
    memset(hot, 0, num_classes * sizeof(float));
    hot[cl] = 1.0;
}

static void initialize_random_params(ezNNType *nn) {
    srand(time(NULL));
    for (int i = 0; i < nn->nlayers - 1; i++) {
        float epsilon = sqrt(1.0 / (nn->layer_sizes[i] * nn->layer_sizes[i + 1]));
        for (int j = 0; j < nn->layer_sizes[i]; j++) {
            for (int k = 0; k < nn->layer_sizes[i + 1]; k++) {
                nn->weights[i][j][k] = (float)rand() / (float)RAND_MAX * 2 * epsilon - epsilon;
            }
        }
        for (int j = 0; j < nn->layer_sizes[i + 1]; j++) {
            nn->biases[i][j] = (float)rand() / (float)RAND_MAX * 2 * epsilon - epsilon;
        }
    }
}

float get_regression_l2_loss(ezNNType *nn, float **train_data, int train_samples, float **outputs) {
    double sse = 0.0; 

    for (int i = 0; i < train_samples; i++) {
        for (int j = 0; j < nn->layer_sizes[nn->nlayers - 1]; j++) {
            float expected = train_data[i][nn->layer_sizes[0] + j];
            float actual = outputs[i][j];
            float diff = (expected - actual);
            sse += diff*diff;
        }
    }

    float mse = sse / train_samples; 
    return mse;
}

float get_regression_l1_loss(ezNNType *nn, float **train_data, int train_samples, float **outputs) {
    double sae = 0.0; 
    for (int i = 0; i < train_samples; i++) {
        for (int j = 0; j < nn->layer_sizes[nn->nlayers - 1]; j++) {
            float expected = train_data[i][nn->layer_sizes[0] + j];
            float actual = outputs[i][j];
            float diff = fabs(expected - actual);
            sae += diff;
        }
    }
    float mae = sae / train_samples; 
    return mae;
}

float get_binary_classification_loss(ezNNType *nn, float **train_data, int train_samples, float **outputs) {
    double entropy = 0.0;
    for (int i = 0; i < train_samples; i++) {
        for (int j = 0; j < nn->layer_sizes[nn->nlayers - 1]; j++) {
            float expected = train_data[i][nn->layer_sizes[0] + j];
            float actual = outputs[i][j];
            if (expected) {
                entropy -= log(actual);
            }
        }
    }
    entropy /= train_samples;
    return (float) entropy;
}

float get_multicat_classification_loss(ezNNType *nn, float **train_data, int train_samples, float **outputs) {
    double entropy = 0.0;
    for (int i = 0; i < train_samples; i++) {
        int expected = (int)train_data[i][nn->layer_sizes[0]];
        float actual = outputs[i][(int)expected];
        entropy -= log(actual);
    }
    entropy /= train_samples;
    return (float) entropy;
}

float get_binary_classification_accuracy(ezNNType *nn, float **train_data, int train_samples, float **outputs) {
    int correct = 0;
    for (int i = 0; i < train_samples; i++) {
        for (int j = 0; j < nn->layer_sizes[nn->nlayers - 1]; j++) {
            float expected = train_data[i][nn->layer_sizes[0] + j];
            float actual = outputs[i][j];
            int predicted = (actual >= 0.5) ? 1 : 0;
            if (predicted == (int)expected) {
                correct++;
            }
        }
    }
    return (float)correct / (train_samples * nn->layer_sizes[nn->nlayers - 1]) * 100.0;
}


float get_multicat_classification_accuracy(ezNNType *nn, float **train_data, int train_samples, float **outputs) {
    int correct = 0;
    for (int i = 0; i < train_samples; i++) {
        int expected = (int)train_data[i][nn->layer_sizes[0]];
        int predicted = hard_multicat_classify(outputs[i], nn->layer_sizes[nn->nlayers - 1]);
        if (predicted == expected) {
            correct++;
        }
    }
    return (float)correct / train_samples * 100.0;
}

void save_model_to_file(ezNNType *nn, char *model_filename) {
    FILE *fp = fopen(model_filename, "wb");
    unsigned char byte;
    byte = nn->mode + (nn->nlayers << 3);
    fwrite(&byte, 1, 1, fp);
    for (int i = 0; i < nn->nlayers; ++i) {
        byte = nn->layer_sizes[i] & 255;
        fwrite(&byte, 1, 1, fp);
        byte = (nn->layer_sizes[i] >> 8) & 255;
        fwrite(&byte, 1, 1, fp);
    }
    for (int i = 0; i < nn->nlayers - 1; i += 2) {
        byte = nn->activations[i];
        if (i < nn->nlayers - 2)
            byte += (nn->activations[i + 1] << 4);
        fwrite(&byte, 1, 1, fp);
    }
    for (int i = 0; i < nn->nlayers - 1; i++) {
        for (int j = 0; j < nn->layer_sizes[i]; j++) {
            fwrite(nn->weights[i][j], nn->layer_sizes[i + 1], sizeof(float), fp);
        }
        fwrite(nn->biases[i], nn->layer_sizes[i + 1], sizeof(float), fp);
    }
    fclose(fp);
}

void load_model_from_file(ezNNType *nn, char *model_filename) {
    FILE *fp = fopen(model_filename, "rb");
    unsigned char byte;
    fread(&byte, 1, 1, fp);
    modeType mode = (modeType)(byte & 7);
    int nlayers = byte >> 3;
    int sizes[MAX_LAYERS];
    for (int i = 0; i < nlayers; ++i) {
        fread(&byte, 1, 1, fp);
        sizes[i] = byte;
        fread(&byte, 1, 1, fp);
        sizes[i] += (byte << 8);
    }
    actType activations[MAX_LAYERS];
    for (int i = 0; i < nlayers - 1; i += 2) {
        fread(&byte, 1, 1, fp);
        activations[i] = byte & 15;
        if (i < nlayers - 2) {
            activations[i + 1] = byte >> 4;
        }
    }
    init_ezNN(nn, mode, nlayers, sizes, activations);
    for (int i = 0; i < nn->nlayers - 1; i++) {
        for (int j = 0; j < nn->layer_sizes[i]; j++) {
            fread(nn->weights[i][j], nn->layer_sizes[i + 1], sizeof(float), fp);
        }
        fread(nn->biases[i], nn->layer_sizes[i + 1], sizeof(float), fp);
    }
    fclose(fp);
}

int get_num_features(ezNNType *nn) {
    return nn->layer_sizes[0] + (nn->mode == MODE_MULTICAT_CLASSIFICATION ? 1 : nn->layer_sizes[nn->nlayers - 1]);
}

int get_num_out_features(ezNNType *nn) {
    return nn->mode == MODE_MULTICAT_CLASSIFICATION ? 1 : nn->layer_sizes[nn->nlayers - 1];
}

static void random_shuffle(int n, int *indices) {
    for (int i = 0; i < n; ++i) indices[i] = i;
    for (int k = n - 1; k >= 1; --k) {
        int r = rand() % (k + 1);
        if (k!=r) {
            int t = indices[k];
            indices[k] = indices[r];
            indices[r] = t;
        }
    }
}

void do_training(ezNNType *nn, float **train_data, int train_samples, float learning_rate, int max_epochs, int reset) {
    float *out;
    if (nn->mode == MODE_MULTICAT_CLASSIFICATION)
        out = malloc(nn->layer_sizes[nn->nlayers-1] * sizeof(float));
    if (reset) {
        // initialize random weights and biases
        initialize_random_params(nn);
    }
    int *indices = (int *)malloc(train_samples * sizeof(int));
    float **outputs = (float **)malloc(train_samples * sizeof(float *));
    for (int i = 0; i < train_samples; ++i) {
        outputs[i] = (float *)malloc(nn->layer_sizes[nn->nlayers-1] * sizeof(float));
    }
    for (int epoch = 0; epoch < max_epochs; ++epoch) {
        random_shuffle(train_samples, indices);
        for (int sample = 0; sample < train_samples; ++sample) {
            float *this_sample = train_data[indices[sample]];
            if (nn->mode == MODE_MULTICAT_CLASSIFICATION) {
                int cl = (int)this_sample[nn->layer_sizes[0]];
                one_hot_encode(cl, nn->layer_sizes[nn->nlayers - 1], out);
            } else {
                out = this_sample + nn->layer_sizes[0];
            }
            forward_pass(nn, this_sample, NULL);
            back_propogation(nn, this_sample, out, learning_rate);
        }
        
        do_inference(nn, train_data, train_samples, outputs);

        // metrics
        printf("epoch: %d  ", epoch);
        if (nn->mode == MODE_REGRESSION_L2) {
            printf("loss: %f \r", get_regression_l2_loss(nn, train_data, train_samples, outputs));
        } else if (nn->mode == MODE_REGRESSION_L1) {
            printf("loss: %f \r", get_regression_l1_loss(nn, train_data, train_samples, outputs));
        } else if (nn->mode == MODE_BINARY_CLASSIFICATION) {
            printf("loss: %f, accuracy: %f \r", get_binary_classification_loss(nn, train_data, train_samples, outputs), get_binary_classification_accuracy(nn, train_data, train_samples, outputs));
        } else {
            printf("loss: %f, accuracy: %f \r", get_multicat_classification_loss(nn, train_data, train_samples, outputs), get_multicat_classification_accuracy(nn, train_data, train_samples, outputs));
        }   
    }
    printf("\n");
    for (int i = 0; i < train_samples; ++i) {
        free(outputs[i]);
    }    
    free(outputs);

    free(indices);
    if (nn->mode == MODE_MULTICAT_CLASSIFICATION)
        free(out);
}


    