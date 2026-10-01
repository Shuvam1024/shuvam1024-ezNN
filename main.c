#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ezNN.h"
#include "readwrite_csv.h"

static int str_to_int_array(char *str, int *arr, int maxn) {
    int i = 0;
    char *token = strtok(str, ",");
    while (token) {
        char *end = NULL;
        long v;
        if (i >= maxn) return -1;
        v = strtol(token, &end, 10);
        if (end == token || (end && *end != '\0')) return -1;
        arr[i++] = (int)v;
        token = strtok(NULL, ",");
    }
    return i;
}

static int str_to_act_array(char *str, actType *arr, int maxn) {
    int i = 0;
    char *token = strtok(str, ",");
    while (token) {
        if (i >= maxn) return -1;
        if (strcmp(token, "ACT_IDENTITY") == 0) arr[i] = ACT_IDENTITY;
        else if (strcmp(token, "ACT_RELU") == 0) arr[i] = ACT_RELU;
        else if (strcmp(token, "ACT_SIGMOID") == 0) arr[i] = ACT_SIGMOID;
        else if (strcmp(token, "ACT_TANH") == 0) arr[i] = ACT_TANH;
        else if (strcmp(token, "ACT_SOFTMAX") == 0) arr[i] = ACT_SOFTMAX;
        else return -1;
        i++;
        token = strtok(NULL, ",");
    }
    return i;
}

static void usage_and_exit(char *prog, const char *message) {
    fprintf(stderr, "%s: %s\n\n", prog ? prog : "runezNN", message);
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  runezNN <mode> <layers> <activations> <learning_rate> <epochs>\n");
    fprintf(stderr, "          <training_csv[,testing_csv]> <model_filename> [column_map]\n\n");
    fprintf(stderr, "Modes: MODE_REGRESSION_L2, MODE_REGRESSION_L1,\n");
    fprintf(stderr, "       MODE_BINARY_CLASSIFICATION, MODE_MULTICAT_CLASSIFICATION\n");
    fprintf(stderr, "Activations: ACT_IDENTITY, ACT_RELU, ACT_TANH, ACT_SIGMOID, ACT_SOFTMAX\n");
    fprintf(stderr, "Environment: EZNN_SEED (unsigned) fixes init and shuffle;\n");
    fprintf(stderr, "             EZNN_VERBOSE=0 hides per-epoch training lines.\n");
    exit(1);
}

static void free_matrix(float **m, int rows) {
    int i;
    if (!m) return;
    for (i = 0; i < rows; i++) free(m[i]);
    free(m);
}

static float **alloc_matrix(int rows, int cols) {
    int i;
    float **m = (float **)calloc((size_t)rows, sizeof(float *));
    if (!m) return NULL;
    for (i = 0; i < rows; i++) {
        m[i] = (float *)calloc((size_t)cols, sizeof(float));
        if (!m[i]) {
            free_matrix(m, i);
            return NULL;
        }
    }
    return m;
}

static void print_split_metrics(ezNNType *nn, float **data, int n, const char *label) {
    int nout = nn->layer_sizes[nn->nlayers - 1];
    float **outputs = alloc_matrix(n, nout);
    if (!outputs) {
        fprintf(stderr, "ezNN: out of memory\n");
        return;
    }
    do_inference(nn, data, n, outputs);
    if (nn->mode == MODE_REGRESSION_L2) {
        printf("%s loss: %f\n", label, get_regression_l2_loss(nn, data, n, outputs));
    } else if (nn->mode == MODE_REGRESSION_L1) {
        printf("%s loss: %f\n", label, get_regression_l1_loss(nn, data, n, outputs));
    } else if (nn->mode == MODE_BINARY_CLASSIFICATION) {
        printf("%s loss: %f, accuracy: %f\n", label,
               get_binary_classification_loss(nn, data, n, outputs),
               get_binary_classification_accuracy(nn, data, n, outputs));
    } else {
        printf("%s loss: %f, accuracy: %f\n", label,
               get_multicat_classification_loss(nn, data, n, outputs),
               get_multicat_classification_accuracy(nn, data, n, outputs));
    }
    free_matrix(outputs, n);
}

int main(int argc, char *argv[]) {
    ezNNType myNN;
    modeType mode;
    int layer_sizes[MAX_LAYERS];
    actType layer_activations[MAX_LAYERS];
    int nlayers, n_acts, epochs, cols = 0, num_samples, features, csv_map_size, i, j;
    int num_test_samples = 0;
    float learning_rate;
    char *training_csv, *testing_csv = NULL, *model_filename;
    int *csv_map = NULL;
    float **train_csv = NULL, **train_data = NULL;
    float **test_csv = NULL, **test_data = NULL;
    const char *seed_env, *verb_env;

    memset(&myNN, 0, sizeof(myNN));
    if (argc < 8 || argc > 9) usage_and_exit(argv[0], "Wrong number of arguments");

    if (strcmp(argv[1], "MODE_REGRESSION_L2") == 0) mode = MODE_REGRESSION_L2;
    else if (strcmp(argv[1], "MODE_REGRESSION_L1") == 0) mode = MODE_REGRESSION_L1;
    else if (strcmp(argv[1], "MODE_BINARY_CLASSIFICATION") == 0) mode = MODE_BINARY_CLASSIFICATION;
    else if (strcmp(argv[1], "MODE_MULTICAT_CLASSIFICATION") == 0) mode = MODE_MULTICAT_CLASSIFICATION;
    else usage_and_exit(argv[0], "Unknown mode");

    nlayers = str_to_int_array(argv[2], layer_sizes, MAX_LAYERS);
    if (nlayers < 2) usage_and_exit(argv[0], "Need at least an input and an output layer");
    for (i = 0; i < nlayers; i++) {
        if (layer_sizes[i] <= 0 || layer_sizes[i] > 65535)
            usage_and_exit(argv[0], "Layer sizes must be in 1..65535");
    }

    n_acts = str_to_act_array(argv[3], layer_activations, MAX_LAYERS);
    if (n_acts != nlayers - 1)
        usage_and_exit(argv[0], "Number of activations should be num layers - 1");
    if (mode == MODE_MULTICAT_CLASSIFICATION && layer_activations[nlayers - 2] != ACT_SOFTMAX)
        usage_and_exit(argv[0], "Multiclass classification requires ACT_SOFTMAX on the output layer");
    if (mode == MODE_BINARY_CLASSIFICATION && layer_activations[nlayers - 2] != ACT_SIGMOID)
        usage_and_exit(argv[0], "Binary classification requires ACT_SIGMOID on the output layer");

    learning_rate = (float)atof(argv[4]);
    epochs = atoi(argv[5]);
    if (epochs < 0) usage_and_exit(argv[0], "Epoch count must be >= 0");

    training_csv = strtok(argv[6], ",");
    testing_csv = strtok(NULL, ",");
    if (!training_csv || !training_csv[0]) usage_and_exit(argv[0], "Missing training csv");
    model_filename = argv[7];

    num_samples = read_csv_size(training_csv, &cols);
    if (num_samples < 0) return 1;
    if (num_samples == 0) {
        fprintf(stderr, "ezNN: training csv has no rows\n");
        return 1;
    }

    csv_map_size = layer_sizes[0] + (mode == MODE_MULTICAT_CLASSIFICATION ? 1 : layer_sizes[nlayers - 1]);
    csv_map = (int *)malloc((size_t)csv_map_size * sizeof(int));
    if (!csv_map) return 1;
    for (i = 0; i < csv_map_size; ++i) csv_map[i] = i;
    if (argc == 9) {
        if (str_to_int_array(argv[8], csv_map, csv_map_size) != csv_map_size) {
            free(csv_map);
            usage_and_exit(argv[0], "Wrong column map size");
        }
    }
    for (i = 0; i < csv_map_size; i++) {
        if (csv_map[i] < 0 || csv_map[i] >= cols) {
            fprintf(stderr, "ezNN: column map index %d is outside 0..%d\n", csv_map[i], cols - 1);
            free(csv_map);
            return 1;
        }
    }

    seed_env = getenv("EZNN_SEED");
    if (seed_env && seed_env[0]) ezNN_seed((unsigned)strtoul(seed_env, NULL, 10));
    verb_env = getenv("EZNN_VERBOSE");
    if (verb_env && strcmp(verb_env, "0") == 0) ezNN_set_verbose(0);

    init_ezNN(&myNN, mode, nlayers, layer_sizes, layer_activations);
    if (myNN.nlayers != nlayers) {
        free(csv_map);
        return 1;
    }

    train_csv = alloc_matrix(num_samples, cols);
    if (!train_csv || read_csv(training_csv, num_samples, cols, train_csv) != 0) {
        free_matrix(train_csv, num_samples);
        free_ezNN(&myNN);
        free(csv_map);
        return 1;
    }
    features = get_num_features(&myNN);
    train_data = alloc_matrix(num_samples, features);
    if (!train_data) {
        free_matrix(train_csv, num_samples);
        free_ezNN(&myNN);
        free(csv_map);
        return 1;
    }
    for (i = 0; i < num_samples; ++i) {
        for (j = 0; j < features; ++j) train_data[i][j] = train_csv[i][csv_map[j]];
    }
    free_matrix(train_csv, num_samples);
    train_csv = NULL;

    printf("Starting Training with %s\n", training_csv);
    do_training(&myNN, train_data, num_samples, learning_rate, epochs, 1);
    if (getenv("EZNN_VERBOSE") && strcmp(getenv("EZNN_VERBOSE"), "0") == 0 && num_samples > 0)
        print_split_metrics(&myNN, train_data, num_samples, "train");

    save_model_to_file(&myNN, model_filename);
    printf("Saved model to file: %s\n", model_filename);
    free_matrix(train_data, num_samples);
    train_data = NULL;
    free_ezNN(&myNN);

    if (testing_csv && testing_csv[0]) {
        printf("-------------------\n");
        load_model_from_file(&myNN, model_filename);
        if (myNN.nlayers != nlayers) {
            fprintf(stderr, "ezNN: failed to reload %s\n", model_filename);
            free(csv_map);
            return 1;
        }
        printf("Loaded model from file: %s\n", model_filename);
        num_test_samples = read_csv_size(testing_csv, &cols);
        if (num_test_samples < 0) {
            free_ezNN(&myNN);
            free(csv_map);
            return 1;
        }
        test_csv = alloc_matrix(num_test_samples, cols);
        test_data = alloc_matrix(num_test_samples, features);
        if (!test_csv || !test_data || read_csv(testing_csv, num_test_samples, cols, test_csv) != 0) {
            free_matrix(test_csv, num_test_samples);
            free_matrix(test_data, num_test_samples);
            free_ezNN(&myNN);
            free(csv_map);
            return 1;
        }
        for (i = 0; i < num_test_samples; ++i) {
            for (j = 0; j < features; ++j) {
                if (csv_map[j] >= cols) {
                    fprintf(stderr, "ezNN: test csv has fewer columns than the column map\n");
                    free_matrix(test_csv, num_test_samples);
                    free_matrix(test_data, num_test_samples);
                    free_ezNN(&myNN);
                    free(csv_map);
                    return 1;
                }
                test_data[i][j] = test_csv[i][csv_map[j]];
            }
        }
        free_matrix(test_csv, num_test_samples);
        printf("Testing with %s\n", testing_csv);
        print_split_metrics(&myNN, test_data, num_test_samples, "test");
        free_matrix(test_data, num_test_samples);
        free_ezNN(&myNN);
    }

    free(csv_map);
    return 0;
}
