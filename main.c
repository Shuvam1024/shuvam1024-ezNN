#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "simpleNN.h"
#include "readwrite_csv.h"

int str_to_int_array(char *str, int *arr) {
    int i;
    char *token = strtok(str, ",");
    for (i = 0; token != NULL; i++) {
        arr[i] = atoi(token);
        token = strtok(NULL, ",");
    }
    return i;
}

int str_to_act_array(char *str, actType *arr) {
    int i;
    char *token = strtok(str, ",");
    for (i = 0; token != NULL; i++) {
        if (strcmp(token, "ACT_IDENTITY") == 0) arr[i] = ACT_IDENTITY;
        else if (strcmp(token, "ACT_RELU") == 0) arr[i] = ACT_RELU;
        else if (strcmp(token, "ACT_SIGMOID") == 0) arr[i] = ACT_SIGMOID;
        else if (strcmp(token, "ACT_TANH") == 0) arr[i] = ACT_TANH;
        else if (strcmp(token, "ACT_SOFTMAX") == 0) arr[i] = ACT_SOFTMAX;
        token = strtok(NULL, ",");
    }
    return i;
}

void usage_and_exit(char *prog, char *message) {
    printf("%s: %s\n", prog, message);
    printf("\n");
    printf("Usage:\n");
    printf("%s\n", prog);
    printf("    <mode>\n");
    printf("    <list_of_layers>\n");
    printf("    <list_of_activations>\n");
    printf("    <learning_rates>\n");
    printf("    <num_epochs>\n");
    printf("    <training_csv>[,<testing_csv>]\n");
    printf("    <model_filename>\n");
    printf("    [<map_of_columns_in_csv>]\n");
    exit(1);
}

int main(int argc, char* argv[]) {

    simpleNNType myNN;
    int num_samples, cols;

    if (argc < 8) {
        usage_and_exit(argv[0], "Wrong number of arguments");
    }

    // Parse mode
    modeType mode;
    if (strcmp(argv[1], "MODE_REGRESSION_L2") == 0) mode = MODE_REGRESSION_L2;
    else if (strcmp(argv[1], "MODE_REGRESSION_L1") == 0) mode = MODE_REGRESSION_L1;
    else if (strcmp(argv[1], "MODE_BINARY_CLASSIFICATION") == 0) mode = MODE_BINARY_CLASSIFICATION;
    else if (strcmp(argv[1], "MODE_MULTICAT_CLASSIFICATION") == 0) mode = MODE_MULTICAT_CLASSIFICATION;
    else usage_and_exit(argv[0], "Unknown mode");

    // Parse layer sizes
    int layer_sizes[MAX_LAYERS];
    int nlayers = str_to_int_array(argv[2], layer_sizes);
    
    // Parse activation functions
    int n_acts = nlayers - 1;
    actType layer_activations[MAX_LAYERS];
    if (str_to_act_array(argv[3], layer_activations) != n_acts)
        usage_and_exit(argv[0], "Number of activations should be num layers - 1");
    if (mode == MODE_MULTICAT_CLASSIFICATION && layer_activations[nlayers-2] != ACT_SOFTMAX)
        usage_and_exit(argv[0], "For multicategory classification lst layer must have ACT_SOFTMAX activation");
    if (mode == MODE_BINARY_CLASSIFICATION && layer_activations[nlayers-2] != ACT_SIGMOID)
        usage_and_exit(argv[0], "For multicategory classification lst layer must have ACT_SIGMOID activation");

    // Parse learning rate and epochs
    float learning_rate = atof(argv[4]);
    int epochs = atoi(argv[5]);

    char *training_csv = strtok(argv[6], ",");
    char *testing_csv = strtok(NULL, ",");
    num_samples = read_csv_size(training_csv, &cols);

    int csv_map_size = layer_sizes[0] + (mode == MODE_MULTICAT_CLASSIFICATION ? 1 : layer_sizes[nlayers - 1]);
    int *csv_map = malloc(csv_map_size * sizeof(int));
    for (int i = 0; i < csv_map_size; ++i) csv_map[i] = i;
    if (argc == 9) {
        if (str_to_int_array(argv[8], csv_map) != csv_map_size)
            usage_and_exit(argv[0], "Wrong map size");
    }

    init_simpleNN(&myNN, mode, nlayers, layer_sizes, layer_activations);

    float **train_csv_data = (float **)malloc(num_samples * sizeof(float *));
    for (int i = 0; i < num_samples; ++i) {
        train_csv_data[i] = (float *)malloc(cols * sizeof(float));
    }
    read_csv(training_csv, num_samples, cols, train_csv_data);

    int features = get_num_features(&myNN);
    float **train_data = (float **)malloc(num_samples * sizeof(float *));
    for (int i = 0; i < num_samples; ++i) {
        train_data[i] = (float *)malloc(features * sizeof(float));
        for (int j = 0; j < features; ++j) {
            train_data[i][j] = train_csv_data[i][csv_map[j]];
        }
    }
    for (int i = 0; i < num_samples; ++i) {
        free(train_csv_data[i]);
    }
    free(train_csv_data);

    printf("Starting Training with %s\n", training_csv);
    int reset = 1;
    do_training(&myNN, train_data, num_samples, learning_rate, epochs, reset);

    for (int i = 0; i < num_samples; ++i) {
        free(train_data[i]);
    }
    free(train_data);

    char * model_filename = argv[7];

    save_model_to_file(&myNN, model_filename);

    printf("Saved model to file: %s \n", model_filename);

    free_simpleNN(&myNN);

    if (testing_csv) {
        printf("-------------------\n");
        load_model_from_file(&myNN, model_filename);
        printf("Loaded model from file: %s \n", model_filename);  
        int num_test_samples = read_csv_size(testing_csv, &cols);
        float **test_csv_data = (float **)malloc(num_test_samples * sizeof(float *));
        for (int i = 0; i < num_test_samples; ++i) {
            test_csv_data[i] = (float *)malloc(cols * sizeof(float));
        }
        read_csv(testing_csv, num_test_samples, cols, test_csv_data);
        float **test_data = (float **)malloc(num_test_samples * sizeof(float *));
        for (int i = 0; i < num_test_samples; ++i) {
            test_data[i] = (float *)malloc(features * sizeof(float));
            for (int j = 0; j < features; ++j) {
                test_data[i][j] = test_csv_data[i][csv_map[j]];
            }
        }
        for (int i = 0; i < num_test_samples; ++i) {
            free(test_csv_data[i]);
        }
        free(test_csv_data);

        printf("Testing with %s\n", testing_csv);
        float **outputs = (float **)malloc(num_test_samples * sizeof(float *));
        for (int i = 0; i < num_test_samples; ++i) {
            outputs[i] = (float *)malloc(myNN.layer_sizes[myNN.nlayers-1] * sizeof(float));
        }
        do_inference(&myNN, test_data, num_test_samples, outputs);
        // metrics
        if (myNN.mode == MODE_REGRESSION_L2) {
            printf("loss: %f \n", get_regression_l2_loss(&myNN, test_data, num_test_samples, outputs));
        } else if (myNN.mode == MODE_REGRESSION_L1) {
            printf("loss: %f \n", get_regression_l1_loss(&myNN, test_data, num_test_samples, outputs));
        } else if (myNN.mode == MODE_BINARY_CLASSIFICATION) {
            printf("loss: %f, accuracy: %f \n", get_binary_classification_loss(&myNN, test_data, num_test_samples, outputs), get_binary_classification_accuracy(&myNN, test_data, num_test_samples, outputs));
        } else {
            printf("loss: %f, accuracy: %f \n", get_multicat_classification_loss(&myNN, test_data, num_test_samples, outputs), get_multicat_classification_accuracy(&myNN, test_data, num_test_samples, outputs));
        }

        for (int i = 0; i < num_test_samples; ++i) {
            free(outputs[i]);
        }
        free(outputs);
        free_simpleNN(&myNN);
    }
    
    free(csv_map);

    return 0;
}
