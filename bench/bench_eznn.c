#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#include "ezNN.h"
#include "readwrite_csv.h"

#ifdef EZNN_OPENMP
#include <omp.h>
#endif

/* Wide synthetic net used only for the timing comparison. */
enum {
    MICRO_IN = 768,
    MICRO_H = 768,
    MICRO_OUT = 128,
    MICRO_SAMPLES = 64,
    MICRO_EPOCHS = 8,
    MICRO_TRIALS = 3
};

#ifdef __clang__
#define BENCH_COMPILER "clang " __clang_version__
#elif defined(__GNUC__)
#define BENCH_COMPILER "gcc " __VERSION__
#else
#define BENCH_COMPILER "unknown"
#endif

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double median_sorted_pick(double *v, int n) {
    double tmp[16];
    int i, j;
    if (n > 16) n = 16;
    for (i = 0; i < n; i++) tmp[i] = v[i];
    for (i = 1; i < n; i++) {
        double key = tmp[i];
        j = i;
        while (j > 0 && tmp[j - 1] > key) {
            tmp[j] = tmp[j - 1];
            j--;
        }
        tmp[j] = key;
    }
    return tmp[n / 2];
}

static void die(const char *msg) {
    fprintf(stderr, "bench: %s\n", msg);
    exit(1);
}

static int parse_ints(const char *s, int *out, int maxn) {
    char buf[256];
    char *tok;
    int n = 0;
    snprintf(buf, sizeof buf, "%s", s);
    tok = strtok(buf, ",");
    while (tok) {
        char *end = NULL;
        long v;
        if (n >= maxn) return -1;
        v = strtol(tok, &end, 10);
        if (end == tok || (end && *end != '\0')) return -1;
        out[n++] = (int)v;
        tok = strtok(NULL, ",");
    }
    return n;
}

static int parse_acts(const char *s, actType *out, int maxn) {
    char buf[256];
    char *tok;
    int n = 0;
    snprintf(buf, sizeof buf, "%s", s);
    tok = strtok(buf, ",");
    while (tok) {
        if (n >= maxn) return -1;
        if (strcmp(tok, "ACT_IDENTITY") == 0) out[n] = ACT_IDENTITY;
        else if (strcmp(tok, "ACT_RELU") == 0) out[n] = ACT_RELU;
        else if (strcmp(tok, "ACT_SIGMOID") == 0) out[n] = ACT_SIGMOID;
        else if (strcmp(tok, "ACT_TANH") == 0) out[n] = ACT_TANH;
        else if (strcmp(tok, "ACT_SOFTMAX") == 0) out[n] = ACT_SOFTMAX;
        else return -1;
        n++;
        tok = strtok(NULL, ",");
    }
    return n;
}

static float **load_csv(const char *path, int *rows, int *cols) {
    int n, c = 0, i;
    float **m;
    n = read_csv_size((char *)path, &c);
    if (n <= 0 || c <= 0) return NULL;
    m = (float **)calloc((size_t)n, sizeof(float *));
    if (!m) return NULL;
    for (i = 0; i < n; i++) {
        m[i] = (float *)calloc((size_t)c, sizeof(float));
        if (!m[i]) return NULL;
    }
    if (read_csv((char *)path, n, c, m) != 0) return NULL;
    *rows = n;
    *cols = c;
    return m;
}

static void free_matrix(float **m, int rows) {
    int i;
    if (!m) return;
    for (i = 0; i < rows; i++) free(m[i]);
    free(m);
}

static void evaluate(ezNNType *nn, float **data, int n, const char *prefix) {
    int nout = nn->layer_sizes[nn->nlayers - 1];
    int i;
    float **outs = (float **)calloc((size_t)n, sizeof(float *));
    if (!outs) die("out of memory");
    for (i = 0; i < n; i++) {
        outs[i] = (float *)calloc((size_t)nout, sizeof(float));
        if (!outs[i]) die("out of memory");
    }
    do_inference(nn, data, n, outs);
    if (nn->mode == MODE_REGRESSION_L2) {
        printf("%s_loss %.8f\n", prefix, get_regression_l2_loss(nn, data, n, outs));
    } else if (nn->mode == MODE_REGRESSION_L1) {
        printf("%s_loss %.8f\n", prefix, get_regression_l1_loss(nn, data, n, outs));
    } else if (nn->mode == MODE_BINARY_CLASSIFICATION) {
        printf("%s_loss %.8f\n", prefix, get_binary_classification_loss(nn, data, n, outs));
        printf("%s_accuracy %.8f\n", prefix, get_binary_classification_accuracy(nn, data, n, outs));
    } else {
        printf("%s_loss %.8f\n", prefix, get_multicat_classification_loss(nn, data, n, outs));
        printf("%s_accuracy %.8f\n", prefix, get_multicat_classification_accuracy(nn, data, n, outs));
    }
    free_matrix(outs, n);
}

static void run_task(const char *name, modeType mode, const char *layers, const char *acts,
                     const char *lr_s, const char *epochs_s, const char *train_path,
                     const char *test_path, unsigned seed, int trials) {
    int sizes[MAX_LAYERS];
    actType activations[MAX_LAYERS];
    int nlayers, nacts, epochs, ntrain = 0, ntest = 0, cols = 0, tcols = 0, need;
    float lr;
    float **train, **test;
    ezNNType nn;
    char train_key[64], test_key[64];

    memset(&nn, 0, sizeof nn);
    if (trials < 1 || trials > 16) die("trials must be in 1..16");
    nlayers = parse_ints(layers, sizes, MAX_LAYERS);
    nacts = parse_acts(acts, activations, MAX_LAYERS);
    if (nlayers < 2 || nacts != nlayers - 1) die("bad architecture string");
    lr = (float)atof(lr_s);
    epochs = atoi(epochs_s);
    if (epochs < 1) die("epochs must be positive");
    train = load_csv(train_path, &ntrain, &cols);
    test = load_csv(test_path, &ntest, &tcols);
    if (!train || !test) die("failed to read csv");
    if (tcols != cols) die("train/test column mismatch");
    init_ezNN(&nn, mode, nlayers, sizes, activations);
    if (nn.nlayers != nlayers) die("init failed");
    need = get_num_features(&nn);
    if (cols != need) {
        fprintf(stderr, "bench: %s has %d columns, network expects %d\n", train_path, cols, need);
        exit(1);
    }
    printf("%s_arch %s\n", name, layers);
    printf("%s_activations %s\n", name, acts);
    printf("%s_lr %s\n", name, lr_s);
    printf("%s_epochs %d\n", name, epochs);
    printf("%s_seed %u\n", name, seed);
    printf("%s_train_rows %d\n", name, ntrain);
    printf("%s_test_rows %d\n", name, ntest);
    printf("%s_trials %d\n", name, trials);

    ezNN_set_verbose(0);
    ezNN_seed(seed);
    {
        double t0 = now_sec();
        do_training(&nn, train, ntrain, lr, epochs, 1);
        double t1 = now_sec();
        printf("%s_warmup_seconds %.8f\n", name, t1 - t0);
    }
    {
        double samples[16];
        int i;
        printf("%s_trial_seconds", name);
        for (i = 0; i < trials; i++) {
            double t0, t1;
            ezNN_seed(seed);
            t0 = now_sec();
            do_training(&nn, train, ntrain, lr, epochs, 1);
            t1 = now_sec();
            samples[i] = t1 - t0;
            printf(" %.8f", samples[i]);
        }
        printf("\n");
        printf("%s_median_seconds %.8f\n", name, median_sorted_pick(samples, trials));
    }
    snprintf(train_key, sizeof train_key, "%s_train", name);
    snprintf(test_key, sizeof test_key, "%s_test", name);
    evaluate(&nn, train, ntrain, train_key);
    evaluate(&nn, test, ntest, test_key);
    free_ezNN(&nn);
    free_matrix(train, ntrain);
    free_matrix(test, ntest);
}

static uint32_t g_xs = 0xC0FFEEu;

static float next_sym(void) {
    g_xs ^= g_xs << 13;
    g_xs ^= g_xs >> 17;
    g_xs ^= g_xs << 5;
    return ((g_xs >> 8) * (1.f / 16777216.f)) * 2.f - 1.f;
}

static void run_micro(unsigned seed) {
    int sizes[3] = {MICRO_IN, MICRO_H, MICRO_OUT};
    actType acts[2] = {ACT_RELU, ACT_IDENTITY};
    int feat = MICRO_IN + MICRO_OUT;
    int s, i, t;
    float **data;
    ezNNType nn;
    double samples[MICRO_TRIALS];
    double t0, t1;

    memset(&nn, 0, sizeof nn);
    data = (float **)calloc(MICRO_SAMPLES, sizeof(float *));
    if (!data) die("out of memory");
    for (s = 0; s < MICRO_SAMPLES; s++) {
        data[s] = (float *)calloc((size_t)feat, sizeof(float));
        if (!data[s]) die("out of memory");
        for (i = 0; i < feat; i++) data[s][i] = next_sym();
    }
    init_ezNN(&nn, MODE_REGRESSION_L2, 3, sizes, acts);
    if (nn.nlayers != 3) die("micro init failed");
    ezNN_set_verbose(0);
    printf("micro_arch %d,%d,%d\n", MICRO_IN, MICRO_H, MICRO_OUT);
    printf("micro_activations ACT_RELU,ACT_IDENTITY\n");
    printf("micro_mode MODE_REGRESSION_L2\n");
    printf("micro_samples %d\n", MICRO_SAMPLES);
    printf("micro_epochs %d\n", MICRO_EPOCHS);
    printf("micro_lr 0.001\n");
    printf("micro_seed %u\n", seed);
    printf("micro_trials %d\n", MICRO_TRIALS);
    printf("micro_note synthetic_inputs_timing_only\n");
    ezNN_seed(seed);
    t0 = now_sec();
    do_training(&nn, data, MICRO_SAMPLES, 0.001f, MICRO_EPOCHS, 1);
    t1 = now_sec();
    printf("micro_warmup_seconds %.8f\n", t1 - t0);
    printf("micro_trial_seconds");
    for (t = 0; t < MICRO_TRIALS; t++) {
        ezNN_seed(seed);
        t0 = now_sec();
        do_training(&nn, data, MICRO_SAMPLES, 0.001f, MICRO_EPOCHS, 1);
        t1 = now_sec();
        samples[t] = t1 - t0;
        printf(" %.8f", samples[t]);
    }
    printf("\n");
    printf("micro_median_seconds %.8f\n", median_sorted_pick(samples, MICRO_TRIALS));
    {
        int nout = MICRO_OUT;
        float **outs = (float **)calloc(MICRO_SAMPLES, sizeof(float *));
        float loss;
        for (s = 0; s < MICRO_SAMPLES; s++) outs[s] = (float *)calloc((size_t)nout, sizeof(float));
        do_inference(&nn, data, MICRO_SAMPLES, outs);
        loss = get_regression_l2_loss(&nn, data, MICRO_SAMPLES, outs);
        printf("micro_loss_finite %d\n", isfinite(loss) ? 1 : 0);
        free_matrix(outs, MICRO_SAMPLES);
    }
    free_ezNN(&nn);
    free_matrix(data, MICRO_SAMPLES);
}

static void usage(void) {
    fprintf(stderr,
            "usage: bench_eznn seed trials iris_layers iris_acts iris_lr iris_epochs iris_train iris_test reg_layers reg_acts reg_lr reg_epochs reg_train reg_test\n");
    exit(1);
}

int main(int argc, char **argv) {
    unsigned seed;
    int trials;
    if (argc != 15) usage();
    seed = (unsigned)strtoul(argv[1], NULL, 10);
    trials = atoi(argv[2]);
    printf("compiler %s\n", BENCH_COMPILER);
#ifdef EZNN_OPENMP
    printf("openmp 1\n");
    printf("openmp_max_threads %d\n", omp_get_max_threads());
#else
    printf("openmp 0\n");
#endif
    printf("timing do_training_wall_seconds\n");
    printf("timing_includes init_when_reset\n");
    printf("timing_excludes dataset_pass_used_only_for_printing\n");
    run_task("iris", MODE_MULTICAT_CLASSIFICATION, argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], seed, trials);
    run_task("regression", MODE_REGRESSION_L2, argv[9], argv[10], argv[11], argv[12], argv[13], argv[14], seed, trials);
    run_micro(seed);
    return 0;
}
