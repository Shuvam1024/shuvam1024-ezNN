#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include <fcntl.h>
#include <unistd.h>

#include "ezNN.h"
#include "readwrite_csv.h"

#ifdef EZNN_OPENMP
#include <omp.h>
#endif

static int g_pass = 0;
static int g_fail = 0;

static void checkf_impl(int cond, const char *file, int line, const char *fmt, ...) {
    va_list ap;
    if (cond) {
        g_pass++;
        return;
    }
    g_fail++;
    fprintf(stderr, "FAIL %s:%d: ", file, line);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

#define CHECKF(cond, ...) checkf_impl(!!(cond), __FILE__, __LINE__, __VA_ARGS__)

static int silence(int fd) {
    int saved = dup(fd);
    int devnull = open("/dev/null", O_WRONLY);
    if (fd == STDOUT_FILENO) fflush(stdout);
    if (fd == STDERR_FILENO) fflush(stderr);
    if (saved >= 0 && devnull >= 0) dup2(devnull, fd);
    if (devnull >= 0) close(devnull);
    return saved;
}

static void restore(int fd, int saved) {
    if (fd == STDOUT_FILENO) fflush(stdout);
    if (fd == STDERR_FILENO) fflush(stderr);
    if (saved >= 0) {
        dup2(saved, fd);
        close(saved);
    }
}

static int near_enough(double a, double b, double atol, double rtol) {
    double diff, scale;
    if (isnan(a) || isnan(b) || isinf(a) || isinf(b)) return 0;
    diff = fabs(a - b);
    scale = fabs(a) > fabs(b) ? fabs(a) : fabs(b);
    return diff <= atol + rtol * scale;
}

static int must_init(ezNNType *nn, modeType mode, int n, int *sizes, actType *acts,
                     const char *file, int line) {
    int err;
    memset(nn, 0, sizeof(*nn));
    err = silence(STDERR_FILENO);
    init_ezNN(nn, mode, n, sizes, acts);
    restore(STDERR_FILENO, err);
    if (nn->nlayers != n) {
        checkf_impl(0, file, line, "init failed for nlayers=%d", n);
        return 0;
    }
    return 1;
}

#define MUST_INIT(nn, mode, n, sizes, acts) must_init((nn), (mode), (n), (sizes), (acts), __FILE__, __LINE__)

static void zero_params(ezNNType *nn) {
    int i, j, k;
    for (i = 0; i < nn->nlayers - 1; i++) {
        for (j = 0; j < nn->layer_sizes[i]; j++) {
            for (k = 0; k < nn->layer_sizes[i + 1]; k++) nn->weights[i][j][k] = 0.f;
        }
        for (k = 0; k < nn->layer_sizes[i + 1]; k++) nn->biases[i][k] = 0.f;
    }
}

/* Scalar differentiated by one SGD step. L2 is 1/2 sum (a-y)^2, matching
   backprop; get_regression_l2_loss is the sum of squares (n == 1). */
static float objective_one(ezNNType *nn, float *row) {
    int nout = nn->layer_sizes[nn->nlayers - 1];
    float *out = (float *)calloc((size_t)nout, sizeof(float));
    float *rows[1];
    float *outs[1];
    float loss = 0.f;
    rows[0] = row;
    outs[0] = out;
    do_inference(nn, rows, 1, outs);
    if (nn->mode == MODE_REGRESSION_L2) loss = 0.5f * get_regression_l2_loss(nn, rows, 1, outs);
    else if (nn->mode == MODE_REGRESSION_L1) loss = get_regression_l1_loss(nn, rows, 1, outs);
    else if (nn->mode == MODE_BINARY_CLASSIFICATION) loss = get_binary_classification_loss(nn, rows, 1, outs);
    else loss = get_multicat_classification_loss(nn, rows, 1, outs);
    free(out);
    return loss;
}

static void check_grads(const char *name, ezNNType *nn, float *row,
                        float h, double atol, double rtol, int expect_signal) {
    float *batch[1];
    float sentinel;
    int layer, j, k;
    double max_ana = 0.0;
    batch[0] = row;
    ezNN_set_verbose(0);
    sentinel = nn->weights[0][0][0];
    do_training(nn, batch, 1, 0.f, 1, 0);
    CHECKF(nn->weights[0][0][0] == sentinel, "%s: lr 0 changed a weight", name);
    for (layer = 0; layer < nn->nlayers - 1; layer++) {
        int in = nn->layer_sizes[layer];
        int out = nn->layer_sizes[layer + 1];
        for (j = 0; j < in; j++) {
            for (k = 0; k < out; k++) {
                float ana = nn->weights_grad[layer][j][k];
                float w = nn->weights[layer][j][k];
                float lp, lm;
                double num;
                if (fabs((double)ana) > max_ana) max_ana = fabs((double)ana);
                nn->weights[layer][j][k] = w + h;
                lp = objective_one(nn, row);
                nn->weights[layer][j][k] = w - h;
                lm = objective_one(nn, row);
                nn->weights[layer][j][k] = w;
                num = ((double)lp - (double)lm) / (2.0 * (double)h);
                CHECKF(near_enough(num, (double)ana, atol, rtol),
                       "%s W[%d][%d][%d] analytic %.8g numeric %.8g",
                       name, layer, j, k, (double)ana, num);
            }
        }
        for (k = 0; k < out; k++) {
            float ana = nn->biases_grad[layer][k];
            float b = nn->biases[layer][k];
            float lp, lm;
            double num;
            if (fabs((double)ana) > max_ana) max_ana = fabs((double)ana);
            nn->biases[layer][k] = b + h;
            lp = objective_one(nn, row);
            nn->biases[layer][k] = b - h;
            lm = objective_one(nn, row);
            nn->biases[layer][k] = b;
            num = ((double)lp - (double)lm) / (2.0 * (double)h);
            CHECKF(near_enough(num, (double)ana, atol, rtol),
                   "%s b[%d][%d] analytic %.8g numeric %.8g",
                   name, layer, k, (double)ana, num);
        }
    }
    if (expect_signal) CHECKF(max_ana > 1e-4, "%s: gradient vanished (max %.3g)", name, max_ana);
    else CHECKF(max_ana < 1e-5, "%s: expected a zero gradient, max %.3g", name, max_ana);
}

static int params_equal(const ezNNType *a, const ezNNType *b) {
    int i, j;
    if (a->nlayers != b->nlayers || a->mode != b->mode) return 0;
    for (i = 0; i < a->nlayers; i++) {
        if (a->layer_sizes[i] != b->layer_sizes[i]) return 0;
    }
    for (i = 0; i < a->nlayers - 1; i++) {
        int in = a->layer_sizes[i];
        int out = a->layer_sizes[i + 1];
        if (a->activations[i] != b->activations[i]) return 0;
        for (j = 0; j < in; j++) {
            if (memcmp(a->weights[i][j], b->weights[i][j], (size_t)out * sizeof(float)) != 0)
                return 0;
        }
        if (memcmp(a->biases[i], b->biases[i], (size_t)out * sizeof(float)) != 0) return 0;
    }
    return 1;
}

static void test_invalid_init(void) {
    ezNNType nn;
    int sizes[2] = {2, 1};
    actType bad = (actType)9;
    actType ok = ACT_IDENTITY;
    int err = silence(STDERR_FILENO);
    memset(&nn, 0, sizeof(nn));
    nn.nlayers = 42;
    init_ezNN(&nn, MODE_REGRESSION_L2, 1, sizes, &ok);
    CHECKF(nn.nlayers == 42, "short network must not be overwritten");
    init_ezNN(&nn, MODE_REGRESSION_L2, 2, NULL, &ok);
    CHECKF(nn.nlayers == 42, "null sizes must not be overwritten");
    sizes[1] = 0;
    init_ezNN(&nn, MODE_REGRESSION_L2, 2, sizes, &ok);
    CHECKF(nn.nlayers == 42, "zero width must not be overwritten");
    sizes[1] = 1;
    init_ezNN(&nn, MODE_REGRESSION_L2, 2, sizes, &bad);
    CHECKF(nn.nlayers == 42, "bad activation must not be overwritten");
    restore(STDERR_FILENO, err);
    free_ezNN(&nn);
    free_ezNN(&nn);
    CHECKF(nn.nlayers == 0, "free zeroes the struct");
}

static void test_zero_init_and_stable_activations(void) {
    ezNNType nn;
    int sizes2[2] = {1, 2};
    int sizes1[2] = {1, 1};
    actType soft = ACT_SOFTMAX;
    actType sig = ACT_SIGMOID;
    actType id = ACT_IDENTITY;
    float row[1] = {1.f};
    float *rows[1] = {row};
    float out[2];
    float *outs[1] = {out};

    if (!MUST_INIT(&nn, MODE_REGRESSION_L2, 2, sizes2, &soft)) return;
    do_inference(&nn, rows, 1, outs);
    CHECKF(near_enough(out[0], 0.5, 1e-6, 1e-6) && near_enough(out[1], 0.5, 1e-6, 1e-6),
           "softmax of zero logits is uniform, got %g %g", out[0], out[1]);
    nn.weights[0][0][0] = 1000.f;
    nn.weights[0][0][1] = 0.f;
    do_inference(&nn, rows, 1, outs);
    CHECKF(isfinite(out[0]) && isfinite(out[1]), "softmax overflow");
    CHECKF(out[0] > 0.999999f && out[1] < 1e-6f, "stable softmax peaked, got %g %g", out[0], out[1]);
    nn.weights[0][0][0] = 1000.f;
    nn.weights[0][0][1] = 1000.f;
    do_inference(&nn, rows, 1, outs);
    CHECKF(near_enough(out[0], 0.5, 1e-5, 1e-5) && near_enough(out[1], 0.5, 1e-5, 1e-5),
           "stable softmax common shift, got %g %g", out[0], out[1]);
    free_ezNN(&nn);

    if (!MUST_INIT(&nn, MODE_BINARY_CLASSIFICATION, 2, sizes1, &sig)) return;
    nn.biases[0][0] = 0.f;
    do_inference(&nn, rows, 1, outs);
    CHECKF(near_enough(out[0], 0.5, 1e-6, 1e-6), "sigmoid(0) = 0.5, got %g", out[0]);
    nn.biases[0][0] = 80.f;
    do_inference(&nn, rows, 1, outs);
    CHECKF(isfinite(out[0]) && out[0] > 0.999999f, "sigmoid(+80) got %g", out[0]);
    nn.biases[0][0] = -80.f;
    do_inference(&nn, rows, 1, outs);
    CHECKF(isfinite(out[0]) && out[0] < 1e-6f, "sigmoid(-80) got %g", out[0]);
    free_ezNN(&nn);

    if (!MUST_INIT(&nn, MODE_REGRESSION_L2, 2, sizes1, &id)) return;
    row[0] = 3.5f;
    do_inference(&nn, rows, 1, outs);
    CHECKF(out[0] == 0.f, "zero weights produce a zero identity output, got %g", out[0]);
    free_ezNN(&nn);
}

static void test_feature_counts(void) {
    ezNNType nn;
    int bin_sizes[3] = {4, 5, 2};
    int multi_sizes[3] = {4, 5, 3};
    actType bin_acts[2] = {ACT_RELU, ACT_SIGMOID};
    actType multi_acts[2] = {ACT_RELU, ACT_SOFTMAX};
    if (!MUST_INIT(&nn, MODE_BINARY_CLASSIFICATION, 3, bin_sizes, bin_acts)) return;
    CHECKF(get_num_features(&nn) == 6, "binary features");
    CHECKF(get_num_out_features(&nn) == 2, "binary targets");
    free_ezNN(&nn);
    if (!MUST_INIT(&nn, MODE_MULTICAT_CLASSIFICATION, 3, multi_sizes, multi_acts)) return;
    CHECKF(get_num_features(&nn) == 5, "multiclass features");
    CHECKF(get_num_out_features(&nn) == 1, "multiclass target column");
    free_ezNN(&nn);
}

static void test_loss_values(void) {
    ezNNType nn;
    int sizes[2] = {1, 2};
    actType id = ACT_IDENTITY;
    float row[3];
    float *rows[1] = {row};
    float out[2];
    float *outs[1] = {out};
    double bce;

    if (!MUST_INIT(&nn, MODE_REGRESSION_L2, 2, sizes, &id)) return;
    nn.weights[0][0][0] = 3.f;
    nn.weights[0][0][1] = -1.f;
    row[0] = 1.f;
    row[1] = 1.f;
    row[2] = 0.f;
    do_inference(&nn, rows, 1, outs);
    CHECKF(out[0] == 3.f && out[1] == -1.f, "identity forward");
    CHECKF(near_enough(get_regression_l2_loss(&nn, rows, 1, outs), 4.0 + 1.0, 1e-6, 1e-6),
           "L2 is sum of squares, got %g", get_regression_l2_loss(&nn, rows, 1, outs));
    nn.mode = MODE_REGRESSION_L1;
    CHECKF(near_enough(get_regression_l1_loss(&nn, rows, 1, outs), 2.0 + 1.0, 1e-6, 1e-6),
           "L1 got %g", get_regression_l1_loss(&nn, rows, 1, outs));
    free_ezNN(&nn);

    if (!MUST_INIT(&nn, MODE_BINARY_CLASSIFICATION, 2, sizes, &id)) return;
    nn.weights[0][0][0] = 0.25f;
    nn.weights[0][0][1] = 0.25f;
    row[0] = 1.f;
    row[1] = 1.f;
    row[2] = 0.f;
    do_inference(&nn, rows, 1, outs);
    bce = -(log(0.25) + log(0.75));
    CHECKF(near_enough(get_binary_classification_loss(&nn, rows, 1, outs), bce, 1e-5, 1e-5),
           "binary loss dropped the y=0 term? got %g want %g",
           get_binary_classification_loss(&nn, rows, 1, outs), bce);
    free_ezNN(&nn);

    if (!MUST_INIT(&nn, MODE_MULTICAT_CLASSIFICATION, 2, sizes, &id)) return;
    nn.weights[0][0][0] = 0.25f;
    nn.weights[0][0][1] = 0.5f;
    row[0] = 1.f;
    row[1] = 1.f; /* class 1 */
    do_inference(&nn, rows, 1, outs);
    CHECKF(near_enough(get_multicat_classification_loss(&nn, rows, 1, outs), -log(0.5), 1e-5, 1e-5),
           "multiclass loss got %g", get_multicat_classification_loss(&nn, rows, 1, outs));
    out[0] = 0.1f;
    out[1] = 0.9f;
    row[1] = 1.f;
    CHECKF(get_multicat_classification_accuracy(&nn, rows, 1, outs) == 100.f, "multiclass accuracy");
    out[0] = 0.95f;
    CHECKF(get_multicat_classification_accuracy(&nn, rows, 1, outs) == 0.f, "multiclass miss");
    free_ezNN(&nn);

    if (!MUST_INIT(&nn, MODE_BINARY_CLASSIFICATION, 2, sizes, &id)) return;
    row[0] = 0.f;
    row[1] = 1.f;
    row[2] = 0.f;
    out[0] = 0.8f;
    out[1] = 0.2f;
    CHECKF(get_binary_classification_accuracy(&nn, rows, 1, outs) == 100.f, "binary accuracy both right");
    out[1] = 0.9f;
    CHECKF(get_binary_classification_accuracy(&nn, rows, 1, outs) == 50.f, "binary accuracy one right");
    out[0] = 0.5f;
    row[1] = 1.f;
    CHECKF(get_binary_classification_accuracy(&nn, rows, 1, outs) == 50.f, "threshold 0.5 counts as class 1");
    free_ezNN(&nn);
}

static void test_exact_steps(void) {
    ezNNType nn;
    int sizes[2] = {1, 1};
    actType id = ACT_IDENTITY;
    float row[2];
    float *rows[1] = {row};

    if (!MUST_INIT(&nn, MODE_REGRESSION_L2, 2, sizes, &id)) return;
    nn.weights[0][0][0] = 0.5f;
    nn.biases[0][0] = 0.f;
    row[0] = 1.f;
    row[1] = 0.f; /* a = 0.5, dL/da = 0.5 for the half-square objective */
    ezNN_set_verbose(0);
    do_training(&nn, rows, 1, 0.25f, 1, 0);
    CHECKF(nn.weights[0][0][0] == 0.375f, "L2 step w got %g", nn.weights[0][0][0]);
    CHECKF(nn.biases[0][0] == -0.125f, "L2 step b got %g", nn.biases[0][0]);
    free_ezNN(&nn);

    if (!MUST_INIT(&nn, MODE_REGRESSION_L1, 2, sizes, &id)) return;
    nn.weights[0][0][0] = 0.5f;
    nn.biases[0][0] = 0.25f;
    row[0] = 1.f;
    row[1] = 0.75f; /* exact hit: subgradient of abs is +1 */
    do_training(&nn, rows, 1, 0.f, 1, 0);
    CHECKF(nn.biases_grad[0][0] == 1.f, "L1 sign(0) grad b got %g", nn.biases_grad[0][0]);
    CHECKF(nn.weights_grad[0][0][0] == 1.f, "L1 sign(0) grad w got %g", nn.weights_grad[0][0][0]);
    nn.weights[0][0][0] = 0.5f;
    nn.biases[0][0] = 0.f;
    row[1] = 0.f; /* a = 0.5, sign = +1 */
    do_training(&nn, rows, 1, 0.25f, 1, 0);
    CHECKF(nn.weights[0][0][0] == 0.25f, "L1 step w got %g", nn.weights[0][0][0]);
    CHECKF(nn.biases[0][0] == -0.25f, "L1 step b got %g", nn.biases[0][0]);
    free_ezNN(&nn);
}

static void test_all_grads(void) {
    ezNNType nn;
    const double atol = 2e-3;
    const double rtol = 2e-2;
    const float h = 1e-3f;

    /* Identity + L2 */
    {
        int sizes[2] = {2, 2};
        actType act = ACT_IDENTITY;
        float row[4] = {0.5f, -1.25f, 0.25f, -0.5f};
        if (!MUST_INIT(&nn, MODE_REGRESSION_L2, 2, sizes, &act)) return;
        nn.weights[0][0][0] = 0.4f; nn.weights[0][0][1] = -0.2f;
        nn.weights[0][1][0] = 0.1f; nn.weights[0][1][1] = 0.3f;
        nn.biases[0][0] = 0.05f; nn.biases[0][1] = -0.1f;
        check_grads("id-l2", &nn, row, h, atol, rtol, 1);
        free_ezNN(&nn);
    }
    /* ReLU in the positive region + L2 */
    {
        int sizes[2] = {2, 2};
        actType act = ACT_RELU;
        float row[4] = {1.f, 1.f, 0.2f, -0.4f};
        if (!MUST_INIT(&nn, MODE_REGRESSION_L2, 2, sizes, &act)) return;
        nn.weights[0][0][0] = 0.8f; nn.weights[0][0][1] = 0.7f;
        nn.weights[0][1][0] = 0.6f; nn.weights[0][1][1] = 0.9f;
        nn.biases[0][0] = 0.5f; nn.biases[0][1] = 0.4f;
        check_grads("relu-pos-l2", &nn, row, h, atol, rtol, 1);
        free_ezNN(&nn);
    }
    /* ReLU in the negative region: gradient is identically zero. */
    {
        int sizes[2] = {2, 2};
        actType act = ACT_RELU;
        float row[4] = {1.f, 1.f, 0.5f, -0.2f};
        if (!MUST_INIT(&nn, MODE_REGRESSION_L2, 2, sizes, &act)) return;
        nn.weights[0][0][0] = -0.8f; nn.weights[0][0][1] = -0.7f;
        nn.weights[0][1][0] = -0.6f; nn.weights[0][1][1] = -0.9f;
        nn.biases[0][0] = -0.5f; nn.biases[0][1] = -0.4f;
        check_grads("relu-neg-l2", &nn, row, h, 1e-5, 1e-5, 0);
        free_ezNN(&nn);
    }
    /* ReLU at the kink. Central difference approaches the 1/2 subgradient. */
    {
        int sizes[2] = {2, 2};
        actType act = ACT_RELU;
        float row[4] = {1.f, 0.f, -1.f, 1.f};
        if (!MUST_INIT(&nn, MODE_REGRESSION_L2, 2, sizes, &act)) return;
        zero_params(&nn);
        check_grads("relu-kink-l2", &nn, row, 1e-3f, 2e-3, 2e-2, 1);
        free_ezNN(&nn);
    }
    /* tanh + L2 */
    {
        int sizes[2] = {2, 2};
        actType act = ACT_TANH;
        float row[4] = {0.5f, -0.25f, 0.f, 0.2f};
        if (!MUST_INIT(&nn, MODE_REGRESSION_L2, 2, sizes, &act)) return;
        nn.weights[0][0][0] = 0.3f; nn.weights[0][0][1] = -0.2f;
        nn.weights[0][1][0] = 0.15f; nn.weights[0][1][1] = 0.25f;
        nn.biases[0][0] = 0.1f; nn.biases[0][1] = -0.05f;
        check_grads("tanh-l2", &nn, row, h, atol, rtol, 1);
        free_ezNN(&nn);
    }
    /* sigmoid derivative via L2, not the fused BCE path */
    {
        int sizes[2] = {2, 2};
        actType act = ACT_SIGMOID;
        float row[4] = {0.4f, -0.3f, 0.2f, 0.7f};
        if (!MUST_INIT(&nn, MODE_REGRESSION_L2, 2, sizes, &act)) return;
        nn.weights[0][0][0] = 0.2f; nn.weights[0][0][1] = -0.15f;
        nn.weights[0][1][0] = 0.1f; nn.weights[0][1][1] = 0.25f;
        nn.biases[0][0] = 0.05f; nn.biases[0][1] = -0.1f;
        check_grads("sigmoid-l2", &nn, row, h, atol, rtol, 1);
        free_ezNN(&nn);
    }
    /* softmax Jacobian via L2 */
    {
        int sizes[2] = {2, 3};
        actType act = ACT_SOFTMAX;
        float row[5] = {0.4f, -0.3f, 0.2f, 0.5f, 0.3f};
        if (!MUST_INIT(&nn, MODE_REGRESSION_L2, 2, sizes, &act)) return;
        nn.weights[0][0][0] = 0.2f; nn.weights[0][0][1] = -0.1f; nn.weights[0][0][2] = 0.15f;
        nn.weights[0][1][0] = -0.2f; nn.weights[0][1][1] = 0.25f; nn.weights[0][1][2] = 0.05f;
        nn.biases[0][0] = 0.1f; nn.biases[0][1] = -0.05f; nn.biases[0][2] = 0.0f;
        check_grads("softmax-l2", &nn, row, h, atol, rtol, 1);
        free_ezNN(&nn);
    }
    /* Explicit BCE gradient through an identity output (no sigmoid fuse). */
    {
        int sizes[2] = {1, 2};
        actType act = ACT_IDENTITY;
        float row[3] = {1.f, 1.f, 0.f};
        if (!MUST_INIT(&nn, MODE_BINARY_CLASSIFICATION, 2, sizes, &act)) return;
        nn.weights[0][0][0] = 0.3f;
        nn.weights[0][0][1] = 0.5f;
        nn.biases[0][0] = 0.1f;
        nn.biases[0][1] = 0.2f;
        check_grads("bce-identity", &nn, row, 1e-4f, 2e-3, 2e-2, 1);
        free_ezNN(&nn);
    }
    /* Fused sigmoid + BCE. */
    {
        int sizes[2] = {2, 2};
        actType act = ACT_SIGMOID;
        float row[4] = {0.5f, -0.25f, 1.f, 0.f};
        if (!MUST_INIT(&nn, MODE_BINARY_CLASSIFICATION, 2, sizes, &act)) return;
        nn.weights[0][0][0] = 0.3f; nn.weights[0][0][1] = -0.2f;
        nn.weights[0][1][0] = 0.15f; nn.weights[0][1][1] = 0.4f;
        nn.biases[0][0] = 0.1f; nn.biases[0][1] = -0.05f;
        check_grads("bce-sigmoid", &nn, row, 1e-3f, atol, rtol, 1);
        free_ezNN(&nn);
    }
    /* Explicit multiclass CE through identity. */
    {
        int sizes[2] = {2, 3};
        actType act = ACT_IDENTITY;
        float row[3] = {1.f, 0.f, 1.f};
        if (!MUST_INIT(&nn, MODE_MULTICAT_CLASSIFICATION, 2, sizes, &act)) return;
        nn.weights[0][0][0] = 0.2f; nn.weights[0][0][1] = 0.5f; nn.weights[0][0][2] = 0.3f;
        nn.weights[0][1][0] = 0.1f; nn.weights[0][1][1] = 0.1f; nn.weights[0][1][2] = 0.1f;
        check_grads("ce-identity", &nn, row, 1e-4f, atol, rtol, 1);
        free_ezNN(&nn);
    }
    /* Fused softmax + CE. */
    {
        int sizes[2] = {2, 3};
        actType act = ACT_SOFTMAX;
        float row[3] = {0.4f, -0.2f, 2.f};
        if (!MUST_INIT(&nn, MODE_MULTICAT_CLASSIFICATION, 2, sizes, &act)) return;
        nn.weights[0][0][0] = 0.2f; nn.weights[0][0][1] = -0.3f; nn.weights[0][0][2] = 0.15f;
        nn.weights[0][1][0] = -0.1f; nn.weights[0][1][1] = 0.25f; nn.weights[0][1][2] = 0.05f;
        nn.biases[0][0] = 0.05f; nn.biases[0][1] = -0.1f; nn.biases[0][2] = 0.2f;
        check_grads("ce-softmax", &nn, row, 1e-3f, atol, rtol, 1);
        free_ezNN(&nn);
    }
    /* L1 away from the kink. */
    {
        int sizes[2] = {2, 2};
        actType act = ACT_IDENTITY;
        float row[4] = {1.f, 0.5f, 0.f, 1.f};
        if (!MUST_INIT(&nn, MODE_REGRESSION_L1, 2, sizes, &act)) return;
        nn.weights[0][0][0] = 0.5f; nn.weights[0][0][1] = -0.25f;
        nn.weights[0][1][0] = 0.f; nn.weights[0][1][1] = 0.1f;
        nn.biases[0][0] = 0.25f; nn.biases[0][1] = 0.5f;
        check_grads("l1-identity", &nn, row, h, 1e-4, 1e-4, 1);
        free_ezNN(&nn);
    }
    /* Two-layer chain: tanh then sigmoid, L2. */
    {
        int sizes[3] = {2, 3, 2};
        actType acts[2] = {ACT_TANH, ACT_SIGMOID};
        float row[4] = {0.4f, -0.2f, 0.2f, 0.8f};
        if (!MUST_INIT(&nn, MODE_REGRESSION_L2, 3, sizes, acts)) return;
        nn.weights[0][0][0] = 0.3f; nn.weights[0][0][1] = -0.2f; nn.weights[0][0][2] = 0.4f;
        nn.weights[0][1][0] = 0.15f; nn.weights[0][1][1] = 0.25f; nn.weights[0][1][2] = -0.1f;
        nn.biases[0][0] = 0.1f; nn.biases[0][1] = -0.05f; nn.biases[0][2] = 0.2f;
        nn.weights[1][0][0] = 0.4f; nn.weights[1][0][1] = -0.3f;
        nn.weights[1][1][0] = 0.2f; nn.weights[1][1][1] = 0.5f;
        nn.weights[1][2][0] = -0.25f; nn.weights[1][2][1] = 0.15f;
        nn.biases[1][0] = 0.05f; nn.biases[1][1] = -0.1f;
        check_grads("chain-tanh-sigmoid-l2", &nn, row, h, atol, rtol, 1);
        free_ezNN(&nn);
    }
    /* ReLU hidden (strictly positive) into softmax CE. */
    {
        int sizes[3] = {2, 3, 3};
        actType acts[2] = {ACT_RELU, ACT_SOFTMAX};
        float row[3] = {1.f, 1.f, 1.f};
        if (!MUST_INIT(&nn, MODE_MULTICAT_CLASSIFICATION, 3, sizes, acts)) return;
        nn.weights[0][0][0] = 0.4f; nn.weights[0][0][1] = 0.5f; nn.weights[0][0][2] = 0.35f;
        nn.weights[0][1][0] = 0.45f; nn.weights[0][1][1] = 0.3f; nn.weights[0][1][2] = 0.55f;
        nn.biases[0][0] = 0.3f; nn.biases[0][1] = 0.25f; nn.biases[0][2] = 0.4f;
        nn.weights[1][0][0] = 0.2f; nn.weights[1][0][1] = -0.1f; nn.weights[1][0][2] = 0.15f;
        nn.weights[1][1][0] = -0.2f; nn.weights[1][1][1] = 0.3f; nn.weights[1][1][2] = 0.05f;
        nn.weights[1][2][0] = 0.1f; nn.weights[1][2][1] = 0.2f; nn.weights[1][2][2] = -0.25f;
        nn.biases[1][0] = 0.05f; nn.biases[1][1] = -0.05f; nn.biases[1][2] = 0.1f;
        check_grads("chain-relu-softmax-ce", &nn, row, h, atol, rtol, 1);
        free_ezNN(&nn);
    }
}

static void test_classification_hard(void) {
    ezNNType nn;
    int sizes[2] = {1, 4};
    int bin_sizes[2] = {1, 2};
    actType id = ACT_IDENTITY;
    float x = 1.f;
    float *rows[1] = {&x};
    int decision = -1;
    int *decisions[1] = {&decision};
    int i;

    if (!MUST_INIT(&nn, MODE_MULTICAT_CLASSIFICATION, 2, sizes, &id)) return;
    nn.weights[0][0][0] = 0.1f;
    nn.weights[0][0][1] = 0.2f;
    nn.weights[0][0][2] = 0.4f;
    nn.weights[0][0][3] = 0.9f;
    do_classification_hard(&nn, rows, 1, decisions);
    CHECKF(decision == 3, "argmax should be 3, got %d", decision);
    free_ezNN(&nn);

    /* Sample count differs from the output width. The old code passed the
       sample count into the classifier and walked off the output buffer. */
    if (!MUST_INIT(&nn, MODE_BINARY_CLASSIFICATION, 2, bin_sizes, &id)) return;
    nn.weights[0][0][0] = 0.2f;
    nn.weights[0][0][1] = 0.8f;
    {
        float *xs[5];
        int *ys[5];
        float xb[5];
        for (i = 0; i < 5; i++) {
            xb[i] = 1.f;
            xs[i] = &xb[i];
            ys[i] = (int *)malloc(2 * sizeof(int));
            CHECKF(ys[i] != NULL, "alloc");
        }
        do_classification_hard(&nn, xs, 5, ys);
        for (i = 0; i < 5; i++) {
            CHECKF(ys[i][0] == 0 && ys[i][1] == 1, "binary hard sample %d -> %d %d", i, ys[i][0], ys[i][1]);
            free(ys[i]);
        }
    }
    nn.weights[0][0][0] = 0.f;
    nn.weights[0][0][1] = 0.f;
    nn.biases[0][0] = 0.5f;
    nn.biases[0][1] = 0.2f;
    {
        float one = 0.f;
        float *one_row[1] = {&one};
        int tied[2] = {-1, -1};
        int *tied_p[1] = {tied};
        do_classification_hard(&nn, one_row, 1, tied_p);
        CHECKF(tied[0] == 1, "threshold tie maps to class 1, got %d", tied[0]);
        CHECKF(tied[1] == 0, "0.2 maps to class 0, got %d", tied[1]);
    }
    free_ezNN(&nn);
}

static void test_save_load(void) {
    ezNNType nn, loaded;
    int shapes[3][4] = {
        {2, 1, 0, 0},
        {3, 4, 2, 0},
        {2, 3, 4, 1}
    };
    int nlayers[3] = {2, 3, 4};
    actType acts[3][3] = {
        {ACT_SIGMOID, 0, 0},
        {ACT_RELU, ACT_TANH, 0},
        {ACT_RELU, ACT_TANH, ACT_SOFTMAX}
    };
    modeType modes[3] = {MODE_BINARY_CLASSIFICATION, MODE_REGRESSION_L1, MODE_MULTICAT_CLASSIFICATION};
    int s;
    const char *path = "/tmp/eznn-test-model.bin";
    unsigned char header[8];
    FILE *fp;
    int err;

    for (s = 0; s < 3; s++) {
        int n = nlayers[s];
        float in[4] = {0.25f, -0.5f, 0.125f, 0.75f};
        float *rows[1] = {in};
        int nout;
        float *out_a, *out_b;
        float *oa[1], *ob[1];
        int i, j, k, t = 0;
        if (!MUST_INIT(&nn, modes[s], n, shapes[s], acts[s])) return;
        for (i = 0; i < n - 1; i++) {
            for (j = 0; j < nn.layer_sizes[i]; j++) {
                for (k = 0; k < nn.layer_sizes[i + 1]; k++) {
                    float v = ((t % 7) - 3) * 0.125f;
                    if (t == 5) v = -1.5f;
                    if (t == 6) v = 1e-20f;
                    nn.weights[i][j][k] = v;
                    t++;
                }
            }
            for (k = 0; k < nn.layer_sizes[i + 1]; k++) nn.biases[i][k] = 0.5f - 0.25f * (float)k;
        }
        save_model_to_file(&nn, (char *)path);
        memset(&loaded, 0, sizeof(loaded));
        load_model_from_file(&loaded, (char *)path);
        CHECKF(params_equal(&nn, &loaded), "round trip shape %d", n);
        nout = nn.layer_sizes[n - 1];
        out_a = (float *)calloc((size_t)nout, sizeof(float));
        out_b = (float *)calloc((size_t)nout, sizeof(float));
        oa[0] = out_a;
        ob[0] = out_b;
        do_inference(&nn, rows, 1, oa);
        do_inference(&loaded, rows, 1, ob);
        CHECKF(memcmp(out_a, out_b, (size_t)nout * sizeof(float)) == 0, "inference mismatch after load");
        free(out_a);
        free(out_b);
        free_ezNN(&nn);
        free_ezNN(&loaded);
    }

    /* Header layout: mode in the low 3 bits, nlayers in the next 5, then
       little-endian layer widths, then activation nibbles. */
    {
        int sizes[2] = {2, 1};
        actType act = ACT_SIGMOID;
        if (!MUST_INIT(&nn, MODE_REGRESSION_L2, 2, sizes, &act)) return;
        nn.weights[0][0][0] = 1.5f;
        nn.weights[0][1][0] = -2.f;
        nn.biases[0][0] = 0.25f;
        save_model_to_file(&nn, (char *)path);
        fp = fopen(path, "rb");
        CHECKF(fp != NULL, "reopen model");
        CHECKF(fread(header, 1, 6, fp) == 6, "short header");
        if (fp) fclose(fp);
        CHECKF(header[0] == (unsigned char)((int)MODE_REGRESSION_L2 + (2 << 3)), "mode/nlayers byte %u", header[0]);
        CHECKF(header[1] == 2 && header[2] == 0, "width 2");
        CHECKF(header[3] == 1 && header[4] == 0, "width 1");
        CHECKF(header[5] == (unsigned char)ACT_SIGMOID, "activation nibble %u", header[5]);
        free_ezNN(&nn);
    }

    fp = fopen(path, "wb");
    if (fp) {
        unsigned char trunc[4] = {16, 1, 0, 1};
        fwrite(trunc, 1, 4, fp);
        fclose(fp);
    }
    memset(&loaded, 0, sizeof(loaded));
    err = silence(STDERR_FILENO);
    load_model_from_file(&loaded, (char *)path);
    restore(STDERR_FILENO, err);
    CHECKF(loaded.nlayers == 0, "truncated file must not leave a live network");

    err = silence(STDERR_FILENO);
    load_model_from_file(&loaded, (char *)"/tmp/eznn-does-not-exist.bin");
    restore(STDERR_FILENO, err);
    CHECKF(loaded.nlayers == 0, "missing file");

    {
        int big[2] = {1, 70000};
        actType act = ACT_IDENTITY;
        if (!MUST_INIT(&nn, MODE_REGRESSION_L2, 2, big, &act)) return;
        err = silence(STDERR_FILENO);
        save_model_to_file(&nn, (char *)path);
        restore(STDERR_FILENO, err);
        free_ezNN(&nn);
    }
    unlink(path);
}

static void test_determinism(void) {
    ezNNType a, b;
    int sizes[3] = {3, 5, 1};
    actType acts[2] = {ACT_RELU, ACT_IDENTITY};
    float data[4][4] = {
        {0.2f, -0.4f, 0.8f, 0.1f},
        {-0.7f, 0.3f, 0.0f, -0.2f},
        {0.5f, 0.5f, -0.5f, 0.4f},
        {1.f, -1.f, 0.25f, 0.0f}
    };
    float *rows[4] = {data[0], data[1], data[2], data[3]};
    int err = silence(STDOUT_FILENO);

    if (!MUST_INIT(&a, MODE_REGRESSION_L2, 3, sizes, acts)) {
        restore(STDOUT_FILENO, err);
        return;
    }
    /* Time-seeded path still has to run without crashing. */
    do_training(&a, rows, 4, 0.05f, 1, 1);
    free_ezNN(&a);

    if (!MUST_INIT(&a, MODE_REGRESSION_L2, 3, sizes, acts)) {
        restore(STDOUT_FILENO, err);
        return;
    }
    if (!MUST_INIT(&b, MODE_REGRESSION_L2, 3, sizes, acts)) {
        free_ezNN(&a);
        restore(STDOUT_FILENO, err);
        return;
    }
    ezNN_seed(123);
    ezNN_set_verbose(0);
    do_training(&a, rows, 4, 0.05f, 8, 1);
    ezNN_seed(123);
    ezNN_set_verbose(1);
    do_training(&b, rows, 4, 0.05f, 8, 1);
    CHECKF(params_equal(&a, &b), "verbose flag changed the updates");
    free_ezNN(&b);

    if (!MUST_INIT(&b, MODE_REGRESSION_L2, 3, sizes, acts)) {
        free_ezNN(&a);
        restore(STDOUT_FILENO, err);
        return;
    }
    ezNN_set_verbose(0);
    ezNN_seed(123);
    do_training(&b, rows, 4, 0.05f, 8, 1);
    CHECKF(params_equal(&a, &b), "same seed diverged");
    free_ezNN(&b);
    if (!MUST_INIT(&b, MODE_REGRESSION_L2, 3, sizes, acts)) {
        free_ezNN(&a);
        restore(STDOUT_FILENO, err);
        return;
    }
    ezNN_seed(456);
    do_training(&b, rows, 4, 0.05f, 8, 1);
    CHECKF(!params_equal(&a, &b), "different seeds produced the same network");
    free_ezNN(&a);
    free_ezNN(&b);

#ifdef EZNN_OPENMP
    {
        int wide[3] = {8, 80, 2};
        float wide_rows_store[4][10];
        float *wide_rows[4];
        int r, c;
        for (r = 0; r < 4; r++) {
            for (c = 0; c < 10; c++) wide_rows_store[r][c] = 0.01f * (float)((r + 1) * (c - 3));
            wide_rows[r] = wide_rows_store[r];
        }
        if (!MUST_INIT(&a, MODE_REGRESSION_L2, 3, wide, acts)) {
            restore(STDOUT_FILENO, err);
            return;
        }
        if (!MUST_INIT(&b, MODE_REGRESSION_L2, 3, wide, acts)) {
            free_ezNN(&a);
            restore(STDOUT_FILENO, err);
            return;
        }
        omp_set_dynamic(0);
        ezNN_set_verbose(0);
        omp_set_num_threads(2);
        ezNN_seed(7);
        do_training(&a, wide_rows, 4, 0.01f, 2, 1);
        omp_set_num_threads(1);
        ezNN_seed(7);
        do_training(&b, wide_rows, 4, 0.01f, 2, 1);
        CHECKF(params_equal(&a, &b), "OpenMP thread count changed the updates");
        free_ezNN(&a);
        free_ezNN(&b);
    }
#endif
    restore(STDOUT_FILENO, err);
}

static void train_cyclic(ezNNType *nn, float **rows, int n, float lr, int epochs) {
    int e, i;
    ezNN_set_verbose(0);
    for (e = 0; e < epochs; e++) {
        for (i = 0; i < n; i++) {
            float *one[1];
            one[0] = rows[i];
            do_training(nn, one, 1, lr, 1, 0);
        }
    }
}

static float binary_accuracy(ezNNType *nn, float **rows, int n) {
    int nout = nn->layer_sizes[nn->nlayers - 1];
    int nin = nn->layer_sizes[0];
    float **outs = (float **)malloc((size_t)n * sizeof(float *));
    float acc;
    int i;
    for (i = 0; i < n; i++) outs[i] = (float *)malloc((size_t)nout * sizeof(float));
    do_inference(nn, rows, n, outs);
    acc = get_binary_classification_accuracy(nn, rows, n, outs);
    (void)nin;
    for (i = 0; i < n; i++) free(outs[i]);
    free(outs);
    return acc;
}

static float regression_loss(ezNNType *nn, float **rows, int n) {
    int nout = nn->layer_sizes[nn->nlayers - 1];
    float **outs = (float **)malloc((size_t)n * sizeof(float *));
    float loss;
    int i;
    for (i = 0; i < n; i++) outs[i] = (float *)malloc((size_t)nout * sizeof(float));
    do_inference(nn, rows, n, outs);
    loss = get_regression_l2_loss(nn, rows, n, outs);
    for (i = 0; i < n; i++) free(outs[i]);
    free(outs);
    return loss;
}

static void test_learns(void) {
    ezNNType nn;
    int lin_sizes[2] = {1, 1};
    int and_sizes[2] = {2, 1};
    int xor_sizes[3] = {2, 4, 1};
    actType id = ACT_IDENTITY;
    actType sig = ACT_SIGMOID;
    actType xor_acts[2] = {ACT_TANH, ACT_SIGMOID};
    float lin[4][2] = {{-1.f, -2.f}, {0.f, 0.f}, {1.f, 2.f}, {2.f, 4.f}};
    float and_rows[4][3] = {{0.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {1.f, 0.f, 0.f}, {1.f, 1.f, 1.f}};
    float xor_rows[4][3] = {{0.f, 0.f, 0.f}, {0.f, 1.f, 1.f}, {1.f, 0.f, 1.f}, {1.f, 1.f, 0.f}};
    float *lp[4] = {lin[0], lin[1], lin[2], lin[3]};
    float *ap[4] = {and_rows[0], and_rows[1], and_rows[2], and_rows[3]};
    float *xp[4] = {xor_rows[0], xor_rows[1], xor_rows[2], xor_rows[3]};
    float before, after, acc;

    if (!MUST_INIT(&nn, MODE_REGRESSION_L2, 2, lin_sizes, &id)) return;
    before = regression_loss(&nn, lp, 4);
    train_cyclic(&nn, lp, 4, 0.1f, 80);
    after = regression_loss(&nn, lp, 4);
    CHECKF(after < before && after < 1e-3, "line fit loss %g -> %g, w=%g b=%g",
           before, after, nn.weights[0][0][0], nn.biases[0][0]);
    CHECKF(near_enough(nn.weights[0][0][0], 2.0, 0.05, 0.0), "slope %g", nn.weights[0][0][0]);
    free_ezNN(&nn);

    if (!MUST_INIT(&nn, MODE_BINARY_CLASSIFICATION, 2, and_sizes, &sig)) return;
    train_cyclic(&nn, ap, 4, 1.f, 400);
    acc = binary_accuracy(&nn, ap, 4);
    CHECKF(acc == 100.f, "AND accuracy %g", acc);
    free_ezNN(&nn);

    if (!MUST_INIT(&nn, MODE_BINARY_CLASSIFICATION, 3, xor_sizes, xor_acts)) return;
    nn.weights[0][0][0] = 0.6f; nn.weights[0][0][1] = -0.5f; nn.weights[0][0][2] = 0.4f; nn.weights[0][0][3] = -0.3f;
    nn.weights[0][1][0] = -0.4f; nn.weights[0][1][1] = 0.5f; nn.weights[0][1][2] = -0.6f; nn.weights[0][1][3] = 0.2f;
    nn.biases[0][0] = 0.1f; nn.biases[0][1] = -0.2f; nn.biases[0][2] = 0.2f; nn.biases[0][3] = -0.1f;
    nn.weights[1][0][0] = 0.7f; nn.weights[1][1][0] = -0.8f; nn.weights[1][2][0] = 0.5f; nn.weights[1][3][0] = -0.4f;
    nn.biases[1][0] = 0.f;
    before = 0.f;
    {
        float outs_store[4];
        float *os[4];
        int i;
        for (i = 0; i < 4; i++) os[i] = &outs_store[i];
        do_inference(&nn, xp, 4, os);
        before = get_binary_classification_loss(&nn, xp, 4, os);
    }
    train_cyclic(&nn, xp, 4, 0.1f, 2000);
    acc = binary_accuracy(&nn, xp, 4);
    {
        float outs_store[4];
        float *os[4] = {&outs_store[0], &outs_store[1], &outs_store[2], &outs_store[3]};
        do_inference(&nn, xp, 4, os);
        after = get_binary_classification_loss(&nn, xp, 4, os);
    }
    CHECKF(acc == 100.f, "XOR accuracy %g, loss %g -> %g", acc, before, after);
    free_ezNN(&nn);
}

static void test_csv(void) {
    const char *path = "/tmp/eznn-test.csv";
    FILE *fp = fopen(path, "wb");
    int cols = -1, rows, err;
    float **data;
    int i;
    CHECKF(fp != NULL, "csv temp");
    if (!fp) return;
    fputs("1.5,-2,3.25\r\n", fp);
    fputs("\r\n", fp);
    fputs("4,5,6\n", fp);
    fclose(fp);
    rows = read_csv_size((char *)path, &cols);
    CHECKF(rows == 2, "rows %d", rows);
    CHECKF(cols == 3, "cols %d", cols);
    data = (float **)malloc(2 * sizeof(float *));
    data[0] = (float *)malloc(3 * sizeof(float));
    data[1] = (float *)malloc(3 * sizeof(float));
    CHECKF(read_csv((char *)path, rows, cols, data) == 0, "read");
    CHECKF(data[0][0] == 1.5f && data[0][1] == -2.f && data[0][2] == 3.25f, "row0");
    CHECKF(data[1][0] == 4.f && data[1][1] == 5.f && data[1][2] == 6.f, "row1");
    CHECKF(write_csv((char *)path, 2, 3, data) == 0, "write");
    fp = fopen(path, "r");
    if (fp) {
        char line[128];
        CHECKF(fgets(line, sizeof line, fp) != NULL, "header");
        CHECKF(strncmp(line, "Node 1 output,", 14) == 0, "header text %s", line);
        fclose(fp);
    }
    for (i = 0; i < 2; i++) free(data[i]);
    free(data);
    err = silence(STDERR_FILENO);
    CHECKF(read_csv_size((char *)"/tmp/eznn-missing.csv", &cols) < 0, "missing csv");
    restore(STDERR_FILENO, err);
    unlink(path);
}

static void test_regression_hard_matches(void) {
    ezNNType nn;
    int sizes[2] = {2, 2};
    actType act = ACT_TANH;
    float row[2] = {0.3f, -0.7f};
    float *rows[1] = {row};
    float a[2], b[2];
    float *oa[1] = {a};
    float *ob[1] = {b};
    if (!MUST_INIT(&nn, MODE_REGRESSION_L2, 2, sizes, &act)) return;
    nn.weights[0][0][0] = 0.4f;
    nn.weights[0][0][1] = -0.2f;
    nn.weights[0][1][0] = 0.1f;
    nn.weights[0][1][1] = 0.3f;
    nn.biases[0][0] = 0.05f;
    nn.biases[0][1] = -0.1f;
    do_inference(&nn, rows, 1, oa);
    do_regression_hard(&nn, rows, 1, ob);
    CHECKF(memcmp(a, b, sizeof a) == 0, "regression_hard");
    free_ezNN(&nn);
}

int main(void) {
    test_invalid_init();
    test_zero_init_and_stable_activations();
    test_feature_counts();
    test_loss_values();
    test_exact_steps();
    test_all_grads();
    test_classification_hard();
    test_save_load();
    test_determinism();
    test_learns();
    test_csv();
    test_regression_hard_matches();
    printf("%d checks passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
