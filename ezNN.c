#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "ezNN.h"

#ifdef EZNN_OPENMP
#include <omp.h>
#endif

/* Log and explicit probability quotients are clipped to this open interval
   so a saturated unit cannot produce NaN. Inside the interval the clip is
   the identity, and gradient checks are run there. The sigmoid+BCE and
   softmax+CE parameter updates do not use the quotient; they use the
   simplified logit gradient, which stays finite at 0 and 1. */
#define EZNN_PROB_EPS 1e-7

static int g_have_seed = 0;
static unsigned int g_seed = 1u;
static int g_verbose = 1;

void ezNN_seed(unsigned int seed) {
    g_have_seed = 1;
    g_seed = seed;
    srand(seed);
}

void ezNN_set_verbose(int enabled) {
    g_verbose = enabled ? 1 : 0;
}

static void *xmalloc(size_t n) {
    void *p = calloc(1u, n ? n : 1u);
    if (!p) {
        fprintf(stderr, "ezNN: out of memory\n");
        exit(1);
    }
    return p;
}

static double clip_prob(double p) {
    if (p < EZNN_PROB_EPS) return EZNN_PROB_EPS;
    if (p > 1.0 - EZNN_PROB_EPS) return 1.0 - EZNN_PROB_EPS;
    return p;
}

void init_ezNN(ezNNType *nn, modeType mode, int nlayers, int *sizes, actType *acts) {
    int i, j;

    if (!nn) return;
    if (nlayers < 2 || nlayers > MAX_LAYERS || !sizes || !acts) {
        fprintf(stderr, "ezNN: init rejected (nlayers=%d)\n", nlayers);
        return;
    }
    for (i = 0; i < nlayers; i++) {
        if (sizes[i] <= 0) {
            fprintf(stderr, "ezNN: init rejected (layer %d size %d)\n", i, sizes[i]);
            return;
        }
    }
    for (i = 0; i < nlayers - 1; i++) {
        if ((int)acts[i] < (int)ACT_IDENTITY || (int)acts[i] > (int)ACT_SOFTMAX) {
            fprintf(stderr, "ezNN: init rejected (activation %d)\n", (int)acts[i]);
            return;
        }
    }

    memset(nn, 0, sizeof(*nn));
    nn->mode = mode;
    nn->nlayers = nlayers;
    memcpy(nn->layer_sizes, sizes, (size_t)nlayers * sizeof(int));
    memcpy(nn->activations, acts, (size_t)(nlayers - 1) * sizeof(actType));

    for (i = 0; i < nlayers - 1; i++) {
        nn->weights[i] = (float **)xmalloc((size_t)sizes[i] * sizeof(float *));
        nn->weights_grad[i] = (float **)xmalloc((size_t)sizes[i] * sizeof(float *));
        for (j = 0; j < sizes[i]; j++) {
            nn->weights[i][j] = (float *)xmalloc((size_t)sizes[i + 1] * sizeof(float));
            nn->weights_grad[i][j] = (float *)xmalloc((size_t)sizes[i + 1] * sizeof(float));
        }
        nn->biases[i] = (float *)xmalloc((size_t)sizes[i + 1] * sizeof(float));
        nn->biases_grad[i] = (float *)xmalloc((size_t)sizes[i + 1] * sizeof(float));
    }
    for (i = 0; i < nlayers; i++) {
        nn->act_inputs[i] = (float *)xmalloc((size_t)sizes[i] * sizeof(float));
        nn->act_outputs[i] = (float *)xmalloc((size_t)sizes[i] * sizeof(float));
        nn->error[i] = (float *)xmalloc((size_t)sizes[i] * sizeof(float));
        nn->grad[i] = (float *)xmalloc((size_t)sizes[i] * sizeof(float));
    }
}

void free_ezNN(ezNNType *nn) {
    int i, j;

    if (!nn) return;
    if (nn->nlayers > 0) {
        int nlayers = nn->nlayers;
        if (nlayers > MAX_LAYERS) nlayers = MAX_LAYERS;
        for (i = 0; i < nlayers - 1; i++) {
            if (nn->weights[i]) {
                for (j = 0; j < nn->layer_sizes[i]; j++) free(nn->weights[i][j]);
            }
            if (nn->weights_grad[i]) {
                for (j = 0; j < nn->layer_sizes[i]; j++) free(nn->weights_grad[i][j]);
            }
            free(nn->weights[i]);
            free(nn->weights_grad[i]);
            free(nn->biases[i]);
            free(nn->biases_grad[i]);
        }
        for (i = 0; i < nlayers; i++) {
            free(nn->act_inputs[i]);
            free(nn->act_outputs[i]);
            free(nn->error[i]);
            free(nn->grad[i]);
        }
    }
    memset(nn, 0, sizeof(*nn));
}

/* y[i] = b[i] + sum_j x[j] * W[j][i].
   The j loop is the inner reduction for a single output, so each y[i]
   is accumulated in j-order. The i loop is contiguous in W[j] and is the
   one OpenMP slices. That keeps the sum bitwise-identical to one thread. */
static void matmul_range(const float *x, int inlen, int i0, int i1,
                          float **W, const float *b, float *y) {
    int i, j;
    for (i = i0; i < i1; i++) y[i] = b[i];
    for (j = 0; j < inlen; j++) {
        float xj = x[j];
        const float *wrow = W[j];
        for (i = i0; i < i1; i++) y[i] += xj * wrow[i];
    }
}

static void matmul(const float *x, int inlen, int outlen, float **W,
                    const float *b, float *y) {
#ifdef EZNN_OPENMP
    if (outlen >= 64) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            int start = (int)(((long long)outlen * tid) / nt);
            int end = (int)(((long long)outlen * (tid + 1)) / nt);
            matmul_range(x, inlen, start, end, W, b, y);
        }
        return;
    }
#endif
    matmul_range(x, inlen, 0, outlen, W, b, y);
}

static void identity(const float *x, int n, float *y) {
    int k;
    for (k = 0; k < n; ++k) y[k] = x[k];
}

static void relu(const float *x, int n, float *y) {
    int k;
    for (k = 0; k < n; ++k) y[k] = x[k] < 0.f ? 0.f : x[k];
}

static void tan_h(const float *x, int n, float *y) {
    int k;
    for (k = 0; k < n; ++k) y[k] = tanhf(x[k]);
}

/* Stable sigmoid. exp(-x) overflows to +inf for large negative x and
   would otherwise turn the unit into NaN. */
static void sigmoid(const float *x, int n, float *y) {
    int k;
    for (k = 0; k < n; ++k) {
        double z = (double)x[k];
        if (z >= 0.0) {
            double e = exp(-z);
            y[k] = (float)(1.0 / (1.0 + e));
        } else {
            double e = exp(z);
            y[k] = (float)(e / (1.0 + e));
        }
    }
}

/* Subtract the max logit before exp so a large common offset cannot overflow. */
static void softmax(const float *x, int n, float *y) {
    int i;
    double maxv, denom, inv;
    if (n <= 0) return;
    maxv = (double)x[0];
    for (i = 1; i < n; ++i) {
        if ((double)x[i] > maxv) maxv = (double)x[i];
    }
    denom = 0.0;
    for (i = 0; i < n; ++i) denom += exp((double)x[i] - maxv);
    inv = 1.0 / denom;
    for (i = 0; i < n; ++i) y[i] = (float)(exp((double)x[i] - maxv) * inv);
}

static void activate(actType act, const float *x, int n, float *y) {
    switch (act) {
    case ACT_IDENTITY: identity(x, n, y); break;
    case ACT_RELU:     relu(x, n, y); break;
    case ACT_SIGMOID:  sigmoid(x, n, y); break;
    case ACT_TANH:     tan_h(x, n, y); break;
    case ACT_SOFTMAX:  softmax(x, n, y); break;
    default:           identity(x, n, y); break;
    }
}

static void forward_pass(ezNNType *nn, const float *x, float *y) {
    int i;
    int n0 = nn->layer_sizes[0];
    for (i = 0; i < n0; ++i) {
        nn->act_inputs[0][i] = x[i];
        nn->act_outputs[0][i] = x[i];
    }
    for (i = 0; i < nn->nlayers - 1; ++i) {
        matmul(nn->act_outputs[i], nn->layer_sizes[i], nn->layer_sizes[i + 1],
               nn->weights[i], nn->biases[i], nn->act_inputs[i + 1]);
        activate(nn->activations[i], nn->act_inputs[i + 1], nn->layer_sizes[i + 1],
                 nn->act_outputs[i + 1]);
    }
    if (y) {
        int nout = nn->layer_sizes[nn->nlayers - 1];
        memcpy(y, nn->act_outputs[nn->nlayers - 1], (size_t)nout * sizeof(float));
    }
}

static void gradient_identity(const float *in, const float *out, int n,
                              const float *error, float *grad) {
    int k;
    (void)in;
    (void)out;
    for (k = 0; k < n; ++k) grad[k] = error[k];
}

/* At exactly zero the left derivative is 0 and the right derivative is 1.
   The subgradient used here is their average, 1/2, which is also what a
   central difference sees when it straddles the kink. */
static void gradient_relu(const float *in, const float *out, int n,
                          const float *error, float *grad) {
    int k;
    (void)out;
    for (k = 0; k < n; ++k) {
        float deriv = 0.5f;
        if (in[k] < 0.f) deriv = 0.f;
        else if (in[k] > 0.f) deriv = 1.f;
        grad[k] = error[k] * deriv;
    }
}

static void gradient_tanh(const float *in, const float *out, int n,
                          const float *error, float *grad) {
    int k;
    (void)in;
    for (k = 0; k < n; ++k) grad[k] = (1.f - out[k] * out[k]) * error[k];
}

static void gradient_sigmoid(const float *in, const float *out, int n,
                             const float *error, float *grad) {
    int k;
    (void)in;
    for (k = 0; k < n; ++k) grad[k] = (1.f - out[k]) * out[k] * error[k];
}

static void gradient_softmax(const float *in, const float *out, int n,
                             const float *error, float *grad) {
    int k, j;
    (void)in;
    for (k = 0; k < n; ++k) {
        float g = 0.f;
        for (j = 0; j < n; ++j) {
            float jac = (j == k) ? out[k] * (1.f - out[k]) : -out[k] * out[j];
            g += error[j] * jac;
        }
        grad[k] = g;
    }
}

static void activate_grad(actType act, const float *in, const float *out, int n,
                          const float *error, float *grad) {
    switch (act) {
    case ACT_IDENTITY: gradient_identity(in, out, n, error, grad); break;
    case ACT_RELU:     gradient_relu(in, out, n, error, grad); break;
    case ACT_SIGMOID:  gradient_sigmoid(in, out, n, error, grad); break;
    case ACT_TANH:     gradient_tanh(in, out, n, error, grad); break;
    case ACT_SOFTMAX:  gradient_softmax(in, out, n, error, grad); break;
    default:           gradient_identity(in, out, n, error, grad); break;
    }
}

/* dL/da for the output layer, except for the two fused cases handled
   in fused_output_grad. L2 differentiates 1/2 sum (a-y)^2, so dL/da = a-y.
   The printed L2 metric is the mean of sum (a-y)^2 and is twice that
   objective, up to the mean. L1 uses sign(0) = +1. */
static void loss_grad_output(ezNNType *nn, int layer, const float *expected) {
    int i;
    int n = nn->layer_sizes[layer];
    float *a = nn->act_outputs[layer];
    float *err = nn->error[layer];

    switch (nn->mode) {
    case MODE_MULTICAT_CLASSIFICATION:
        for (i = 0; i < n; i++) {
            float ai = a[i];
            if (ai < (float)EZNN_PROB_EPS) ai = (float)EZNN_PROB_EPS;
            if (ai > 1.f - (float)EZNN_PROB_EPS) ai = 1.f - (float)EZNN_PROB_EPS;
            err[i] = -expected[i] / ai;
        }
        break;
    case MODE_BINARY_CLASSIFICATION:
        for (i = 0; i < n; i++) {
            float ai = a[i];
            if (ai < (float)EZNN_PROB_EPS) ai = (float)EZNN_PROB_EPS;
            if (ai > 1.f - (float)EZNN_PROB_EPS) ai = 1.f - (float)EZNN_PROB_EPS;
            err[i] = (ai - expected[i]) / (ai * (1.f - ai));
        }
        break;
    case MODE_REGRESSION_L1:
        for (i = 0; i < n; i++) {
            err[i] = (a[i] - expected[i]) >= 0.f ? 1.f : -1.f;
        }
        break;
    case MODE_REGRESSION_L2:
    default:
        for (i = 0; i < n; i++) err[i] = a[i] - expected[i];
        break;
    }
}

/* Algebraically equal to loss_grad_output * activation'(z) for the
   canonical pairs, without dividing by a probability that can be ~0. */
static int fused_output_grad(ezNNType *nn, int layer, const float *expected) {
    int i, n;
    actType act;
    float *a;
    float *g;
    if (layer != nn->nlayers - 1) return 0;
    n = nn->layer_sizes[layer];
    act = nn->activations[layer - 1];
    a = nn->act_outputs[layer];
    g = nn->grad[layer];
    if (nn->mode == MODE_BINARY_CLASSIFICATION && act == ACT_SIGMOID) {
        for (i = 0; i < n; i++) g[i] = a[i] - expected[i];
        return 1;
    }
    if (nn->mode == MODE_MULTICAT_CLASSIFICATION && act == ACT_SOFTMAX) {
        float ysum = 0.f;
        for (i = 0; i < n; i++) ysum += expected[i];
        for (i = 0; i < n; i++) g[i] = a[i] * ysum - expected[i];
        return 1;
    }
    return 0;
}

static void backprop_linear(ezNNType *nn, int layer) {
    int i;
    int n = nn->layer_sizes[layer];
    int next = nn->layer_sizes[layer + 1];
    float *err = nn->error[layer];
    float **W = nn->weights[layer];
    const float *gn = nn->grad[layer + 1];

#ifdef EZNN_OPENMP
    if (n >= 64) {
        #pragma omp parallel for schedule(static)
        for (i = 0; i < n; ++i) {
            const float *wrow = W[i];
            float s = 0.f;
            int k;
            for (k = 0; k < next; ++k) s += wrow[k] * gn[k];
            err[i] = s;
        }
        return;
    }
#endif
    for (i = 0; i < n; ++i) {
        const float *wrow = W[i];
        float s = 0.f;
        int k;
        for (k = 0; k < next; ++k) s += wrow[k] * gn[k];
        err[i] = s;
    }
}

static void calc_param_grads(ezNNType *nn, int layer) {
    int in = nn->layer_sizes[layer - 1];
    int out = nn->layer_sizes[layer];
    const float *g = nn->grad[layer];
    const float *act = nn->act_outputs[layer - 1];
    float **Wg = nn->weights_grad[layer - 1];
    float *bg = nn->biases_grad[layer - 1];
    int k, j;

#ifdef EZNN_OPENMP
    if (in >= 64) {
        #pragma omp parallel for schedule(static)
        for (k = 0; k < in; ++k) {
            float ak = act[k];
            float *grow = Wg[k];
            int t;
            for (t = 0; t < out; ++t) grow[t] = g[t] * ak;
        }
        for (j = 0; j < out; ++j) bg[j] = g[j];
        return;
    }
#endif
    for (k = 0; k < in; ++k) {
        float ak = act[k];
        float *grow = Wg[k];
        for (j = 0; j < out; ++j) grow[j] = g[j] * ak;
    }
    for (j = 0; j < out; ++j) bg[j] = g[j];
}

static void update_params(ezNNType *nn, float learning_rate) {
    int layer;
    for (layer = 0; layer < nn->nlayers - 1; layer++) {
        int in = nn->layer_sizes[layer];
        int out = nn->layer_sizes[layer + 1];
        float **W = nn->weights[layer];
        float **G = nn->weights_grad[layer];
        float *b = nn->biases[layer];
        float *bg = nn->biases_grad[layer];
        int j, k;
#ifdef EZNN_OPENMP
        if (in >= 64) {
            #pragma omp parallel for schedule(static)
            for (j = 0; j < in; j++) {
                int t;
                for (t = 0; t < out; t++) W[j][t] -= G[j][t] * learning_rate;
            }
        } else
#endif
        {
            for (j = 0; j < in; j++) {
                for (k = 0; k < out; k++) W[j][k] -= G[j][k] * learning_rate;
            }
        }
        for (k = 0; k < out; k++) b[k] -= bg[k] * learning_rate;
    }
}

void do_inference(ezNNType *nn, float **inputs, int n, float **outputs) {
    int i;
    if (!nn || nn->nlayers < 2 || n <= 0 || !inputs || !outputs) return;
    for (i = 0; i < n; ++i) {
        if (!inputs[i] || !outputs[i]) continue;
        forward_pass(nn, inputs[i], outputs[i]);
    }
}

static void hard_binary_classify(const float *p, int n, int *hard) {
    int i;
    for (i = 0; i < n; ++i) hard[i] = p[i] < 0.5f ? 0 : 1;
}

static int hard_multicat_classify(const float *p, int n) {
    int i, max_index = 0;
    float maxv;
    if (n <= 0) return 0;
    maxv = p[0];
    for (i = 1; i < n; ++i) {
        if (p[i] > maxv) {
            maxv = p[i];
            max_index = i;
        }
    }
    return max_index;
}

void do_classification_hard(ezNNType *nn, float **inputs, int n, int **outputs) {
    int i;
    int width;
    float *outs;
    if (!nn || nn->nlayers < 2 || n <= 0 || !inputs || !outputs) return;
    width = nn->layer_sizes[nn->nlayers - 1];
    outs = (float *)xmalloc((size_t)width * sizeof(float));
    for (i = 0; i < n; ++i) {
        if (!inputs[i] || !outputs[i]) continue;
        forward_pass(nn, inputs[i], outs);
        if (nn->mode == MODE_BINARY_CLASSIFICATION) {
            hard_binary_classify(outs, width, outputs[i]);
        } else if (nn->mode == MODE_MULTICAT_CLASSIFICATION) {
            outputs[i][0] = hard_multicat_classify(outs, width);
        }
    }
    free(outs);
}

void do_regression_hard(ezNNType *nn, float **inputs, int n, float **outputs) {
    do_inference(nn, inputs, n, outputs);
}

static void back_propagation(ezNNType *nn, const float *expected, float learning_rate) {
    int i;
    for (i = nn->nlayers - 1; i >= 1; --i) {
        if (!fused_output_grad(nn, i, expected)) {
            if (i == nn->nlayers - 1) loss_grad_output(nn, i, expected);
            else backprop_linear(nn, i);
            activate_grad(nn->activations[i - 1], nn->act_inputs[i], nn->act_outputs[i],
                          nn->layer_sizes[i], nn->error[i], nn->grad[i]);
        }
        calc_param_grads(nn, i);
    }
    update_params(nn, learning_rate);
}

static void initialize_random_params(ezNNType *nn) {
    int i, j, k;
    if (g_have_seed) srand(g_seed);
    else srand((unsigned)time(NULL));
    for (i = 0; i < nn->nlayers - 1; i++) {
        double fan = (double)nn->layer_sizes[i] * (double)nn->layer_sizes[i + 1];
        float epsilon = (float)sqrt(1.0 / fan);
        for (j = 0; j < nn->layer_sizes[i]; j++) {
            for (k = 0; k < nn->layer_sizes[i + 1]; k++) {
                nn->weights[i][j][k] =
                    (float)rand() / (float)RAND_MAX * 2.f * epsilon - epsilon;
            }
        }
        for (j = 0; j < nn->layer_sizes[i + 1]; j++) {
            nn->biases[i][j] = (float)rand() / (float)RAND_MAX * 2.f * epsilon - epsilon;
        }
    }
}

static int loss_args_ok(const ezNNType *nn, float **train_data, int train_samples, float **outputs) {
    return nn && nn->nlayers >= 2 && train_samples > 0 && train_data && outputs;
}

float get_regression_l2_loss(ezNNType *nn, float **train_data, int train_samples, float **outputs) {
    int i, j;
    int nin, nout;
    double sse = 0.0;
    if (!loss_args_ok(nn, train_data, train_samples, outputs)) return 0.f;
    nin = nn->layer_sizes[0];
    nout = nn->layer_sizes[nn->nlayers - 1];
    for (i = 0; i < train_samples; i++) {
        for (j = 0; j < nout; j++) {
            double diff = (double)train_data[i][nin + j] - (double)outputs[i][j];
            sse += diff * diff;
        }
    }
    return (float)(sse / (double)train_samples);
}

float get_regression_l1_loss(ezNNType *nn, float **train_data, int train_samples, float **outputs) {
    int i, j;
    int nin, nout;
    double sae = 0.0;
    if (!loss_args_ok(nn, train_data, train_samples, outputs)) return 0.f;
    nin = nn->layer_sizes[0];
    nout = nn->layer_sizes[nn->nlayers - 1];
    for (i = 0; i < train_samples; i++) {
        for (j = 0; j < nout; j++) {
            sae += fabs((double)train_data[i][nin + j] - (double)outputs[i][j]);
        }
    }
    return (float)(sae / (double)train_samples);
}

float get_binary_classification_loss(ezNNType *nn, float **train_data, int train_samples, float **outputs) {
    int i, j;
    int nin, nout;
    double entropy = 0.0;
    if (!loss_args_ok(nn, train_data, train_samples, outputs)) return 0.f;
    nin = nn->layer_sizes[0];
    nout = nn->layer_sizes[nn->nlayers - 1];
    for (i = 0; i < train_samples; i++) {
        for (j = 0; j < nout; j++) {
            double y = (double)train_data[i][nin + j];
            double a = clip_prob((double)outputs[i][j]);
            entropy -= y * log(a) + (1.0 - y) * log(1.0 - a);
        }
    }
    return (float)(entropy / (double)train_samples);
}

float get_multicat_classification_loss(ezNNType *nn, float **train_data, int train_samples, float **outputs) {
    int i;
    int nin, nout;
    double entropy = 0.0;
    if (!loss_args_ok(nn, train_data, train_samples, outputs)) return 0.f;
    nin = nn->layer_sizes[0];
    nout = nn->layer_sizes[nn->nlayers - 1];
    for (i = 0; i < train_samples; i++) {
        int expected = (int)train_data[i][nin];
        double a;
        if (expected < 0 || expected >= nout) continue;
        a = clip_prob((double)outputs[i][expected]);
        entropy -= log(a);
    }
    return (float)(entropy / (double)train_samples);
}

float get_binary_classification_accuracy(ezNNType *nn, float **train_data, int train_samples, float **outputs) {
    int i, j, correct = 0;
    int nin, nout;
    if (!loss_args_ok(nn, train_data, train_samples, outputs)) return 0.f;
    nin = nn->layer_sizes[0];
    nout = nn->layer_sizes[nn->nlayers - 1];
    for (i = 0; i < train_samples; i++) {
        for (j = 0; j < nout; j++) {
            int expected = (int)train_data[i][nin + j];
            int predicted = outputs[i][j] >= 0.5f ? 1 : 0;
            if (predicted == expected) correct++;
        }
    }
    return (float)correct / (float)(train_samples * nout) * 100.f;
}

float get_multicat_classification_accuracy(ezNNType *nn, float **train_data, int train_samples, float **outputs) {
    int i, correct = 0;
    int nin, nout;
    if (!loss_args_ok(nn, train_data, train_samples, outputs)) return 0.f;
    nin = nn->layer_sizes[0];
    nout = nn->layer_sizes[nn->nlayers - 1];
    for (i = 0; i < train_samples; i++) {
        int expected = (int)train_data[i][nin];
        int predicted = hard_multicat_classify(outputs[i], nout);
        if (predicted == expected) correct++;
    }
    return (float)correct / (float)train_samples * 100.f;
}

static int write_full(FILE *fp, const void *ptr, size_t size, size_t count) {
    return fwrite(ptr, size, count, fp) == count;
}

static int read_full(FILE *fp, void *ptr, size_t size, size_t count) {
    return fread(ptr, size, count, fp) == count;
}

void save_model_to_file(ezNNType *nn, char *model_filename) {
    FILE *fp;
    unsigned char byte;
    int i, j;
    if (!nn || !model_filename || nn->nlayers < 2 || nn->nlayers > MAX_LAYERS) {
        fprintf(stderr, "ezNN: cannot save model\n");
        return;
    }
    if ((int)nn->mode < 0 || (int)nn->mode > (int)MODE_MULTICAT_CLASSIFICATION) {
        fprintf(stderr, "ezNN: cannot save model (mode %d)\n", (int)nn->mode);
        return;
    }
    for (i = 0; i < nn->nlayers; i++) {
        if (nn->layer_sizes[i] <= 0 || nn->layer_sizes[i] > 65535) {
            fprintf(stderr, "ezNN: layer size %d does not fit in the model file\n", nn->layer_sizes[i]);
            return;
        }
    }
    fp = fopen(model_filename, "wb");
    if (!fp) {
        fprintf(stderr, "ezNN: cannot open %s for writing\n", model_filename);
        return;
    }
    byte = (unsigned char)((int)nn->mode + (nn->nlayers << 3));
    if (!write_full(fp, &byte, 1, 1)) goto fail;
    for (i = 0; i < nn->nlayers; ++i) {
        byte = (unsigned char)(nn->layer_sizes[i] & 255);
        if (!write_full(fp, &byte, 1, 1)) goto fail;
        byte = (unsigned char)((nn->layer_sizes[i] >> 8) & 255);
        if (!write_full(fp, &byte, 1, 1)) goto fail;
    }
    for (i = 0; i < nn->nlayers - 1; i += 2) {
        byte = (unsigned char)nn->activations[i];
        if (i < nn->nlayers - 2) byte = (unsigned char)(byte + (nn->activations[i + 1] << 4));
        if (!write_full(fp, &byte, 1, 1)) goto fail;
    }
    for (i = 0; i < nn->nlayers - 1; i++) {
        for (j = 0; j < nn->layer_sizes[i]; j++) {
            if (!write_full(fp, nn->weights[i][j], sizeof(float), (size_t)nn->layer_sizes[i + 1]))
                goto fail;
        }
        if (!write_full(fp, nn->biases[i], sizeof(float), (size_t)nn->layer_sizes[i + 1]))
            goto fail;
    }
    fclose(fp);
    return;
fail:
    fprintf(stderr, "ezNN: short write to %s\n", model_filename);
    fclose(fp);
}

void load_model_from_file(ezNNType *nn, char *model_filename) {
    FILE *fp;
    unsigned char byte;
    modeType mode;
    int nlayers, i, j;
    int sizes[MAX_LAYERS];
    actType activations[MAX_LAYERS];
    if (!nn || !model_filename) return;
    fp = fopen(model_filename, "rb");
    if (!fp) {
        fprintf(stderr, "ezNN: cannot open %s\n", model_filename);
        memset(nn, 0, sizeof(*nn));
        return;
    }
    if (!read_full(fp, &byte, 1, 1)) goto bad;
    mode = (modeType)(byte & 7);
    nlayers = byte >> 3;
    if ((int)mode > (int)MODE_MULTICAT_CLASSIFICATION || nlayers < 2 || nlayers > MAX_LAYERS)
        goto bad;
    for (i = 0; i < nlayers; ++i) {
        if (!read_full(fp, &byte, 1, 1)) goto bad;
        sizes[i] = byte;
        if (!read_full(fp, &byte, 1, 1)) goto bad;
        sizes[i] += (byte << 8);
        if (sizes[i] <= 0) goto bad;
    }
    for (i = 0; i < nlayers - 1; i += 2) {
        if (!read_full(fp, &byte, 1, 1)) goto bad;
        activations[i] = (actType)(byte & 15);
        if ((int)activations[i] > (int)ACT_SOFTMAX) goto bad;
        if (i < nlayers - 2) {
            activations[i + 1] = (actType)(byte >> 4);
            if ((int)activations[i + 1] > (int)ACT_SOFTMAX) goto bad;
        }
    }
    init_ezNN(nn, mode, nlayers, sizes, activations);
    if (nn->nlayers != nlayers) goto bad;
    for (i = 0; i < nn->nlayers - 1; i++) {
        for (j = 0; j < nn->layer_sizes[i]; j++) {
            if (!read_full(fp, nn->weights[i][j], sizeof(float), (size_t)nn->layer_sizes[i + 1]))
                goto bad_init;
        }
        if (!read_full(fp, nn->biases[i], sizeof(float), (size_t)nn->layer_sizes[i + 1]))
            goto bad_init;
    }
    fclose(fp);
    return;
bad_init:
    free_ezNN(nn);
    fprintf(stderr, "ezNN: invalid model file %s\n", model_filename);
    fclose(fp);
    return;
bad:
    memset(nn, 0, sizeof(*nn));
    fprintf(stderr, "ezNN: invalid model file %s\n", model_filename);
    fclose(fp);
}

int get_num_features(ezNNType *nn) {
    if (!nn || nn->nlayers < 2) return 0;
    return nn->layer_sizes[0] + (nn->mode == MODE_MULTICAT_CLASSIFICATION ? 1 : nn->layer_sizes[nn->nlayers - 1]);
}

int get_num_out_features(ezNNType *nn) {
    if (!nn || nn->nlayers < 2) return 0;
    return nn->mode == MODE_MULTICAT_CLASSIFICATION ? 1 : nn->layer_sizes[nn->nlayers - 1];
}

static void random_shuffle(int n, int *indices) {
    int i, k;
    for (i = 0; i < n; ++i) indices[i] = i;
    for (k = n - 1; k >= 1; --k) {
        int r = rand() % (k + 1);
        int t = indices[k];
        indices[k] = indices[r];
        indices[r] = t;
    }
}

static void print_epoch_metrics(ezNNType *nn, float **train_data, int n, float **outputs, int epoch) {
    printf("epoch: %d  ", epoch);
    if (nn->mode == MODE_REGRESSION_L2) {
        printf("loss: %f \r", get_regression_l2_loss(nn, train_data, n, outputs));
    } else if (nn->mode == MODE_REGRESSION_L1) {
        printf("loss: %f \r", get_regression_l1_loss(nn, train_data, n, outputs));
    } else if (nn->mode == MODE_BINARY_CLASSIFICATION) {
        printf("loss: %f, accuracy: %f \r",
               get_binary_classification_loss(nn, train_data, n, outputs),
               get_binary_classification_accuracy(nn, train_data, n, outputs));
    } else {
        printf("loss: %f, accuracy: %f \r",
               get_multicat_classification_loss(nn, train_data, n, outputs),
               get_multicat_classification_accuracy(nn, train_data, n, outputs));
    }
    fflush(stdout);
}

void do_training(ezNNType *nn, float **train_data, int train_sample, float learning_rate, int max_epochs, int reset) {
    int epoch, sample, i;
    int nout = 0;
    int *indices = NULL;
    float *onehot = NULL;
    float **outputs = NULL;

    if (!nn || nn->nlayers < 2 || train_sample < 0) return;
    if (train_sample > 0 && !train_data) return;
    if (reset) initialize_random_params(nn);

    nout = nn->layer_sizes[nn->nlayers - 1];
    if (nn->mode == MODE_MULTICAT_CLASSIFICATION) {
        onehot = (float *)xmalloc((size_t)nout * sizeof(float));
    }
    if (train_sample > 0) {
        indices = (int *)xmalloc((size_t)train_sample * sizeof(int));
        if (g_verbose) {
            outputs = (float **)xmalloc((size_t)train_sample * sizeof(float *));
            for (i = 0; i < train_sample; ++i) {
                outputs[i] = (float *)xmalloc((size_t)nout * sizeof(float));
            }
        }
    }

    for (epoch = 0; epoch < max_epochs; ++epoch) {
        if (train_sample > 0) random_shuffle(train_sample, indices);
        for (sample = 0; sample < train_sample; ++sample) {
            float *row = train_data[indices[sample]];
            float *target;
            if (!row) continue;
            if (nn->mode == MODE_MULTICAT_CLASSIFICATION) {
                int cl = (int)row[nn->layer_sizes[0]];
                memset(onehot, 0, (size_t)nout * sizeof(float));
                if (cl >= 0 && cl < nout) onehot[cl] = 1.f;
                target = onehot;
            } else {
                target = row + nn->layer_sizes[0];
            }
            forward_pass(nn, row, NULL);
            back_propagation(nn, target, learning_rate);
        }
        if (g_verbose && train_sample > 0) {
            do_inference(nn, train_data, train_sample, outputs);
            print_epoch_metrics(nn, train_data, train_sample, outputs, epoch);
        }
    }
    if (g_verbose && train_sample > 0) printf("\n");

    if (outputs) {
        for (i = 0; i < train_sample; ++i) free(outputs[i]);
        free(outputs);
    }
    free(indices);
    free(onehot);
}
