/* Rebuilds the small datasets under data/.
   Iris source is the UCI file data/iris.data (150 records, no header).
   sha256 6f608b71a7317216319b4d27b4d9bc84e6abd734eda7872b71a458569e2656c0
   The regression set is a deterministic polynomial sample, not an external file.
   Run from the repository root. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define IRIS_N 150
#define PER_CLASS 50
#define TRAIN_PER_CLASS 40

typedef struct {
    char raw[4][64];
    double x[4];
    int y;
} IrisRow;

static void die(const char *msg) {
    fprintf(stderr, "gen_datasets: %s\n", msg);
    exit(1);
}

static void strip_eol(char *s) {
    char *p = strchr(s, '\n');
    if (p) *p = '\0';
    p = strchr(s, '\r');
    if (p) *p = '\0';
}

static int class_index(const char *name) {
    if (strcmp(name, "Iris-setosa") == 0) return 0;
    if (strcmp(name, "Iris-versicolor") == 0) return 1;
    if (strcmp(name, "Iris-virginica") == 0) return 2;
    return -1;
}

static int load_iris(const char *path, IrisRow *rows) {
    FILE *fp = fopen(path, "r");
    char line[256];
    int n = 0;
    if (!fp) die("cannot open data/iris.data");
    while (fgets(line, sizeof line, fp)) {
        char *tok;
        int k;
        strip_eol(line);
        if (line[0] == '\0') continue;
        if (n >= IRIS_N) die("more than 150 iris records");
        tok = strtok(line, ",");
        for (k = 0; k < 4; k++) {
            char *end = NULL;
            if (!tok) die("short iris row");
            if (strlen(tok) >= sizeof rows[n].raw[k]) die("iris token too long");
            memcpy(rows[n].raw[k], tok, strlen(tok) + 1);
            rows[n].x[k] = strtod(tok, &end);
            if (end == tok) die("bad iris feature");
            tok = strtok(NULL, ",");
        }
        if (!tok) die("missing iris class");
        rows[n].y = class_index(tok);
        if (rows[n].y < 0) die("unknown iris class");
        if (rows[n].y != n / PER_CLASS) die("iris classes are not grouped 50/50/50");
        n++;
    }
    fclose(fp);
    if (n != IRIS_N) die("expected 150 iris records");
    return n;
}

static uint32_t g_xs = 1u;

static double u01(void) {
    g_xs ^= g_xs << 13;
    g_xs ^= g_xs >> 17;
    g_xs ^= g_xs << 5;
    return ((double)g_xs + 0.5) / 4294967296.0;
}

static int is_train_iris(int index) {
    return (index % PER_CLASS) < TRAIN_PER_CLASS;
}

int main(void) {
    IrisRow iris[IRIS_N];
    double mean[4] = {0, 0, 0, 0};
    double var[4] = {0, 0, 0, 0};
    double stdv[4];
    FILE *fp;
    int i, k, ntrain = 0;
    const int reg_train = 200;
    const int reg_test = 50;

    load_iris("data/iris.data", iris);

    fp = fopen("data/iris.csv", "w");
    if (!fp) die("cannot write data/iris.csv");
    for (i = 0; i < IRIS_N; i++) {
        fprintf(fp, "%s,%s,%s,%s,%d\n", iris[i].raw[0], iris[i].raw[1], iris[i].raw[2], iris[i].raw[3], iris[i].y);
    }
    fclose(fp);

    for (i = 0; i < IRIS_N; i++) {
        if (!is_train_iris(i)) continue;
        ntrain++;
        for (k = 0; k < 4; k++) mean[k] += iris[i].x[k];
    }
    if (ntrain != 120) die("train split is not 120 rows");
    for (k = 0; k < 4; k++) mean[k] /= (double)ntrain;
    for (i = 0; i < IRIS_N; i++) {
        if (!is_train_iris(i)) continue;
        for (k = 0; k < 4; k++) {
            double d = iris[i].x[k] - mean[k];
            var[k] += d * d;
        }
    }
    for (k = 0; k < 4; k++) {
        var[k] /= (double)ntrain;
        if (var[k] <= 0.0) die("zero iris variance");
        stdv[k] = sqrt(var[k]);
    }

    fp = fopen("data/iris_train.csv", "w");
    if (!fp) die("cannot write iris_train");
    for (i = 0; i < IRIS_N; i++) {
        if (!is_train_iris(i)) continue;
        for (k = 0; k < 4; k++) fprintf(fp, "%.17g,", (iris[i].x[k] - mean[k]) / stdv[k]);
        fprintf(fp, "%d\n", iris[i].y);
    }
    fclose(fp);

    fp = fopen("data/iris_test.csv", "w");
    if (!fp) die("cannot write iris_test");
    for (i = 0; i < IRIS_N; i++) {
        if (is_train_iris(i)) continue;
        for (k = 0; k < 4; k++) fprintf(fp, "%.17g,", (iris[i].x[k] - mean[k]) / stdv[k]);
        fprintf(fp, "%d\n", iris[i].y);
    }
    fclose(fp);

    {
        FILE *train = fopen("data/regression_train.csv", "w");
        FILE *test = fopen("data/regression_test.csv", "w");
        if (!train || !test) die("cannot write regression csv");
        for (i = 0; i < reg_train + reg_test; i++) {
            double x0 = u01() * 2.0 - 1.0;
            double x1 = u01() * 2.0 - 1.0;
            double x2 = u01() * 2.0 - 1.0;
            double y = 0.5 * x0 - 0.8 * x0 * x0 * x0 + 0.4 * x1 * x1 - 0.7 * x2;
            FILE *out = (i < reg_train) ? train : test;
            fprintf(out, "%.17g,%.17g,%.17g,%.17g\n", x0, x1, x2, y);
        }
        fclose(train);
        fclose(test);
    }

    fp = fopen("data/MANIFEST.txt", "w");
    if (!fp) die("cannot write MANIFEST");
    fprintf(fp, "iris_records %d\n", IRIS_N);
    fprintf(fp, "iris_train %d\n", ntrain);
    fprintf(fp, "iris_test %d\n", IRIS_N - ntrain);
    fprintf(fp, "iris_split per_class_first_%d_train_last_%d_test\n", TRAIN_PER_CLASS, PER_CLASS - TRAIN_PER_CLASS);
    fprintf(fp, "iris_normalization zscore_population_over_train\n");
    fprintf(fp, "iris_train_mean %.17g %.17g %.17g %.17g\n", mean[0], mean[1], mean[2], mean[3]);
    fprintf(fp, "iris_train_std %.17g %.17g %.17g %.17g\n", stdv[0], stdv[1], stdv[2], stdv[3]);
    fprintf(fp, "regression_train %d\n", reg_train);
    fprintf(fp, "regression_test %d\n", reg_test);
    fprintf(fp, "regression_seed 1\n");
    fprintf(fp, "regression_x_range [-1,1]\n");
    fprintf(fp, "regression_formula y=0.5*x0-0.8*x0^3+0.4*x1^2-0.7*x2\n");
    fclose(fp);

    printf("wrote iris %d/%d and regression %d/%d\n", ntrain, IRIS_N - ntrain, reg_train, reg_test);
    return 0;
}
