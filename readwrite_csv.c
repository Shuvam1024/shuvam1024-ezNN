#include "readwrite_csv.h"

static int blank_line(const char *s) {
    if (!s) return 1;
    while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') s++;
    return *s == '\0';
}

static void strip_eol(char *s) {
    char *p;
    if (!s) return;
    p = strchr(s, '\n');
    if (p) *p = '\0';
    p = strchr(s, '\r');
    if (p) *p = '\0';
}

int read_csv_size(char *filename, int *cols) {
    FILE *fp;
    char *line;
    int rows = 0;
    if (cols) *cols = 0;
    if (!filename) return -1;
    fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "ezNN: cannot open %s\n", filename);
        return -1;
    }
    line = (char *)malloc(MAX_LINE_SIZE);
    if (!line) {
        fclose(fp);
        return -1;
    }
    while (fgets(line, MAX_LINE_SIZE, fp)) {
        char *tok;
        int j;
        if (blank_line(line)) continue;
        if (cols && rows == 0) {
            strip_eol(line);
            tok = strtok(line, ",");
            for (j = 0; tok && *tok; j++) tok = strtok(NULL, ",");
            *cols = j;
        }
        rows++;
    }
    free(line);
    fclose(fp);
    return rows;
}

int read_csv(char *filename, int rows, int cols, float **data) {
    FILE *fp;
    char *line;
    int i = 0;
    if (!filename || rows < 0 || cols < 0 || (rows > 0 && !data)) return -1;
    fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "ezNN: cannot open %s\n", filename);
        return -1;
    }
    line = (char *)malloc(MAX_LINE_SIZE);
    if (!line) {
        fclose(fp);
        return -1;
    }
    while (fgets(line, MAX_LINE_SIZE, fp) && i < rows) {
        char *tok;
        int j;
        if (blank_line(line)) continue;
        for (j = 0; j < cols; j++) data[i][j] = 0.f;
        strip_eol(line);
        tok = strtok(line, ",");
        for (j = 0; tok && *tok && j < cols; j++) {
            data[i][j] = (float)atof(tok);
            tok = strtok(NULL, ",");
        }
        i++;
    }
    free(line);
    fclose(fp);
    return 0;
}

int write_csv(char *filename, int rows, int cols, float **data) {
    FILE *fp;
    int i, j;
    if (!filename || rows < 0 || cols <= 0 || (rows > 0 && !data)) return -1;
    fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "ezNN: cannot create %s\n", filename);
        return -1;
    }
    for (i = 1; i <= cols - 1; i++) fprintf(fp, "Node %d output,", i);
    fprintf(fp, "Node %d output\n", cols);
    for (i = 0; i < rows; i++) {
        for (j = 0; j <= cols - 2; j++) fprintf(fp, "%lf,", data[i][j]);
        fprintf(fp, "%lf\n", data[i][cols - 1]);
    }
    fclose(fp);
    return 0;
}
