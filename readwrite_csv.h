#ifndef READWRITE_CSV_H
#define READWRITE_CSV_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_SIZE 1048576 // 2^20

void read_csv(char *, int, int, float **);
int read_csv_size(char *filename, int *cols);
void write_csv(char *, int, int, float **);

#endif

