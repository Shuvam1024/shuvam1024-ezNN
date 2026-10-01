#ifndef READWRITE_CSV_H
#define READWRITE_CSV_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_SIZE 1048576 /* 2^20 */

/* Numeric CSV, no header. Returns 0 on success and -1 on error.
   read_csv_size writes the column count of the first data row to *cols
   when cols is not NULL. Blank lines are ignored. */
int read_csv(char *filename, int rows, int cols, float **data);
int read_csv_size(char *filename, int *cols);
int write_csv(char *filename, int rows, int cols, float **data);

#endif
