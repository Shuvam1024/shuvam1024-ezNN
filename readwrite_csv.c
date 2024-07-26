#include "readwrite_csv.h"

int read_csv_size(char *filename, int *cols) {
  // Open file and perform sanity check
  FILE* fp = fopen(filename, "r");
  if (NULL == fp) {
    printf("Error opening %s file. Make sure you mentioned the file path correctly\n", filename);
    exit(0);
  }
  // Create memory to read a line/row from the file
  char* line = (char*)malloc(MAX_LINE_SIZE * sizeof(char));
  // Read the file line by line and save it in the matrix 'data'
  int rows, j;
  for (rows = 0; fgets(line, MAX_LINE_SIZE, fp); rows++) {
    if (cols && rows == 0) {
      char* tok = strtok(line, ",");
      for (j = 0; tok && *tok; j++) {
        tok = strtok(NULL, ",\n");
      }
      *cols = j;
    }
  }
  // Free the allocated memory in Heap for line
  free(line);
  // Close the file
  fclose(fp);

  return rows;
}

void read_csv(char* filename, int rows, int cols, float** data) {
  // Open file and perform sanity check
  FILE* fp = fopen(filename, "r");
  if (NULL == fp) {
    printf("Error opening %s file. Make sure you mentioned the file path correctly\n", filename);
    exit(0);
  }
  // Create memory to read a line/row from the file
  char* line = (char*)malloc(MAX_LINE_SIZE * sizeof(char));

  // Read the file line by line and save it in the matrix 'data'
  int i, j;
  for (i = 0; fgets(line, MAX_LINE_SIZE, fp) && i < rows; i++) {
    char* tok = strtok(line, ",");
    for (j = 0; tok && *tok; j++) {
      data[i][j] = atof(tok);
      tok = strtok(NULL, ",\n");
    }
  }

  // Free the allocated memory in Heap for line
  free(line);
  // Close the file
  fclose(fp);
}

void write_csv(char* filename, int rows, int cols, float** data) {
  FILE* fp = fopen(filename, "w");
  if (NULL == fp) {
    printf("Cannot create/open file %s. Make sure you have permission to create/open a file in the directory\n", filename);
    exit(0);
  }

  // Create a header in the file with the output layer node numbers
  int i;
  for (i = 1; i <= cols-1; i++)
    fprintf(fp, "Node %d output,", i);
  fprintf(fp, "Node %d output\n", cols);

  // Dump the matrix into the file element by element
  int j;
  for (i = 0; i < rows; i++) {
    for (j = 0; j <= cols-2; j++) {
      fprintf(fp, "%lf,", data[i][j]);
    }
    fprintf(fp, "%lf\n", data[i][cols-1]);
  }

  // Close the file
  fclose(fp);
}
