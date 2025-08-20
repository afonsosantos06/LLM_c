#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAXCHAR 100

Matrix *create_matrix(int row, int col){
  Matrix *m = malloc(sizeof(Matrix));
  m->rows = row;
  m->cols = col;
  m->entries = malloc(sizeof(double*) * row);
  for (int i = 0; i < row; i++)
    m->entries[i] = malloc(col * sizeof(double));
  return m;
}

void fill_matrix(Matrix *m, int n){
  for (int i = 0; i < m->rows; i++){
    for (int j = 0; j < m->cols; j++)
      m->entries[i][j] = n;
  }
}

void free_matrix(Matrix *m){
  for (int i = 0; i < m->rows; i++)
    free(m->entries[i]);
  free(m->entries);
  free(m);
  m = NULL;
}

void print_matrix(Matrix *m){
  for (int i = 0; i < m->rows; i++){
    for (int j = 0; j < m->cols; j++)
      printf(" %1.3f", m->entries[i][j]);
    putchar('\n');
  }
}
Matrix *copy_matrix(Matrix *m){
  Matrix *m1 = create_matrix(m->rows, m->cols);
  for (int i = 0; i < m->rows; i++){
    for (int j = 0; j < m->cols; j++)
      m1->entries[i][j] = m->entries[i][j];
  }
  return m1;
}

void save_matrix(Matrix *m, char *file_string){
  FILE *file = fopen(file_string, "w");
  fprintf(file, "%d\n", m->rows);
  fprintf(file, "%d\n", m->cols);
  for (int i = 0; i < m->rows; i++){
    for (int j = 0; j < m->cols; j++)
      fprintf(file, "%.6f\n", m->entries[i][j]);
  }
  printf("Sucessfully saved matrix to %s\n", file_string);
  fclose(file);
}

Matrix *load_matrix(char *file_string){
  FILE *file = fopen(file_string, "r");
  char entry[MAXCHAR];
  fgets(entry, MAXCHAR, file);
  int rows = atoi(entry);
  fgets(entry, MAXCHAR, file);
  int cols = atoi(entry);
  Matrix *m = create_matrix(rows, cols);
  for (int i = 0; i < m->rows; i++){
    for (int j = 0; j < m->cols; j++){
      fgets(entry, MAXCHAR, file);
      m->entries[i][j] = strtod(entry, NULL);
    }
  }
  printf("Sucessfully loaded matrix from %s\n", file_string);
  fclose(file);
  return m;
}

double uniform_distribuition(double low, double high){
    double diff = high - low;
    int scale = 10000;
    int scaled_diff = (int)(diff * scale);
    return low + (1.0 * (rand() % scaled_diff) / scale);
}

void randomize_matrix(Matrix *m, int n){
  double min = -1.0 / sqrt(n);
  double max = 1.0 / sqrt(n);
  for (int i = 0; i < m->rows; i++){
    for (int j = 0; j < m->cols; j++){
      m->entries[i][j] = uniform_distribuition(min, max);
    }
  }
}

int argmax_matrix(Matrix *m){
  // expects a Mx1 matrix
  double max_score = 0;
  int max_idx = 0;
  for (int i = 0; i < m->rows; i++){
    if (m->entries[i][0] > max_score){
      max_score = m->entries[i][0];
      max_idx = i;
    }
  }
  return max_idx;
}

Matrix *flatten_matrix(Matrix *m, int axis){
  Matrix *m1;
  if (axis == 0)  
    m1 = create_matrix(m->rows * m->cols, 1);
  else if (axis == 1) 
    m1 = create_matrix(1, m->rows * m->cols);
  else {
    printf("Argument to flatten_matrix must be 0 or 1");
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < m->rows; i++){
    for (int j = 0; j < m->cols; j++){
      if (axis == 0) m1->entries[i * m->cols + j][0] = m->entries[i][j];
      else m1->entries[0][i * m->cols + j] = m->entries[i][j];
    }
  }
  return m1;
}

int check_dimensions(Matrix *m1, Matrix *m2){
  if (m1->cols == m2->cols && m1->rows == m2->rows) return 1; // true
  return 0; 
}

Matrix* multiply(Matrix* m1, Matrix* m2){
  if (check_dimensions(m1, m2) == 1){
    Matrix *m3 = create_matrix(m1->rows, m2->cols);
    for (int i = 0; i < m1->rows; i++){
      for (int j = 0; j < m2->cols; j++)
        m3->entries[i][j] = m1->entries[i][j] * m2->entries[i][j];
    }
    return m3;
  }
  else {
		printf("Dimension mistmatch multiply: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}

Matrix* add(Matrix* m1, Matrix* m2){
  if (check_dimensions(m1, m2) == 1){
    Matrix *m3 = create_matrix(m1->rows, m2->cols);
    for (int i = 0; i < m1->rows; i++){
      for (int j = 0; j < m2->cols; j++)
        m3->entries[i][j] = m1->entries[i][j] + m2->entries[i][j];
    }
    return m3;
  }
  else {
		printf("Dimension mistmatch add: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}

Matrix* subtract(Matrix* m1, Matrix* m2){
  if (check_dimensions(m1, m2) == 1){
    Matrix *m3 = create_matrix(m1->rows, m2->cols);
    for (int i = 0; i < m1->rows; i++){
      for (int j = 0; j < m2->cols; j++)
        m3->entries[i][j] = m1->entries[i][j] - m2->entries[i][j];
    }
    return m3;
  }
  else {
		printf("Dimension mistmatch subtract: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}

Matrix* dot(Matrix *m1, Matrix *m2) {
	if (m1->cols == m2->rows) {
		Matrix *m = create_matrix(m1->rows, m2->cols);
		for (int i = 0; i < m1->rows; i++) {
			for (int j = 0; j < m2->cols; j++) {
				double sum = 0;
				for (int k = 0; k < m2->rows; k++) {
          sum += m1->entries[i][k] * m2->entries[k][j];
			  }
				m->entries[i][j] = sum;
			}
		}
		return m;
	} else {
		printf("Dimension mistmatch dot: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}

Matrix* apply(double (*func)(double), Matrix* m) {
	Matrix *mat = copy_matrix(m);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			mat->entries[i][j] = (*func)(m->entries[i][j]);
		}
	}
	return mat;
}

Matrix* scale(double n, Matrix* m){
  Matrix *res = create_matrix(m->rows, m->cols);
  for (int i = 0; i < m->rows; i++){
    for (int j = 0; j < m->cols; j++){
      res->entries[i][j] = n * m->entries[i][j];
    }
  }
  return res;
}

Matrix* addScalar(double n, Matrix* m){
  Matrix *res = create_matrix(m->rows, m->cols);
  for (int i = 0; i < m->rows; i++){
    for (int j = 0; j < m->cols; j++){
      res->entries[i][j] = m->entries[i][j] + n;
    }
  }
  return res;
}

Matrix* transpose(Matrix* m){
  Matrix *m1 = create_matrix(m->cols, m->rows);
  for (int i = 0; i < m->rows; i++){
    for (int j = 0; j < m->cols; j++){
      m1->entries[j][i] = m->entries[i][j];
    }
  }
  return m1;
}
