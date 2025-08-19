#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
  double **entries;
  int rows;
  int cols;
} Matrix;

Matrix *create_matrix(int row, int col);
void fill_matrix(Matrix *m, int n);
void free_matrix(Matrix *m);
void print_matrix(Matrix *m);
Matrix *copy_matrix(Matrix *m);
void save_matrix(Matrix *m, char *file_string);
Matrix *load_matrix(char *file_string);
void randomize_matrix(Matrix *m, int n);
double uniform_distribuition(double low, double high);
int argmax_matrix(Matrix *m);
Matrix *flatten_matrix(Matrix *m, int axis);
Matrix* multiply(Matrix* m1, Matrix* m2);
Matrix* add(Matrix* m1, Matrix* m2);
Matrix* subtract(Matrix* m1, Matrix* m2);
Matrix* dot(Matrix* m1, Matrix* m2);
Matrix* apply(double (*func)(double), Matrix* m);
Matrix* scale(double n, Matrix* m);
Matrix* addScalar(double n, Matrix* m);
Matrix* transpose(Matrix* m);

#endif