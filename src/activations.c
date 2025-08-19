#include "activations.h"
#include <math.h>

// all the methods have the results be between 0 and 1

double sigmoid(double input) {
	return 1.0 / (1 + exp(-1 * input));
}

Matrix *sigmoidPrime(Matrix *m) { // sigmoid_prime = sigmoid(x) * (1 - sigmoid(x))
	Matrix *ones = create_matrix(m->rows, m->cols);
	fill_matrix(ones, 1);
	Matrix *subtracted = subtract(ones, m);
	Matrix *multiplied = multiply(m, subtracted);
  free_matrix(ones);
  free_matrix(subtracted);
	return multiplied;
}

Matrix* softmax(Matrix* m) {
  // Exponentiation - ensures all values are positive
	double total = 0; // calculates the sum of all the exponencials
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			total += exp(m->entries[i][j]);
		}
	} 
	Matrix* mat = create_matrix(m->rows, m->cols); 
  // Normalization - the exponentiated values are divided by the sum of all exponentiated values
	for (int i = 0; i < mat->rows; i++) {
		for (int j = 0; j < mat->cols; j++) {
			mat->entries[i][j] = exp(m->entries[i][j]) / total;
		}
	}
  // creates a probability distribuition
	return mat;
}