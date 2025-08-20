#include <stdio.h>
#include "../matrix/matrix.h"
#include "utils.h"

typedef struct neural_network {
  int input;
  int hidden;
  int output;
  double learning_rate;
  Matrix *hidden_weights;
  Matrix *output_weights;
} NeuralNetwork;

NeuralNetwork *create_nn(int input, int hidden, int output, double lr);
void train_nn(NeuralNetwork *net, Matrix *input_data, Matrix *output_data);
void train_nn_batch_imgs(NeuralNetwork *net, Img **imgs, int batch_size);
void train_nn_minibatch_imgs(NeuralNetwork *net, Img **imgs, int n, int batch_size);
Matrix *nn_img_predict(NeuralNetwork *net, Img *img);
double nn_imgs_predict(NeuralNetwork *net, Img **imgs, int n);
Matrix *nn_predict(NeuralNetwork *net, Matrix *input_data);
void save_nn(NeuralNetwork *net, char *file_string);
NeuralNetwork *load_nn(char *file_string);
void print_nn(NeuralNetwork *net);
void free_nn(NeuralNetwork *net);