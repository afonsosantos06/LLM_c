#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "matrix/matrix.h"
#include "src/activations.h"
#include "src/network.h"
#include "src/utils.h"

int main(void){
  srand(time(NULL));

  // TRAINING
  int train_imgs = 10000;
  Img** train_data = csv_to_imgs("data/mnist_train.csv", train_imgs);
  NeuralNetwork* net = create_nn(784, 300, 10, 0.1);
  train_nn_batch_imgs(net, train_data, train_imgs);
  save_nn(net, "testing_net");

  free_imgs(train_data, train_imgs);
  free_nn(net);

  // PREDICTING
  int test_imgs = 3000;
  Img** test_data = csv_to_imgs("data/mnist_test.csv", test_imgs);
  NeuralNetwork* net2 = load_nn("testing_net");
  double score = nn_imgs_predict(net2, test_data, 1000);
  printf("Score: %1.5f\n", score);

  free_imgs(test_data, test_imgs);
  free_nn(net2);
  return 0;
}
