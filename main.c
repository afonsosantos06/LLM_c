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
  int train_imgs = 50000;  // Use more training data
  int test_imgs = 10000;   // Use more test data
  Img** train_data = csv_to_imgs("data/mnist_train.csv", train_imgs);
  Img** test_data = csv_to_imgs("data/mnist_test.csv", test_imgs);
  
  NeuralNetwork* net = create_nn(784, 300, 10, 0.1); 
  
  // Train for multiple epochs with mini-batch SGD
  int epochs = 5;
  int batch_size = 64;
  for (int epoch = 0; epoch < epochs; epoch++) {
    printf("Epoch %d/%d\n", epoch + 1, epochs);
    
    // Shuffle training data for this epoch
    shuffle_imgs(train_data, train_imgs);
    
    train_nn_minibatch_imgs(net, train_data, train_imgs, batch_size);
    
    // Evaluate on test set every epoch
    double score = nn_imgs_predict(net, test_data, 1000);
    printf("Epoch %d accuracy: %.3f\n", epoch + 1, score);
  }
  
  save_nn(net, "testing_net");
  
  // Final evaluation
  double final_score = nn_imgs_predict(net, test_data, test_imgs);
  printf("Final accuracy: %.3f%%\n", final_score * 100.0);

  free_imgs(train_data, train_imgs);
  free_imgs(test_data, test_imgs);
  free_nn(net);
  return 0;
}
