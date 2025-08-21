#include <stdio.h>
#include "matrix/matrix.h"
#include "src/activations.h"
#include "src/network.h"
#include "src/utils.h"

int main(void){
	int number_imgs_train = 10000, number_imgs_test = 3000, epoch = 5;
	int batch_size = 64; // mini-batch size
	NeuralNetwork* net = create_nn(784, 300, 10, 0.1);
	Img **imgs_train = csv_to_imgs("./data/mnist_test.csv", number_imgs_train), **imgs_test = csv_to_imgs("data/mnist_test.csv", number_imgs_test);

	//EPOCH
	for (int i = 0; i < epoch; i++){
		printf("Epoch %d/%d\n", i+1, epoch);
		//TRAINING
		Img** imgs_train = csv_to_imgs("./data/mnist_test.csv", number_imgs_train);
		NeuralNetwork* net = create_nn(784, 300, 10, 0.1);
		train_nn_minibatch_imgs(net, imgs_train, number_imgs_train, batch_size);

		// PREDICTING
		double score = nn_imgs_predict(net, imgs_test, 1000);
		printf("Score: %.2f%%\n", score * 100.0);
	}

	save_nn(net, "testing_net");
	free_imgs(imgs_train, number_imgs_train);
	free_imgs(imgs_test, number_imgs_test);
	free_nn(net);
	return 0;
}