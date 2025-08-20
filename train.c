#include "matrix/matrix.h"
#include "src/activations.h"
#include "src/network.h"
#include "src/utils.h"

int main() {
	//TRAINING
	int number_imgs = 10000;
	Img** imgs = csv_to_imgs("./data/mnist_test.csv", number_imgs);
	NeuralNetwork* net = create_nn(784, 300, 10, 0.1);
	train_nn_batch_imgs(net, imgs, number_imgs);
	save_nn(net, "testing_net");

	free_imgs(imgs, number_imgs);
	free_nn(net);
	return 0;
}