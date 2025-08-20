#include <stdio.h>
#include "matrix/matrix.h"
#include "src/activations.h"
#include "src/network.h"
#include "src/utils.h"

int main() {
	// PREDICTING
	int number_imgs = 3000;
	Img** imgs = csv_to_imgs("data/mnist_test.csv", number_imgs);
	NeuralNetwork* net = load_nn("testing_net");
	double score = nn_imgs_predict(net, imgs, 1000);
	printf("Score: %.2f%%\n", score * 100.0);

	free_imgs(imgs, number_imgs);
	free_nn(net);
	return 0;
}
