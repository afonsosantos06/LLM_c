#include "network.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "../matrix/matrix.h"
#include "activations.h"

#define MAXCHAR 1000

NeuralNetwork *create_nn(int input, int hidden, int output, double lr){
  NeuralNetwork *net = malloc(sizeof(NeuralNetwork));
  net->input = input;
  net->hidden = hidden;
  net->output = output;
  net->learning_rate = lr;
  Matrix *hidden_layer = create_matrix(hidden, input);
  Matrix *output_layer = create_matrix(output, hidden);
  randomize_matrix(hidden_layer, hidden);
  randomize_matrix(output_layer, output);
  net->hidden_weights = hidden_layer;
  net->output_weights = output_layer;
  return net;
}

void train_nn(NeuralNetwork *net, Matrix *input_data, Matrix *output_data){
  // Feed forward
  Matrix *hidden_inputs = dot(net->hidden_weights, input_data);
  Matrix *hidden_outputs = apply(sigmoid, hidden_inputs);
  Matrix *final_inputs = dot(net->output_weights, hidden_outputs);
  Matrix *final_outputs = apply(sigmoid, final_inputs);

  // Find errors
  Matrix *output_errors = subtract(output_data, final_outputs); // difference betwen expected and predicted outputs
  Matrix *transposed_mat = transpose(net->output_weights);
  Matrix *hidden_errors = dot(transposed_mat, output_errors); // Error at the 
  free_matrix(transposed_mat);

  /*Backpropagation
  output_weights = add( 
    output_weights,
    scale(
      net->learning_rate,
      dot(
        multiply(
          output_errors, 
          sigmoidPrime(final_outputs)
        ),
      transpose(hidden_outputs)
      ) 
    )
  )
  */
}

void train_nn_batch_imgs(NeuralNetwork *net, Img **imgs, int batch_size){
  for (int i = 0; i < batch_size; i++){
    if (i % 50 == 0) printf("Image nº %d\n", i);
    Matrix *img_data = flatten_matrix(imgs[i]->img_data, 0); // converts into 1D matrix (matrix with 0 columns)
    Matrix *output = create_matrix(10, 1); // output matrix with 10 rows and 1 column
    output->entries[imgs[i]->label][0] = 1; // Setting the result
    train_nn(net, img_data, output);
    free_matrix(img_data);
    free_matrix(output);
  }
}

Matrix *nn_img_predict(NeuralNetwork *net, Img *img){
  Matrix *img_data = flatten_matrix(img->img_data, 0); 
  Matrix *res = nn_predict(net, img_data);
  free_matrix(img_data);
  return res;
}

double nn_imgs_predict(NeuralNetwork *net, Img **imgs, int n){
  int n_correct = 0;
  for (int i = 0; i < n; i++){
    Matrix *prediction = nn_img_predict(net, imgs[i]);
    if (argmax_matrix(prediction) == imgs[i]->label) 
      n_correct++;
    free_matrix(prediction);
  }
  return 1.0 * n_correct / n;
}

Matrix *nn_predict(NeuralNetwork *net, Matrix *input_data){
  Matrix *hidden_inputs = dot(net->hidden_weights, input_data); // product of input data matrix and the weights matrix
  Matrix *hidden_outputs = apply(sigmoid, hidden_inputs); // activates
  Matrix *final_inputs = dot(net->output_weights, hidden_outputs); 
  Matrix *final_outputs = apply(sigmoid, final_inputs);
  Matrix *result = softmax(final_outputs);

  free_matrix(hidden_inputs);
  free_matrix(hidden_outputs);
  free_matrix(final_inputs);
  free_matrix(final_outputs);
  return result;
}

void save_nn(NeuralNetwork *net, char *file_string){
  mkdir(file_string, 0777);
  chdir(file_string);
  FILE *descriptor = fopen("descriptor", "w");
  fprintf(descriptor, "%d\n", net->input);
  fprintf(descriptor, "%d\n", net->hidden);
  fprintf(descriptor, "%d\n", net->output);
  fclose(descriptor);
  save_matrix(net->hidden_weights, "hidden");
  save_matrix(net->output_weights, "output");
  printf("Successfully written to '%s'\n", file_string);
  chdir("-"); // Go back to the original directory  
}

NeuralNetwork *load_nn(char *file_string){
  NeuralNetwork *net = malloc(sizeof(NeuralNetwork));
  char entry[MAXCHAR];
  chdir(file_string); // goes to file named f"{file_string}"
  FILE *descriptor = fopen("descriptor", "r");
  fgets(entry, MAXCHAR, descriptor); // gets first row
  net->input = atoi(entry);
  fgets(entry, MAXCHAR, descriptor);
  net->hidden = atoi(entry);
  fgets(entry, MAXCHAR, descriptor);
  net->output = atoi(entry); 
  fclose(descriptor);
  net->hidden_weights = load_matrix("hidden");
  net->output_weights = load_matrix("output");
  chdir("-"); // goes back to the original directory
  return net;
}

void print_nn(NeuralNetwork *net){
  printf("Nº of Inputs: %d\nNº of Hidden: %d\nNº of Outputs: %d\nLearning Rate: %f\nHidden Weights:\n", net->input, net->hidden, net->output, net->learning_rate);
  print_matrix(net->hidden_weights);
  printf("Output Weights: \n");
  print_matrix(net->output_weights);
}

void free_nn(NeuralNetwork *net){
  free_matrix(net->hidden_weights);
  free_matrix(net->output_weights);
  free(net);
  net = NULL;
}