#include "network.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <math.h>
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

  Matrix *a_minus_y = subtract(final_outputs, output_data); // a_o - y
  Matrix *sigp_o = sigmoidPrime(final_outputs);             // elementwise σ'(a_o)
  Matrix *delta_o = multiply(a_minus_y, sigp_o);            // (a_o - y) ⊙ σ'(a_o)

  // Gradients for output weights (SGD step)
  Matrix *hidden_T = transpose(hidden_outputs);
  Matrix *grad_out = dot(delta_o, hidden_T);                // δ_o · a_h^T

  // Hidden layer delta using W_o before update
  Matrix *W_o_T = transpose(net->output_weights);
  Matrix *backprop_err = dot(W_o_T, delta_o);
  Matrix *sigp_h = sigmoidPrime(hidden_outputs);
  Matrix *delta_h = multiply(backprop_err, sigp_h);         // (W_o^T δ_o) ⊙ σ'(a_h)

  // Gradients for hidden weights
  Matrix *input_T = transpose(input_data);
  Matrix *grad_hid = dot(delta_h, input_T);                 // δ_h · x^T

  // Apply SGD updates: W <- W - η * grad (IN-PLACE for efficiency)
  double lr = net->learning_rate;
  
  // Update output weights in-place
  for (int i = 0; i < net->output_weights->rows; i++) {
    for (int j = 0; j < net->output_weights->cols; j++) {
      net->output_weights->entries[i][j] -= lr * grad_out->entries[i][j];
    }
  }
  
  // Update hidden weights in-place
  for (int i = 0; i < net->hidden_weights->rows; i++) {
    for (int j = 0; j < net->hidden_weights->cols; j++) {
      net->hidden_weights->entries[i][j] -= lr * grad_hid->entries[i][j];
    }
  }

  // Free all temporaries
  free_matrix(a_minus_y);
  free_matrix(sigp_o);
  free_matrix(delta_o);
  free_matrix(hidden_T);
  free_matrix(grad_out);
  free_matrix(W_o_T);
  free_matrix(backprop_err);
  free_matrix(sigp_h);
  free_matrix(delta_h);
  free_matrix(input_T);
  free_matrix(grad_hid);
  
  // Free forward pass matrices
  free_matrix(hidden_inputs);
  free_matrix(hidden_outputs);
  free_matrix(final_inputs);
  free_matrix(final_outputs);
}

void train_nn_batch_imgs(NeuralNetwork *net, Img **imgs, int batch_size){
  for (int i = 0; i < batch_size; i++){
    if (i % 1000 == 0) printf("Training image %d/%d\n", i, batch_size);
    Matrix *img_data = flatten_matrix(imgs[i]->img_data, 0); // converts into 1D matrix (matrix with 0 columns)
    Matrix *output = create_matrix(10, 1); // output matrix with 10 rows and 1 column
    int label = imgs[i]->label;
    if (label < 0 || label > 9) fprintf(stderr, "Warning: skipping image %d with invalid label %d\n", i, label);
    else output->entries[label][0] = 1; // Setting the result (one-hot)

    train_nn(net, img_data, output);
    free_matrix(img_data);
    free_matrix(output);
  }
}

// Mini-batch training: accumulate gradients over m samples then update once
void train_nn_minibatch_imgs(NeuralNetwork *net, Img **imgs, int n, int batch_size){
  for (int start = 0; start < n; start += batch_size) {
    int end = start + batch_size;
    if (end > n) end = n;
    int m = end - start;

    if (start % 5000 == 0) printf("Processing batch starting at %d/%d\n", start, n);

    // Accumulators - initialize to zero
    Matrix *acc_grad_out = create_matrix(net->output_weights->rows, net->output_weights->cols);
    Matrix *acc_grad_hid = create_matrix(net->hidden_weights->rows, net->hidden_weights->cols);
    
    // Initialize accumulators to zero
    for (int r = 0; r < acc_grad_out->rows; r++)
      for (int c = 0; c < acc_grad_out->cols; c++)
        acc_grad_out->entries[r][c] = 0.0;
    for (int r = 0; r < acc_grad_hid->rows; r++)
      for (int c = 0; c < acc_grad_hid->cols; c++)
        acc_grad_hid->entries[r][c] = 0.0;

    for (int i = start; i < end; i++) {
      Matrix *x = flatten_matrix(imgs[i]->img_data, 0);
      Matrix *y = create_matrix(10, 1);
      int label = imgs[i]->label;
      if (label >= 0 && label <= 9) y->entries[label][0] = 1;

      // Forward
      Matrix *z_h = dot(net->hidden_weights, x);
      Matrix *a_h = apply(sigmoid, z_h);
      Matrix *z_o = dot(net->output_weights, a_h);
      Matrix *a_o = apply(sigmoid, z_o);

      // Deltas
      Matrix *a_minus_y = subtract(a_o, y);
      Matrix *sigp_o = sigmoidPrime(a_o);
      Matrix *delta_o = multiply(a_minus_y, sigp_o);
      Matrix *a_h_T = transpose(a_h);
      Matrix *grad_out = dot(delta_o, a_h_T);

      Matrix *W_o_T = transpose(net->output_weights);
      Matrix *W_o_T_do = dot(W_o_T, delta_o);
      Matrix *sigp_h = sigmoidPrime(a_h);
      Matrix *delta_h = multiply(W_o_T_do, sigp_h);
      Matrix *x_T = transpose(x);
      Matrix *grad_hid = dot(delta_h, x_T);

      // Accumulate
      for (int r = 0; r < acc_grad_out->rows; r++)
        for (int c = 0; c < acc_grad_out->cols; c++)
          acc_grad_out->entries[r][c] += grad_out->entries[r][c];
      for (int r = 0; r < acc_grad_hid->rows; r++)
        for (int c = 0; c < acc_grad_hid->cols; c++)
          acc_grad_hid->entries[r][c] += grad_hid->entries[r][c];

      // Free temps
      free_matrix(x); free_matrix(y);
      free_matrix(z_h); free_matrix(a_h); free_matrix(z_o); free_matrix(a_o);
      free_matrix(a_minus_y); free_matrix(sigp_o); free_matrix(delta_o);
      free_matrix(a_h_T); free_matrix(grad_out);
      free_matrix(W_o_T); free_matrix(W_o_T_do); free_matrix(sigp_h);
      free_matrix(delta_h); free_matrix(x_T); free_matrix(grad_hid);
    }

    // Average gradients and update once
    double scale = net->learning_rate / (double)m;
    
    // Debug: check if gradients are non-zero
    double max_grad_out = 0.0, max_grad_hid = 0.0;
    for (int r = 0; r < acc_grad_out->rows; r++)
      for (int c = 0; c < acc_grad_out->cols; c++) {
        if (fabs(acc_grad_out->entries[r][c]) > max_grad_out) 
          max_grad_out = fabs(acc_grad_out->entries[r][c]);
        net->output_weights->entries[r][c] -= scale * acc_grad_out->entries[r][c];
      }
    for (int r = 0; r < acc_grad_hid->rows; r++)
      for (int c = 0; c < acc_grad_hid->cols; c++) {
        if (fabs(acc_grad_hid->entries[r][c]) > max_grad_hid) 
          max_grad_hid = fabs(acc_grad_hid->entries[r][c]);
        net->hidden_weights->entries[r][c] -= scale * acc_grad_hid->entries[r][c];
      }
    
    if (start % 5000 == 0) {
      printf("Batch gradients - max output: %.6f, max hidden: %.6f\n", max_grad_out, max_grad_hid);
    }

    free_matrix(acc_grad_out);
    free_matrix(acc_grad_hid);
  }
}

Matrix *nn_img_predict(NeuralNetwork *net, Img *img){
  Matrix *img_data = flatten_matrix(img->img_data, 0); // 
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
  chdir(".."); // Go back to the original directory  
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
  chdir(".."); // goes back to the original directory
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