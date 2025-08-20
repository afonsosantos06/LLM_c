# Large Language Model in C (MNIST)
With this project I intend to explore C programming, linear algebra, calculus and machine learning by implementing a small neural network from scratch and training it on MNIST (more about MNIST: https://en.wikipedia.org/wiki/MNIST_database).
I took as an inspiration this book https://neuralnetworksanddeeplearning.com

## Mathematical foundations (gradient descent and backpropagation)

We use a 3-layer feedforward network: input 784 (28x28 pixels) → hidden H → output 10, with sigmoid activations. Matrices are column-oriented (vectors are \(n\times1\)).

- Shapes
  - Hidden weights \(W_h \in \mathbb{R}^{H\times 784}\), output weights \(W_o \in \mathbb{R}^{10\times H}\).
  - Input \(x \in \mathbb{R}^{784\times1}\), hidden activations \(a^h \in \mathbb{R}^{H\times1}\), output activations \(a^o \in \mathbb{R}^{10\times1}\).

- Forward propagation
  - Weighted inputs: \(z^h = W_h x + b_h\), \(z^o = W_o a^h + b_o\).
  - Activations (sigmoid): \(a^h = \sigma(z^h)\), \(a^o = \sigma(z^o)\), with \(\sigma(t)=\tfrac{1}{1+e^{-t}}\).
  - Sigmoid derivative: \(\sigma'(z) = \sigma(z)\,\odot\,(1-\sigma(z))\).

- Cost function (quadratic/MSE, single example with one-hot \(y\))
  \[ C = \tfrac{1}{2}\,\lVert a^o - y \rVert_2^2. \]
  Then \(\nabla_{a^o} C = a^o - y\).

- Hadamard product (elementwise) \(u\odot v\): same shape vectors/matrices multiplied componentwise. It appears whenever we apply scalar derivatives elementwise (e.g., \(\sigma'(z)\)).

### Backpropagation (per example)

Define layer errors (sensitivities) \(\delta^\ell = \partial C / \partial z^\ell\).

- Output error (BP1):
  \[ \delta^o = (a^o - y) \odot \sigma'(z^o). \]
- Hidden error (BP2):
  \[ \delta^h = (W_o^\top\,\delta^o) \odot \sigma'(z^h). \]
- Gradients (BP3–BP4):
  \[ \tfrac{\partial C}{\partial W_o} = \delta^o (a^h)^{\top},\quad \tfrac{\partial C}{\partial b_o} = \delta^o, \]
  \[ \tfrac{\partial C}{\partial W_h} = \delta^h x^{\top},\quad \tfrac{\partial C}{\partial b_h} = \delta^h. \]
- Gradient descent update (learning rate \(\eta\)):
  \[ W \leftarrow W - \eta\,\partial C/\partial W,\quad b \leftarrow b - \eta\,\partial C/\partial b. \]

These are Nielsen’s equations BP1–BP4 (Neural Networks and Deep Learning, Ch. 2) in vector/matrix form.

### Mini-batch SGD (matrix view)

For a mini-batch of size \(m\), stack columns: \(X\in\mathbb{R}^{784\times m}\), \(Y\in\mathbb{R}^{10\times m}\), and use \(\mathbf{1}\in\mathbb{R}^{m\times1}\).

- Forward: \(Z^h=W_hX+b_h\mathbf{1}^\top\), \(A^h=\sigma(Z^h)\); \(Z^o=W_oA^h+b_o\mathbf{1}^\top\), \(A^o=\sigma(Z^o)\).
- Backward: \(\Delta^o=(A^o-Y)\odot\sigma'(Z^o)\), \(\Delta^h=(W_o^\top\Delta^o)\odot\sigma'(Z^h)\).
- Gradients (averaged): \(\partial C/\partial W_o=\tfrac{1}{m}\Delta^o(A^h)^\top\), \(\partial C/\partial b_o=\tfrac{1}{m}\,\Delta^o\mathbf{1}\). Similarly for \(W_h,b_h\) with \(\Delta^h\) and \(X\).

### Optional: softmax + cross-entropy

Using softmax \(p=\mathrm{softmax}(z^o)\) with cross-entropy loss yields \(\delta^o = p - y\) (no \(\sigma'(z^o)\) term), with the same matrix shapes for updates.

## Mapping to this codebase

- Core math operations live in `matrix/matrix.c` and `matrix/matrix.h`:
  - `dot(matrix multiplication)`, `transpose`, `add`, `subtract`, `multiply` (Hadamard), `scale`, `apply` (for activations), `sigmoid`, `sigmoidPrime` in `src/activations.c`.
- Forward pass in `src/network.c`:
  - Hidden: `hidden_inputs = dot(net->hidden_weights, input_data)` → `hidden_outputs = apply(sigmoid, hidden_inputs)`
  - Output: `final_inputs = dot(net->output_weights, hidden_outputs)` → `final_outputs = apply(sigmoid, final_inputs)`
- Backprop (conceptual mapping to code edits in `train_nn`):
  - `output_errors = subtract(output_data, final_outputs)`
  - `output_delta = multiply(output_errors, sigmoidPrime(final_outputs))`
  - `grad_output = dot(output_delta, transpose(hidden_outputs))`
  - `net->output_weights = add(net->output_weights, scale(lr, grad_output))`
  - `hidden_delta = multiply(dot(transpose(W_out_before_update), output_delta), sigmoidPrime(hidden_outputs))`
  - `grad_hidden = dot(hidden_delta, transpose(input_data))`
  - `net->hidden_weights = add(net->hidden_weights, scale(lr, grad_hidden))`

Even without biases this improves learning; adding biases and (optionally) softmax+cross-entropy typically pushes MNIST accuracy >95% after a few epochs.

## Build and run

Prerequisites: `gcc`, `make` (on macOS, Xcode command line tools provide both). Data CSVs are included under `data/`.

```bash
make      # builds and runs ./main per Makefile's default rule
./main    # run manually if desired
```

## Download/Clone

To download this repository, make sure you have git installed and type on your command shell:
```shell
$ git clone https://github.com/afonsosantos06/LLM_c
```

## Credits

All code is free to use, I just ask those who may share it with others to credit this github account.