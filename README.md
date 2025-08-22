# Large Language Model in C (MNIST)
With this project I intend to explore C programming, linear algebra, calculus and machine learning by implementing a small neural network from scratch and training it on MNIST (more about MNIST: https://en.wikipedia.org/wiki/MNIST_database).

## Mathematical foundations (gradient descent and backpropagation)

We use a 3-layer feedforward network: input 784 (28×28 pixels) → hidden H → output 10, with sigmoid activations. Matrices are column-oriented (vectors are n×1).

- Shapes
  - Hidden weights W<sub>h</sub> ∈ ℝ<sup>H×784</sup>, output weights W<sub>o</sub> ∈ ℝ<sup>10×H</sup>.
  - Input x ∈ ℝ<sup>784×1</sup>, hidden activations a<sup>h</sup> ∈ ℝ<sup>H×1</sup>, output activations a<sup>o</sup> ∈ ℝ<sup>10×1</sup>.

- Forward propagation
  - Weighted inputs: z<sup>h</sup> = W<sub>h</sub> x + b<sub>h</sub>, z<sup>o</sup> = W<sub>o</sub> a<sup>h</sup> + b<sub>o</sub>.
  - Activations (sigmoid): a<sup>h</sup> = σ(z<sup>h</sup>), a<sup>o</sup> = σ(z<sup>o</sup>), with σ(t) = 1/(1+e<sup>-t</sup>).
  - Sigmoid derivative: σ'(z) = σ(z) ⊙ (1-σ(z)).

- Cost function (quadratic/MSE, single example with one-hot y)
  C = ½ ||a<sup>o</sup> - y||<sub>2</sub><sup>2</sup>.
  Then ∇<sub>a<sup>o</sup></sub> C = a<sup>o</sup> - y.

- Hadamard product (elementwise) u ⊙ v: same shape vectors/matrices multiplied componentwise. It appears whenever we apply scalar derivatives elementwise (e.g., σ'(z)).

### Backpropagation (per example)

Define layer errors (sensitivities) δ<sup>ℓ</sup> = ∂C/∂z<sup>ℓ</sup>.

- Output error (BP1):
  δ<sup>o</sup> = (a<sup>o</sup> - y) ⊙ σ'(z<sup>o</sup>).
- Hidden error (BP2):
  δ<sup>h</sup> = (W<sub>o</sub><sup>⊤</sup> δ<sup>o</sup>) ⊙ σ'(z<sup>h</sup>).
- Gradients (BP3–BP4):
  ∂C/∂W<sub>o</sub> = δ<sup>o</sup> (a<sup>h</sup>)<sup>⊤</sup>, ∂C/∂b<sub>o</sub> = δ<sup>o</sup>,
  ∂C/∂W<sub>h</sub> = δ<sup>h</sup> x<sup>⊤</sup>, ∂C/∂b<sub>h</sub> = δ<sup>h</sup>.
- Gradient descent update (learning rate η):
  W ← W - η ∂C/∂W, b ← b - η ∂C/∂b.

These are Nielsen's equations BP1–BP4 (Neural Networks and Deep Learning, Ch. 2) in vector/matrix form.

### Mini-batch SGD (matrix view)

For a mini-batch of size m, stack columns: X ∈ ℝ<sup>784×m</sup>, Y ∈ ℝ<sup>10×m</sup>, and use 1 ∈ ℝ<sup>m×1</sup>.

- Forward: Z<sup>h</sup> = W<sub>h</sub>X + b<sub>h</sub>1<sup>⊤</sup>, A<sup>h</sup> = σ(Z<sup>h</sup>); Z<sup>o</sup> = W<sub>o</sub>A<sup>h</sup> + b<sub>o</sub>1<sup>⊤</sup>, A<sup>o</sup> = σ(Z<sup>o</sup>).
- Backward: Δ<sup>o</sup> = (A<sup>o</sup> - Y) ⊙ σ'(Z<sup>o</sup>), Δ<sup>h</sup> = (W<sub>o</sub><sup>⊤</sup>Δ<sup>o</sup>) ⊙ σ'(Z<sup>h</sup>).
- Gradients (averaged): ∂C/∂W<sub>o</sub> = (1/m)Δ<sup>o</sup>(A<sup>h</sup>)<sup>⊤</sup>, ∂C/∂b<sub>o</sub> = (1/m)Δ<sup>o</sup>1. Similarly for W<sub>h</sub>,b<sub>h</sub> with Δ<sup>h</sup> and X.

### Optional: softmax + cross-entropy

Using softmax p = softmax(z<sup>o</sup>) with cross-entropy loss yields δ<sup>o</sup> = p - y (no σ'(z<sup>o</sup>) term), with the same matrix shapes for updates.

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