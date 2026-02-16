# Feb 16, 2026


## Supervised vs Unsupervised Learning

In supervised learning, we are given data consisting of input–output pairs  

$$
(x_i, y_i) \in \mathcal{X} \times \mathcal{Y}, \quad i=1,\dots,N,
$$

and the goal is to learn a function $f_\theta : \mathcal{X} \to \mathcal{Y}$ that minimizes an empirical risk

$$
\min_\theta \frac{1}{N}\sum_{i=1}^N \ell\bigl(f_\theta(x_i), y_i\bigr),
$$

where $\ell$ is a loss function.

Typical examples include:
- Regression (e.g., squared loss)
- Classification (e.g., cross-entropy loss)

In unsupervised learning, we are given only samples

$$
x_i \in \mathcal{X},
$$

and the objective is to discover structure in the data. Examples include:
- Clustering
- Dimensionality reduction
- Density estimation

Many modern methods blur the line between these settings (self-supervised learning, representation learning, contrastive methods).

---


## Aritificial Neural Networks

A neural network is a parameterized function $f_\theta : \mathcal{X} \to \mathcal{Y}$ constructed as a composition of affine maps and nonlinear activation functions.

For a feedforward network with $L$ layers,

$$
f_\theta(x) = W_L \sigma\bigl(W_{L-1}\sigma(\cdots \sigma(W_1 x + b_1)\cdots)+b_{L-1}\bigr)+b_L.
$$

Key components:
- Parameters $\theta = \{W_\ell, b_\ell\}$
- Nonlinear activation functions (ReLU, sigmoid, tanh, etc.)
- Training via gradient-based optimization of a loss function $\ell$

Neural networks are often viewed as flexible function approximators in high dimensions.

---

## Approximation Theory of Neural Networks

**What classes of functions can neural networks approximate, and how efficiently?**

Two main themes:
1. **Universality**: Can networks approximate any continuous function on compact sets?
2. **Rates**: How does approximation error scale with network width, depth, or number of parameters?

Approximation theory provides:
- Existence results (universal approximation)
- Quantitative error bounds
- Insights into the role of depth and sparsity

---


### Cybenko Result

[Cybenko (1989)](https://www.scirp.org/reference/referencespapers?referenceid=3331751&utm_source=chatgpt.com) proved that a single hidden-layer neural network with a sigmoid activation function can approximate any continuous function on a compact subset of $\mathbb{R}^d$ arbitrarily well.

More precisely, functions of the form

$$
f(x)=\sum_{j=1}^m a_j \,\sigma(w_j^\top x + b_j)
$$

are dense in $C(K)$ for compact $K \subset \mathbb{R}^d$, provided $\sigma$ is a suitable non-polynomial activation function.

This establishes **universality**, but does not provide rates of approximation or parameter efficiency.

---

### Leshno, Lin, Pinkus and Schocken

[Leshno et al. (1993)](https://www.sciencedirect.com/science/article/pii/S0893608005801315) generalized the universality result by showing:

A feedforward neural network with a non-polynomial activation function has the universal approximation property.

This result clarifies that:
- Universality does not depend on the specific choice of sigmoid
- The key requirement is that the activation function is not a polynomial

This paper provides a structural characterization of when neural networks are universal approximators.

---


### Yarotsky Result on ReLU

[Yarotsky (2017)](https://arxiv.org/abs/1610.01145) analyzed approximation rates for ReLU networks and showed that deep ReLU networks can approximate functions in Sobolev or Hölder spaces with explicit error bounds.

A key takeaway:
- Depth can significantly improve approximation efficiency.
- ReLU networks can approximate certain classes of functions with error scaling like
  
$$
\varepsilon \sim N^{-p/d}
$$

under appropriate smoothness assumptions, where $N$ is the number of parameters and $d$ is the input dimension.

These results connect neural networks to classical nonlinear approximation theory and sparse representations.

---


### Kolmogorov Superposition Theorem (A Cautionary Tale)

The [Kolmogorov superposition theorem (1957)](https://www.mathnet.ru/php/archive.phtml?wshow=paper&jrnid=dan&paperid=22050&option_lang=eng) states that any continuous function  

$$
f : [0,1]^d \to \mathbb{R}
$$

can be represented as a finite superposition of continuous univariate functions and addition:

$$
f(x_1,\dots,x_d)=
\sum_{q=1}^{2d+1}
\Phi_q\left(
\sum_{p=1}^d \psi_{pq}(x_p)
\right),
$$

for suitable continuous functions $\Phi_q$ and $\psi_{pq}$.

At first glance, this appears to provide an extremely strong representation theorem: arbitrary multivariate functions can be built from compositions of one-dimensional functions.

However, this result is often viewed as a **cautionary tale** in approximation theory and machine learning:

- The theorem is existential and nonconstructive.
- The representing functions can be highly irregular and difficult to compute.
- The representation gives little information about stability, learnability, or numerical efficiency.
- It does not provide useful approximation rates or guidance for practical algorithms.

The analogy to neural networks is instructive.  
Universal approximation theorems (such as Cybenko or Leshno et al.) guarantee that networks \(f_\theta\) can approximate any function in principle, but they do **not** by themselves explain:
- how many parameters are required,
- whether training is feasible,
- whether the representation is stable,
- or whether generalization is possible.

For these reasons, modern theory focuses not only on universality but also on **approximation rates, structure, and inductive bias**, which are often more relevant in practice.


## Tutorials

1. Simple MNIST Classification

2. Cybenko Notebook


## Additional links


1. [TensorFlow playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2,2&seed=0.03614&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false). Tinker with neural networks and gain intuition about hidden neuron states, and the training process.

2. [An interactive node-link visualization of convolutional neural networks](https://adamharley.com/nn_vis/)

3. 

## Setting up a Conda Environment for PyTorch

To run the code, first install **Miniconda** or **Anaconda**:

1. Download Miniconda (recommended, lightweight):  
   https://docs.conda.io/en/latest/miniconda.html

2. Follow the installer instructions for your operating system, then open a new terminal.

### Create a PyTorch Environment

Create a new environment (here called `torchenv`) with Python:

```bash
conda create -n torchenv python=3.11
```

Activate the environment:

```bash
conda activate torchenv
```

### Install PyTorch

Install PyTorch using the official instructions from:  
https://pytorch.org/get-started/locally/

For a typical CPU install:

```bash
conda install pytorch torchvision torchaudio -c pytorch
```

(If using CUDA, select the appropriate command from the PyTorch website.)

### Verify Installation

Start Python and check:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```

If no errors appear, the environment is ready to use.



