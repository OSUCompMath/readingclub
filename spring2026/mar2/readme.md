# March 3

### Neural Tangent Kernel (NTK)

**Background:**

A kernel function is a symmetric, positive definite function $k:\mathcal{X}\times\mathcal{X}\to\mathbb{R}$.

A kernel function is the inner product of a Reproducing Kernel Hilbert space (Moore–Aronszajn theorem).

Kernel methods use kernel functions to operate in a high-dimensional, implicit feature space by computing the inner products between the images of all pairs of data in the feature space (Kernel trick).

> **Example:**
> Support vector machine:
> $$y = sign(\sum_{i=1}^n \alpha_i x_i)^Tx +b)$$
> Kernel SVM:
> $$y = sign(\sum_{i=1}^n \alpha_i k(x_i,x) +b)$$

*(See sci-kit learn: [Classifier Comparison](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html))*

> **Example:**
> Similar for ordinary least square regression:
> $$y = m(x) + \epsilon,\quad \epsilon\sim N(0,\sigma^2).$$
> Assume $m$ is linear and solve OLS:
> $$\min_w \sum_{i=1}^n (w^Tx_i - y_i)^2$$
> Then $w = (XX^T)^{-1}Xy$.
> Similarly to SVM, $w = \sum_{i=1}^n \alpha_i x_i = X\alpha$.
> Then, kernel ordinarily least square is given by:
> $$m(x)=k_X(x)K^{-1}y.$$

**Goal:** Study the training dynamics of neural networks. The optimization problem is highly non-convex.

Consider a neural network:
$$f_\theta(x), \quad \theta \in \mathbb{R}^n$$
and some objective function $L(\theta):=\frac{1}{N} \sum_{i=1}^N \ell(f(x_i;\theta),y_i)$.

* **(Neal 1994)** In the limit as the widths of the hidden layers tend to infinity, the network function at initialization, $f_\theta$ converges to a Gaussian Process.
* **(Jacot 2018)** Let $k:\mathbb{R}^d\times\mathbb{R}^d \to \mathbb{R}$ with $k(x,x';\theta)=\nabla_\theta f(x)^T \nabla_\theta f(x)$ be the neural tangent kernel (NTK).

**Deriving the NTK:**
* Consider the gradient descent limit as $t\to 0$, ODE for the evolution of $\frac{d\theta}{dt}=-\frac{1}{N}\sum_{i=1}^N \nabla_\theta f(x_i)\nabla_f l(f,y_i)$
* Neural network change wrt $t$: $\frac{d}{dt}f(x,\theta(t)) = -\frac{1}{N}\sum_{i=1}^N \color{blue}{[\nabla_\theta f(x)^T \nabla_\theta f(x_i)]} \nabla_f l(f,y_i)$.

In (Jacot 2018), it is shown that as $n\to \infty$ the NTK stays constant during training. Then, the neural network at training time $t$ can be given explicitly by solving:

$$\frac{d}{dt} f(x,\theta(t)) = -\frac{1}{N}\sum_{i=1}^N K_\infty(x,x') \nabla_f l(f,y_i) \tag{1}$$

> **Example:**
> If the loss is the MSE loss $\ell(x,y) = \frac{1}{2}(x-y)^2$ then:
> $$f_t(x) = f_0(x) - k(x)K^{-1}(I-\exp(-Kt))(f_0(X)-Y)$$

**(Lee 2019)** shows the correspondence between linearised networks and infinitely wide networks.

**Details:** Linearize around initialization:
$$f_\theta(x) \approx f_{\theta_0}(x) + \nabla_\theta f_{\theta_0}(x)\cdot(\theta-\theta_0).$$

The dynamics of gradient flow using this linearized version are governed by the evolution of $w_t := \theta - \theta_0$. Write $\frac{d}{dt} w$ and $\frac{d}{dt}f_{lin,t}$ and verify that we obtain the same equation as (1).

**In practice:**
