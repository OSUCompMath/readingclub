# Mean-Field Limits of Neural Network Training: A PDE Perspective on Deep Learning
 
**Date:** March 8, 2026

---

## 1. Introduction: Neural Networks as Interacting Particle Systems

Consider a supervised learning task with data $(x, y) \in \Omega \times \mathbb{R}$, where $\Omega \subset \mathbb{R}^d$. We consider a shallow neural network with $N$ neurons defined by the function
$$f_N(x; \theta) = \frac{1}{N} \sum_{i=1}^N \Psi(x; \theta_i)$$
where $\theta_i \in \mathbb{R}^D$ denotes the parameters of the $i$-th neuron and $\Psi$ is a nonlinear activation function. The objective is to minimize the empirical risk functional

$$E(\theta) = \int_{\Omega} \frac{1}{2} |f_N(x; \theta) - y(x)|^2 d\mu(x)$$

In the **mean-field regime**, we consider Gradient Descent (GD) dynamics where the dynamics are governed by the system of ODEs

$$\frac{d\theta_i}{dt} = - N \nabla_{\theta_i} E(\theta) = -\int_{\Omega} (f_N(x; \theta) - y(x)) \nabla_{\theta_i} \Psi(x; \theta_i) d\mu(x)$$

---

## 2. The Mean-Field Limit and the Wasserstein Metric

We describe the state of the network by the empirical measure $\rho_{N,t} = \frac{1}{N} \sum_{i=1}^N \delta_{\theta_i(t)}$. As $N \to \infty$, the network output becomes an integral transform:

$$f_{\rho}(x) = \int_{\mathbb{R}^D} \Psi(x; \theta) d\rho(\theta).$$

### 2.1 Derivation of the Continuity Equation from Gradient Flow
Any gradient flow of the form 

$$\dot{\theta}_i = v(t, \theta_i)$$

can be reformulated as a continuity equation for the density. To derive the evolution of the empirical measure $\rho_{N,t}$, we test it against an arbitrary smooth, compactly supported function $\phi \in C_c^\infty(\mathbb{R}^D)$. The time derivative of the pairing $\langle \rho_{N,t}, \phi \rangle$ is:

$$\frac{d}{dt} \langle \rho_{N,t}, \phi \rangle = \frac{d}{dt} \left( \frac{1}{N} \sum_{i=1}^N \phi(\theta_i(t)) \right) = \frac{1}{N} \sum_{i=1}^N \nabla_\theta \phi(\theta_i(t)) \cdot \dot{\theta}_i(t).$$

Substituting the velocity $v(t, \theta_i(t)) = \dot{\theta}_i(t)$, we obtain:

$$\frac{d}{dt} \langle \rho_{N,t}, \phi \rangle = \int_{\mathbb{R}^D} \nabla_\theta \phi(\theta) \cdot v(t, \theta) d\rho_{N,t}(\theta).$$

By the definition of the distributional derivative, this identity is exactly the weak formulation of the **continuity equation**:

$$\partial_t \rho_{N,t} + \nabla_\theta \cdot (\rho_{N,t} v) = 0.$$

This derivation shows that the conservation of mass is implicitly satisfied by the particle dynamics.

Define the continuous energy functional $\mathcal{E}$ on the space of probability measures $\mathcal{P}_2(\mathbb{R}^D)$:

$$\mathcal{E}[\rho] = \frac{1}{2} \int_{\Omega} |f_\rho(x) - y(x)|^2 d\mu(x).$$

The first variation (Fréchet derivative) $\frac{\delta \mathcal{E}}{\delta \rho}$ is found by considering a perturbation $\chi$:

$$\delta \mathcal{E}[\rho] (\chi) = \left. \frac{d}{d\epsilon} \right|_{\epsilon=0} \mathcal{E}[\rho + \epsilon \chi] = \int_{\mathbb{R}^D} \left[ \int_{\Omega} (f_\rho(x) - y(x)) \Psi(x; \theta) d\mu(x) \right] d\chi(\theta).$$


The term in the square brackets is the potential function $\Phi_\rho(\theta) = \frac{\delta \mathcal{E}}{\delta \rho}(\theta)$. From Eq. (1), the GD dynamics for a single particle are:

$$\dot{\theta}_i = -\nabla_\theta \Phi_\rho(\theta_i).$$

This implies the Eulerian velocity field is $v(t,\theta) = -\nabla_\theta \Phi_\rho(\theta)$. Substituting $v$ into the conservation law yields the **McKean-Vlasov equation**:

$$\partial_t \rho(\theta, t) = \nabla_\theta \cdot \left( \rho(\theta, t) \nabla_\theta \int_{\Omega} (f_\rho(x) - y(x)) \Psi(x; \theta) d\mu(x) \right).$$

This characterizes the gradient flow $\partial_t \rho = -\text{grad}_{W_2} \mathcal{E}[\rho]$ in the Wasserstein space. Here, the measure $\rho$ evolves significantly, allowing for **feature learning**.

---

## 3. Interesting Applications

* **Mean-Field Analysis of Transformers:** Recent work (e.g., Geshkovski, B. et al. 2025) extends this to Self-Attention. By treating tokens or attention heads as interacting particles, Transformer training is described as a flow in measure space, explaining how attention maps "focus" on specific structures.

---

## 4. Open Problems

1.  **Global Convergence for Deep Architectures:** While well-understood for two-layer networks, extending this to deep, multi-layer networks remains a challenge due to hierarchical coupling.
2.  **Quantifying Convergence Rates:** Most results are qualitative; obtaining sharp rates of convergence for non-convex functionals in Wasserstein space is an active area.
3.  **The Role of Noise (SGD):** Adding noise leads to a second-order term (Laplacian) in the PDE, resulting in a **Fokker-Planck** equation. The entropy-energy interaction in non-linear settings is not yet fully characterized.

---

### Selected References

- Rigollet, P. (2025).  
  **The mean-field dynamics of transformers.**  
  *arXiv preprint arXiv:2512.01868.*

- Geshkovski, B., Letrouit, C., Polyanskiy, Y., & Rigollet, P. (2025).  
  **A mathematical perspective on transformers.**  
  *Bulletin of the American Mathematical Society*, 62(3), 427–479.

- Ambrosio, L., Gigli, N., & Savaré, G. (2005).  
  **Gradient flows: In metric spaces and in the space of probability measures.**  
  Springer.

- Chizat, L., & Bach, F. (2018).  
  **On the global convergence of gradient descent for over-parameterized models using optimal transport.**  
  *Advances in Neural Information Processing Systems*, 31.

- Mei, S., Montanari, A., & Nguyen, P.-M. (2018).  
  **A mean field view of the landscape of two-layer neural networks.**  
  *Proceedings of the National Academy of Sciences*, 115(33), E7665–E7671.



# Neural Collapse — Notes

Based on slides by Vlad Kobzar (OSU ML Paper Club).

---

# 1. Supervised Classification Setup

We consider a standard supervised classification problem.

- Input space  

$$
X \subset \mathbb{R}^r
$$

- Label set  

$$
[n] := \{1,\dots,n\}
$$

- Training data  

$$
\{(x_i, k_i)\}_{i=1}^m \subset X \times [n]
$$

The goal is to learn a classifier

$$
N : X \rightarrow [n].
$$

Deep neural classifiers typically consist of two parts:

1. a **feature map**
2. a **linear classifier**

---

# 2. Feature Representation

Typical classifiers take the form

$$
\text{argmax}_k (L \circ F(x))_k
$$

where

- feature map  

$$
F : X \rightarrow \mathbb{R}^d
$$

- affine classifier  

$$
L : \mathbb{R}^d \rightarrow \mathbb{R}^n
$$

In deep learning:

- the feature map is produced by a neural network  

$$
F_\theta(x)
$$

- the weights $\theta$ are learned from training data.

Thus the network learns a **representation space** $\mathbb{R}^d$ in which classification becomes linearly separable.

---

# 3. Empirical Risk Minimization

Training is performed via empirical risk minimization using cross-entropy loss.

Let

$$
y_i := e_{k_i}
$$

be the **one-hot encoding** of the label.

The loss is

$$
\mathcal{L}(\theta,L)=
-\sum_{i \in [m]} \sum_{j \in [n]}
(y_i)_j
\log\left((\text{softmax}\circ L \circ F_\theta(x_i))_j\right).
$$

Despite the nonconvex nature of this optimization problem, gradient-based methods typically find good solutions in practice.

---

# 4. Neural Collapse

Neural collapse is an empirical phenomenon observed during the **terminal phase of training**, once training error reaches zero.

Assume

$$
\text{argmax}_k (L\circ F(x_i))_k = k_i
$$

for all training points.

Several geometric properties then emerge.

---

## NC1 — Within-class feature collapse

For samples with the same label

$$
k_i = k_j
$$

their features converge to a common vector

$$
F(x_i) = F(x_j) = h_{k_i}.
$$

Thus each class becomes represented by a **single feature prototype**

$$
\{h_1,\dots,h_n\}.
$$

---

## NC2 — Simplex structure of class means

If the dataset is **class balanced**

$$
|\{i : k_i = j\}| = N
$$

then the class prototypes form a **regular simplex**

$$
\{h_j\}_{j=1}^n \subset \mathbb{R}^d.
$$

Meaning

- pairwise distances are equal
- the vectors are maximally separated.

This requires

$$
d \ge n-1.
$$

---

# 5. Alignment of Classifier and Features

The affine classifier has components

$$
L(h)_k = \langle w_k, h \rangle + b_k.
$$

Under neural collapse:

- the classifier weights also form a simplex
- they align with centered feature prototypes.

Specifically

$$
w_k \propto
h_k - \frac{1}{n}\sum_{i=1}^n h_i.
$$

Thus

- classifier weights
- feature prototypes

share the **same geometric structure**.

---

# 6. Unconstrained Features Model

Mixon et al. (2022) propose a simplified theoretical model.

Idea:

Treat the feature vectors as **free optimization variables**

$$
h_i \in \mathbb{R}^d
$$

instead of outputs of a neural network.

This is motivated by the fact that deep networks are typically **highly overparameterized**, so many feature configurations are realizable.

Advantages:

- decouples the geometry from the architecture
- simplifies analysis of neural collapse.

---

# 7. Orthoplex Regime

When

$$
d + 2 \le n \le 2d
$$

the optimal geometry transitions to an **orthoplex configuration**.

In this regime the solution corresponds to a **softmax code**.

---

# 8. Softmax Codes

A **softmax code** is a configuration of vectors maximizing

> the minimum distance between a point and the convex hull of the others.

Geometrically this yields **maximally separated representations** under the softmax loss.

Consequences:

- classes are well separated
- classification margins are large.

---

# 9. Interpretation

The neural collapse structure implies

- learned features become **highly symmetric**
- representations become **maximally separated**
- classifier weights align with feature prototypes.

This geometric structure may help explain

- strong generalization
- efficient representation learning
- stable optimization.

---

# 10. Open Questions

Several theoretical questions remain open.

For example

$$
n > 2d
$$

the orthoplex analysis no longer applies.

This regime is common in practice:

- **face recognition** (many identities = classes)
- **language models** (vocabulary size ≫ embedding dimension)

Understanding neural collapse in these regimes remains an active research direction.

---

# References

Papyan, Han, Donoho (2020)  
[*Prevalence of Neural Collapse during the Terminal Phase of Deep Learning Training.*](https://arxiv.org/abs/2008.08186)

Mixon, Parshall, Pi (2022)  
[*Neural Collapse with Unconstrained Features.*](https://arxiv.org/abs/2011.11619)
