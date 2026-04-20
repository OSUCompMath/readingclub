# Diffusion Sampling: DDPM, Score SDEs, Anderson Reversal, and Tweedie's Lemma

This note gives a mathematically oriented introduction to diffusion sampling. It is written to be readable as a `README.md` on GitHub, while still making the key stochastic-process ideas explicit.

The goals are to explain:

1. the **forward noising process**,
2. the **reverse-time process** used for generation,
3. the relation between **DDPM notation** and **score-SDE notation**,
4. the **Anderson time-reversal theorem** for diffusions, and
5. **Tweedie's lemma**, which turns score estimation into a regression problem.

---

## 1. Two common notational viewpoints: DDPM vs. score SDEs

There are two closely related ways diffusion models are usually presented.

### DDPM viewpoint

In the discrete-time DDPM formulation, one defines a Markov chain

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\!\bigl(x_t; \sqrt{\alpha_t}\,x_{t-1}, (1-\alpha_t)I\bigr),
$$

with

$$
\alpha_t = 1 - \beta_t,
\qquad
\bar{\alpha}_t = \prod_{s=1}^t \alpha_s.
$$

A key closed-form identity is

$$
q(x_t \mid x_0)
=
\mathcal{N}\!\bigl(x_t; \sqrt{\bar{\alpha}_t}\,x_0, (1-\bar{\alpha}_t)I\bigr),
$$

so one may write

$$
x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\varepsilon,
\qquad
\varepsilon \sim \mathcal{N}(0,I).
$$

This is the notation most common in practical implementations.

### Score-SDE viewpoint

In the continuous-time score-based formulation, one defines an Itô SDE

$$
dX_t = f(X_t,t)\,dt + G(X_t,t)\,dW_t,
\qquad t \in [0,T].
$$

Often one specializes to isotropic noise,

$$
G(X_t,t) = g(t)I,
$$

so that

$$
dX_t = f(X_t,t)\,dt + g(t)\,dW_t.
$$

The reverse-time dynamics then involve the score

$$
\nabla_x \log p_t(x),
$$

where \( p_t \) is the density of \( X_t \).

### How they match

The DDPM chain is a discrete approximation to a continuous diffusion. The core ideas are the same in both settings:

- the **forward process** gradually transforms data into noise,
- the **reverse process** transforms noise back into data,
- and the reverse dynamics depend on the **score**.

The DDPM formulation is often easier for implementation; the score-SDE formulation is often cleaner for analysis.

---

## 2. Forward diffusion process

Let

$$
X_0 \sim p_0
$$

be a random variable distributed according to the data law.

A forward diffusion process is an Itô SDE

$$
dX_t = f(X_t,t)\,dt + G(X_t,t)\,dW_t.
$$

For much of this note, we focus on the isotropic case

$$
dX_t = f(X_t,t)\,dt + g(t)\,dW_t.
$$

The law of \( X_t \) is denoted by \( p_t \). The forward process is designed so that for sufficiently large \( T \), the terminal law \( p_T \) is simple, typically close to Gaussian.

### Example: variance-preserving diffusion

A standard linear choice is

$$
dX_t = -\frac{1}{2}\beta(t)X_t\,dt + \sqrt{\beta(t)}\,dW_t.
$$

Conditioned on \( X_0 = x_0 \), this process has the form

$$
X_t = \alpha(t)x_0 + \sigma(t)\varepsilon,
\qquad
\varepsilon \sim \mathcal{N}(0,I),
$$

where

$$
\alpha(t)
=
\exp\!\left(
-\frac{1}{2}\int_0^t \beta(s)\,ds
\right),
\qquad
\sigma^2(t) = 1 - \alpha^2(t).
$$

Hence

$$
p(x_t \mid x_0)
=
\mathcal{N}\!\bigl(x_t; \alpha(t)x_0, \sigma^2(t)I\bigr).
$$

This explicit Gaussian conditional is one of the key reasons diffusion models are trainable.

---

## 3. The score and the Fokker--Planck equation

If \( X_t \) has density \( p_t(x) \), then in the isotropic case the density satisfies the Fokker--Planck equation

$$
\partial_t p_t(x)
=
-\nabla \cdot \bigl(f(x,t)p_t(x)\bigr)
+
\frac{1}{2}g(t)^2 \Delta p_t(x).
$$

The **score** is defined by

$$
s_t(x) := \nabla_x \log p_t(x).
$$

Equivalently,

$$
s_t(x) = \frac{\nabla p_t(x)}{p_t(x)}.
$$

The score points toward regions of higher probability and is precisely the object that enters the reverse-time diffusion.

---

## 4. Reverse-time diffusion

The reverse-time process is the mathematical foundation of diffusion sampling. One starts from a simple terminal law \( p_T \), typically Gaussian, and seeks to evolve backward toward \( p_0 \).

At a formal level, the reverse dynamics are governed by a drift correction involving the score.

In the isotropic setting,

$$
dX_t = f(X_t,t)\,dt + g(t)\,dW_t
$$

reverses to a diffusion whose drift contains the term

$$
-g(t)^2 \nabla_x \log p_t(x).
$$

A precise theorem is given below.

---

## 5. Anderson time-reversal theorem

The following is a standard form of the time-reversal principle underlying score-based diffusion models.

### Theorem (Anderson, informal standard form)

Let \( (X_t)_{t \in [0,T]} \) solve the Itô SDE

$$
dX_t = f(X_t,t)\,dt + G(t)\,dW_t,
$$

where:

1. \( f \) and \( G \) are sufficiently regular,
2. for each \( t \in (0,T] \), the law of \( X_t \) admits a smooth, strictly positive density \( p_t \),
3. the process and coefficients satisfy the integrability and decay assumptions needed to justify time reversal and differentiation under the law.

Define

$$
a(t) := G(t)G(t)^\top.
$$

Then the time-reversed process is again a diffusion, and its drift is

$$
f(x,t) - a(t)\nabla \log p_t(x).
$$

In the isotropic case \( G(t) = g(t)I \), this becomes

$$
f(x,t) - g(t)^2 \nabla \log p_t(x).
$$

### Remarks on the assumptions

The theorem is often stated under somewhat different regularity hypotheses depending on the source. The essential requirements are:

- existence of sufficiently regular transition densities,
- positivity and smoothness of \( p_t \),
- enough integrability to justify reversing the process,
- and enough differentiability to identify the reverse drift.

For diffusion-model applications, the main takeaway is not the sharpest functional-analytic version of the theorem, but the structural fact that **time reversal introduces the score correction**.

### Proof idea

A short heuristic proof goes as follows.

Over a short time interval \( \Delta t \),

$$
X_{t+\Delta t}
\approx
X_t + f(X_t,t)\Delta t + G(t)\sqrt{\Delta t}\,\xi,
\qquad
\xi \sim \mathcal{N}(0,I).
$$

To understand the reverse process, one examines the conditional law of \( X_t \) given \( X_{t+\Delta t} \). By Bayes' rule, the marginal density \( p_t \) enters. Expanding the logarithm of the conditional density to first order in \( \Delta t \) produces an additional drift correction proportional to

$$
-a(t)\nabla \log p_t(x).
$$

That term is exactly what distinguishes the true reverse diffusion from naive reversal of the forward drift.

---

## 6. Why score learning is difficult

The reverse-time sampler requires the score field

$$
s_t(x) = \nabla_x \log p_t(x).
$$

However, the density \( p_t \) is unknown in practice. Even the original data law \( p_0 \) is usually available only through samples. Thus, directly fitting the score seems difficult.

The key simplification is that in diffusion models the conditional law \( p(x_t \mid x_0) \) is explicitly known and Gaussian. This lets one replace direct score estimation by a regression problem.

That is where Tweedie's lemma enters.

---

## 7. Tweedie's lemma

Tweedie's lemma is the basic Gaussian identity that connects denoising to score estimation.

### Theorem (Tweedie)

Let

$$
Y = X + \sigma Z,
\qquad
Z \sim \mathcal{N}(0,I),
$$

where \( Z \) is independent of \( X \), and assume \( Y \) has a differentiable density \( p_Y \). Then

$$
\mathbb{E}[X \mid Y = y]
=
y + \sigma^2 \nabla \log p_Y(y).
$$

Equivalently,

$$
\nabla \log p_Y(y)
=
\frac{\mathbb{E}[X \mid Y = y] - y}{\sigma^2}.
$$

### Interpretation

The score of the noisy variable \( Y \) is determined by the conditional denoiser \( \mathbb{E}[X \mid Y=y] \). Thus, learning to denoise is equivalent to learning the score.

### Proof

The density of \( Y \) is

$$
p_Y(y)
=
\int p_X(x)\,\varphi_\sigma(y-x)\,dx,
$$

where

$$
\varphi_\sigma(y-x)
=
(2\pi \sigma^2)^{-d/2}
\exp\!\left(
-\frac{\|y-x\|^2}{2\sigma^2}
\right).
$$

Differentiate under the integral:

$$
\nabla_y p_Y(y)
=
\int p_X(x)\,\nabla_y \varphi_\sigma(y-x)\,dx.
$$

Since

$$
\nabla_y \varphi_\sigma(y-x)
=
-\frac{y-x}{\sigma^2}\varphi_\sigma(y-x),
$$

it follows that

$$
\nabla_y p_Y(y)
=
-\frac{1}{\sigma^2}
\int (y-x)p_X(x)\varphi_\sigma(y-x)\,dx.
$$

Dividing by \( p_Y(y) \) gives

$$
\nabla \log p_Y(y)
=
-\frac{1}{\sigma^2}
\left(
y - \mathbb{E}[X \mid Y=y]
\right),
$$

which rearranges to

$$
\mathbb{E}[X \mid Y=y]
=
y + \sigma^2 \nabla \log p_Y(y).
$$

---

## 8. Tweedie's lemma in diffusion models

In the forward diffusion process, one typically has

$$
X_t = \alpha(t)X_0 + \sigma(t)\varepsilon,
\qquad
\varepsilon \sim \mathcal{N}(0,I).
$$

Conditioned on \( X_0 \), the variable \( X_t \) is Gaussian. Hence Tweedie's lemma applies directly.

Let

$$
s_t(x_t) := \nabla_{x_t}\log p_t(x_t).
$$

Then

$$
\mathbb{E}[\alpha(t)X_0 \mid X_t = x_t]
=
x_t + \sigma(t)^2 s_t(x_t).
$$

Therefore,

$$
\mathbb{E}[X_0 \mid X_t = x_t]
=
\frac{x_t + \sigma(t)^2 s_t(x_t)}{\alpha(t)}.
$$

So:

- if one knows the score, one can compute the denoised signal,
- and if one learns the denoiser, one can recover the score.

This is the central bridge between stochastic analysis and practical diffusion training.

---

## 9. Denoising score matching

A direct learning objective would fit a network \( s_\theta(x,t) \) to the score \( \nabla \log p_t(x) \), but that target is unavailable.

Instead, one uses the identity

$$
\nabla_{x_t}\log p_t(x_t)
=
\mathbb{E}
\bigl[
\nabla_{x_t}\log p(x_t \mid x_0)
\mid x_t
\bigr].
$$

For Gaussian corruption,

$$
p(x_t \mid x_0)
=
\mathcal{N}\!\bigl(\alpha(t)x_0, \sigma(t)^2 I\bigr),
$$

and therefore

$$
\nabla_{x_t}\log p(x_t \mid x_0)
=
-\frac{x_t - \alpha(t)x_0}{\sigma(t)^2}.
$$

This yields the denoising score matching objective

$$
\mathcal{L}_{\mathrm{DSM}}(\theta)
=
\mathbb{E}
\left[
\left\|
s_\theta(x_t,t)
+
\frac{x_t - \alpha(t)x_0}{\sigma(t)^2}
\right\|^2
\right].
$$

Using

$$
x_t = \alpha(t)x_0 + \sigma(t)\varepsilon,
$$

this becomes

$$
\mathcal{L}_{\mathrm{DSM}}(\theta)
=
\mathbb{E}
\left[
\left\|
s_\theta(x_t,t)
+
\frac{\varepsilon}{\sigma(t)}
\right\|^2
\right].
$$

So score learning is reduced to a supervised regression problem.

---

## 10. Noise-prediction parameterization

In practice, many implementations do not train a score network directly. Instead, they train a network \( \varepsilon_\theta(x_t,t) \) to predict the Gaussian noise in

$$
x_t = \alpha(t)x_0 + \sigma(t)\varepsilon.
$$

Since the score is related to the noise by

$$
\nabla \log p_t(x_t)
\approx
-\frac{\varepsilon_\theta(x_t,t)}{\sigma(t)},
$$

predicting noise is equivalent, up to scaling, to predicting the score.

This gives the practical loss

$$
\mathcal{L}_\varepsilon(\theta)
=
\mathbb{E}
\left[
\|\varepsilon_\theta(x_t,t) - \varepsilon\|^2
\right].
$$

This is the standard objective used in DDPM-style implementations.

---

## 11. Sampling as reverse diffusion

Once the network has learned the score, or equivalently the noise, one simulates the reverse process.

### Continuous-time statement

In score-SDE language, sampling is governed by the reverse-time SDE

$$
dX_t
=
\bigl[
f(X_t,t) - g(t)^2 \nabla \log p_t(X_t)
\bigr]\,dt
+
g(t)\,d\overline{W}_t,
$$

interpreted backward in time.

Equivalently, if one uses a forward sampling variable \( s \) with physical time \( t = T-s \), then

$$
dY_s
=
\bigl[
-f(Y_s,T-s) + g(T-s)^2 \nabla \log p_{T-s}(Y_s)
\bigr]\,ds
+
g(T-s)\,dB_s.
$$

### Discrete-time statement

In DDPM language, one learns a parameterized reverse kernel

$$
p_\theta(x_{t-1} \mid x_t),
$$

whose mean is constructed from the predicted noise or score. Sampling then starts from Gaussian noise at time \( T \) and iteratively applies the learned reverse transition until one reaches an approximate sample from \( p_0 \).

---

## 12. Summary of the chain of ideas

The logic of diffusion models is:

### Forward process

Choose a tractable noising process,

$$
dX_t = f(X_t,t)\,dt + g(t)\,dW_t,
$$

or its discrete DDPM analogue, so that \( p_T \) is simple.

### Reverse process

By Anderson's theorem, the reverse dynamics require the score:

$$
f(x,t) - g(t)^2 \nabla \log p_t(x).
$$

### Tweedie

For Gaussian corruption,

$$
\mathbb{E}[X \mid Y=y]
=
y + \sigma^2 \nabla \log p_Y(y),
$$

so denoising and score estimation are equivalent.

### Training

Because the forward corruption law is explicit,

$$
x_t = \alpha(t)x_0 + \sigma(t)\varepsilon,
$$

one can train on synthetic noisy pairs rather than on unknown density gradients.

### Sampling

Once the score or noise predictor is learned, one simulates the reverse process to generate data.

---

## 13. Useful identities at a glance

### Forward SDE

$$
dX_t = f(X_t,t)\,dt + g(t)\,dW_t
$$

### Reverse SDE

$$
dX_t
=
\bigl[
f(X_t,t) - g(t)^2 \nabla \log p_t(X_t)
\bigr]\,dt
+
g(t)\,d\overline{W}_t
$$

### Gaussian corruption model

$$
X_t = \alpha(t)X_0 + \sigma(t)\varepsilon,
\qquad
\varepsilon \sim \mathcal{N}(0,I)
$$

### Tweedie's identity

$$
\mathbb{E}[X \mid Y=y]
=
y + \sigma^2 \nabla \log p_Y(y)
$$

### Conditional Gaussian score

$$
\nabla_{x_t}\log p(x_t \mid x_0)
=
-\frac{x_t - \alpha(t)x_0}{\sigma(t)^2}
$$

### Denoising score matching objective

$$
\mathbb{E}
\left[
\left\|
s_\theta(x_t,t)
+
\frac{x_t - \alpha(t)x_0}{\sigma(t)^2}
\right\|^2
\right]
$$

### Noise-prediction objective

$$
\mathbb{E}
\left[
\|\varepsilon_\theta(x_t,t) - \varepsilon\|^2
\right]
$$

---

## 14. Final remark

The central mathematical mechanism is simple to state:

- **Anderson reversal** says that reverse-time diffusion requires the score.
- **Tweedie's lemma** says that, under Gaussian corruption, the score can be accessed through denoising.
- Therefore, diffusion modeling turns a difficult density-gradient estimation problem into a tractable regression problem.

That is the bridge from stochastic-process theory to practical diffusion generative modeling.