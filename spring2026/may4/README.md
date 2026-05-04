# Stochastic Interpolants as a Unifying Framework

Reading-group notes on **Albergo, Boffi & Vanden-Eijnden** — *Stochastic Interpolants: A Unifying Framework for Flows and Diffusions*, *JMLR* 26 (2025).
Paper: [jmlr.org/papers/v26/23-1605](https://www.jmlr.org/papers/volume26/23-1605/23-1605.pdf).

A typeset PDF of these notes (with the same content but tighter layout) lives next to this file: [`notes.pdf`](notes.pdf).

---

## 1. The framework

**Skeleton.** Most modern dynamical generative models pick:

1. a time-dependent density $\rho(t,x)$ on $[0,1]\times\mathbb{R}^d$ with $\rho(0)=\rho_0$, $\rho(1)=\rho_1$;
2. a drift $b(t,x)$ such that $\rho(t)$ is the law of either an ODE $\dot X_t = b(t,X_t)$ or an SDE $dX_t = b(t,X_t)\,dt + \sqrt{2\varepsilon}\,dW_t$;
3. and train $b$ (and possibly the score $s = \nabla \log \rho$) by some loss.

The interpolant paper rewrites all three steps in terms of *one* stochastic process and *three orthogonal design choices*.

**Two-sided interpolant (Def. 1).** For arbitrary $\rho_0, \rho_1$,

$$x_t \;=\; I(t, x_0, x_1) \;+\; \gamma(t)\, z, \qquad (x_0, x_1) \sim \nu,\ \; z \sim \mathcal{N}(0, I_d),\ \; t \in [0, 1]. \quad (\star)$$

The function $I : [0,1] \times \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}^d$ is the *interpolation map* satisfying $I(0, x_0, x_1) = x_0$ and $I(1, x_0, x_1) = x_1$. The scalar $\gamma \in C^2([0,1])$ is the *latent-noise schedule*: $\gamma(0) = \gamma(1) = 0$, $\gamma(t) > 0$ on $(0, 1)$. The coupling $\nu$ has marginals $\rho_0, \rho_1$ (often $\nu = \rho_0 \otimes \rho_1$).

**One-sided interpolant (Def. 32).** When $\rho_0 = \mathcal{N}(0, I_d)$ the role of $x_0$ collapses into $z$:

$$x_t^{\mathrm{os}} \;=\; \alpha(t)\, z + J(t, x_1), \qquad x_1 \sim \rho_1,\; z \sim \mathcal{N}(0, I_d). \quad (\dagger)$$

Here $J(t, x_1)$ is the *data-side encoder* ($J(0,x_1)=0$, $J(1,x_1)=x_1$) and $\alpha(t)$ is the *noise envelope* ($\alpha(0)=1$, $\alpha(1)=0$, $\alpha(t)>0$ on $[0,1)$). Equivalently, $(\dagger)$ is $(\star)$ with $I(t,x_0,x_1) = \delta(t)x_0 + J(t,x_1)$ and $\delta^2 + \gamma^2 = \alpha^2$ — but working directly with $(\dagger)$ avoids carrying two redundant noise sources.

**Spatially linear specializations.** The most-studied subfamily takes

$$\text{two-sided linear: } \; x_t^{\mathrm{lin}} = \alpha(t) x_0 + \beta(t) x_1 + \gamma(t) z, \qquad \text{one-sided linear: } \; x_t^{\mathrm{os,lin}} = \alpha(t) z + \beta(t) x_1,$$

with appropriate boundary conditions on $\alpha, \beta, \gamma$. Many existing methods are exactly one of these.

### What each piece of the framework is for

| Term | Role |
|---|---|
| $I(t, x_0, x_1)$ | **Data-side path.** Carries the sample from $x_0$ at $t=0$ to $x_1$ at $t=1$. Choice of $I$ (linear, trigonometric, encoding-decoding, …) shapes the bridge $\rho(t)$. |
| $\gamma(t)\, z$ | **Latent-noise schedule.** Convolves $\rho(t)$ with $\mathcal{N}(0, \gamma^2 I_d)$ on $(0,1)$: smooths the bridge, kills spurious intermediate modes, makes $b, s, \eta_z$ regular, and enables Tweedie. Boundary $\gamma(0) = \gamma(1) = 0$ leaves the marginals exact. |
| $(x_0, x_1) \sim \nu$ | **Coupling.** Joint law on the endpoints; default is $\rho_0 \otimes \rho_1$. Data-aware couplings (minibatch OT) give straighter trajectories. |
| $\varepsilon(t) \ge 0$ | **Sampler diffusion.** Inference-time knob only: switches the generator between ODE ($\varepsilon = 0$), forward SDE, and backward SDE. Does *not* change $\rho(t)$. |
| $b(t, x)$ | **Velocity field** of the probability flow ODE; advects $\rho(t)$ in the continuity equation. Drift target for CFM, FM, rectified flow. |
| $\eta_z(t, x)$ | **Denoiser** (Tweedie posterior mean of the noise $z$). Bounded-variance regression target; what DDPM and EDM-$\epsilon$ learn. |
| $s(t, x)$ | **Score** $\nabla \log \rho(t,x) = -\eta_z / \gamma$. Needed for SDE drift correction $b \pm \varepsilon s$; target of SBDM/DSM. |

### The three core analytical objects

For $\rho(t,x)$ the density of $x_t$, define

$$b(t,x) = \mathbb{E}[\partial_t I + \dot\gamma\, z \mid x_t = x], \quad \eta_z(t,x) = \mathbb{E}[z \mid x_t = x], \quad s(t,x) = \nabla \log \rho(t,x) = -\eta_z / \gamma.$$

### Why this is a continuity equation (Thm. 6, sketch)

Pick a test function $\phi \in C_0^\infty(\mathbb{R}^d)$. By definition of $\rho(t)$, $\int \phi\,\rho\,dx = \mathbb{E}[\phi(x_t)]$. Differentiate in $t$, use the chain rule on $x_t = I(t, x_0, x_1) + \gamma(t) z$, and apply the tower property of conditional expectation:

$$\int \phi\,\partial_t \rho\,dx \;=\; \mathbb{E}[\nabla\phi(x_t) \cdot \dot x_t] \;=\; \mathbb{E}\bigl[\nabla\phi(x_t) \cdot \mathbb{E}[\dot x_t \mid x_t]\bigr] \;=\; \int \nabla\phi \cdot b\,\rho\,dx \;=\; -\!\int \phi\,\nabla\!\cdot\!(b\rho)\,dx,$$

using integration by parts at the last step. Since $\phi$ was arbitrary,

$$\partial_t \rho + \nabla\!\cdot\!(b\,\rho) = 0.$$

This is the transport (continuity) equation: by construction, $b$ is the velocity field that advects $\rho(t)$ from $\rho_0$ at $t=0$ to $\rho_1$ at $t=1$, which is why the ODE $\dot X_t = b(t, X_t)$ generates the same marginals as the interpolant $x_t$.

### Three quadratic losses

All three objects are simulation-free quadratic-loss minimizers (Thms. 6–8):

$$\mathcal{L}_b[\hat b] = \int_0^1 \mathbb{E}\!\left[\tfrac{1}{2}\|\hat b\|^2 - (\partial_t I + \dot\gamma z) \cdot \hat b\right] dt,$$

$$\mathcal{L}_{\eta_z}[\hat\eta] = \int_0^1 \mathbb{E}\!\left[\tfrac{1}{2}\|\hat\eta\|^2 - z \cdot \hat\eta\right] dt,$$

$$\mathcal{L}_s[\hat s] = \int_0^1 \mathbb{E}\!\left[\tfrac{1}{2}\|\hat s\|^2 + \gamma^{-1} z \cdot \hat s\right] dt.$$

You sample $(t, x_0, x_1, z)$, form $x_t$, take a gradient step. No ODE/SDE simulation needed.

### Three equivalent generators of the same $\rho(t)$

$$\dot X_t = b \quad (\text{ODE}), \qquad dX_t^F = (b + \varepsilon s)\,dt + \sqrt{2\varepsilon}\,dW_t, \qquad dX_t^B = (b - \varepsilon s)\,dt + \sqrt{2\varepsilon}\,dW_t^B.$$

The diffusion $\varepsilon(t) \ge 0$ is a *post-training* knob: it changes the sampler but not the marginals $\rho(t)$.

---

## 2. Reading the three losses

**Each loss is a least-squares regression.** Completing the square shows that $\mathcal{L}_b, \mathcal{L}_{\eta_z}, \mathcal{L}_s$ are each fitting a network to a specific per-sample target (constants $C_\bullet$ do not depend on the network):

$$\mathcal{L}_b[\hat b] = \tfrac{1}{2}\int_0^1 \mathbb{E}\,\|\hat b(t, x_t) - \dot x_t\|^2\,dt + C_b, \quad \text{where} \quad \dot x_t = \partial_t I + \dot\gamma\, z,$$

$$\mathcal{L}_{\eta_z}[\hat\eta_z] = \tfrac{1}{2}\int_0^1 \mathbb{E}\,\|\hat\eta_z(t, x_t) - z\|^2\,dt + C_\eta,$$

$$\mathcal{L}_s[\hat s] = \tfrac{1}{2}\int_0^1 \mathbb{E}\,\|\hat s(t, x_t) + \gamma^{-1}(t)\, z\|^2\,dt + C_s.$$

So $\mathcal{L}_b$ regresses against the per-sample *velocity* $\dot x_t$; $\mathcal{L}_{\eta_z}$ against the *noise* $z$; and $\mathcal{L}_s$ against $-z/\gamma$, which equals the conditional log-density gradient $\nabla_{x_t} \log p_t(x_t \mid x_0, x_1)$ (Vincent's denoising score-matching identity).

**The three targets behave very differently as $t \to 0, 1$.**

| Target | Variance / behavior | Used in practice for |
|---|---|---|
| $\dot x_t = \partial_t I + \dot\gamma\, z$ | bounded for smooth $I, \gamma$ | velocity training (CFM, FM, rectified flow) |
| $z$ | unit variance, uniformly in $t$ | denoiser training (DDPM, EDM-$\epsilon$) |
| $-z / \gamma$ | $\Theta(\gamma^{-2}) \to \infty$ as $t \to 0, 1$ | score matching (SBDM, DSM) |

The denoiser target $z$ is the only one with conditional variance bounded *uniformly* on $[0, 1]$ — the technical reason DDPM-style training is so robust, and why Sec. 6.1 of the paper explicitly recommends learning $\eta_z$ rather than $s$. The score loss can be salvaged by *antithetic sampling* $x_t^\pm = I \pm \gamma z$ (Eq. 6.5), which kills the leading $\gamma^{-1}$ term in the variance, but $\eta_z$ avoids the issue entirely.

**Constraint algebra (spatially-linear case).** For $x_t^{\mathrm{lin}} = \alpha(t) x_0 + \beta(t) x_1 + \gamma(t) z$, the three conditional means $\eta_0, \eta_1, \eta_z$ are not independent. The identity $\mathbb{E}[x_t \mid x_t = x] = x$ gives

$$\alpha(t)\,\eta_0(t,x) + \beta(t)\,\eta_1(t,x) + \gamma(t)\,\eta_z(t,x) = x,$$

and differentiating $x_t$ in $t$ gives the velocity factorization

$$b(t,x) = \dot\alpha(t)\,\eta_0(t,x) + \dot\beta(t)\,\eta_1(t,x) + \dot\gamma(t)\,\eta_z(t,x).$$

**Learning any two conditional means determines the third — and the velocity, and the score.** In the one-sided case ($\rho_0 = \mathcal{N}(0, I)$, no $\eta_0$), *one network suffices*. This is precisely why DDPM, despite training a single $\hat\epsilon_\theta$, yields both DDIM (deterministic) and ancestral sampling (stochastic): the trained denoiser is statistically sufficient for both samplers.

**What each method actually trains, and why.**

- **Rectified flow / CFM:** $\hat b$ via $\mathcal{L}_b$. ODE sampling needs only $b$; target $\dot x_t$ is bounded; no score needed.
- **DDPM ($\epsilon$-prediction):** $\hat\eta_z$ via $\mathcal{L}_{\eta_z}$. Best-conditioned target. Velocity recovered at inference via $b = \dot\alpha \eta_0 + \dot\beta \eta_1 + \dot\gamma \eta_z$ plus the constraint; score via $s = -\eta_z / \gamma$.
- **SBDM (Song et al.):** $\hat s$ via $\mathcal{L}_s$ (DSM). Endpoint blow-up handled by truncating $t \in [t_0, T]$ — small but nonzero bias.
- **"$x_1$-prediction"** (used in some DDPMs, EDM): $\hat\eta_1 = \mathbb{E}[x_1 \mid x_t]$ via the analog of $\mathcal{L}_{\eta_z}$. Same regression class as DDPM in a different basis; well-conditioned where $\beta(t)$ stays bounded away from $0$.
- **$v$-prediction (Salimans–Ho '22):** regresses onto $v_t = \dot\beta(t) x_1 - \dot\alpha(t) z$ (with the trig schedule). Designed precisely so the target has unit variance at every $t$ — a different basis for the same 2D regression subspace spanned by $(\eta_z, \eta_1)$.
- **EDM (Karras et al. '22):** fits a denoiser through a time-dependent affine preconditioning $c_{\mathrm{skip}}(t) x + c_{\mathrm{out}}(t) F_\theta(c_{\mathrm{in}}(t) x, c_{\mathrm{noise}}(t))$ chosen so the network output has unit variance at every $t$. The numerically optimal coordinate on the same regression manifold.

**Punchline.** In the one-sided spatially-linear case, all of these methods solve the *same* 2-dimensional regression problem (the joint $(\eta_z, \eta_1)$ task with a linear constraint to $x_t$). They differ only in which basis of the target space is used and which time-weighting is applied. The interpolant framework makes the equivalence algebraic and identifies the uniformly-bounded denoiser $\eta_z$ as the canonical choice.

---

## 3. Dictionary at a glance

| Method | $\rho_0$ | Path: $I$ (or $J, \alpha, \beta$) and $\gamma$ | $\varepsilon$ | Trained target |
|---|---|---|---|---|
| CNF / FFJORD | $\mathcal{N}(0, I)$ | one-sided, any $J$, $\alpha$; $\gamma \equiv 0$ | $0$ | $b_\theta$ via MLE w/ adjoint |
| Rectified flow (Liu '23b) | arbitrary | $I = (1-t) x_0 + t x_1$, $\gamma \equiv 0$ | $0$ | $b$ via $\mathcal{L}_b$ |
| Albergo & V.E. '23 | arbitrary | general $I$, $\gamma \equiv 0$ | $0$ or $> 0$ | $b$ (and $s$) via $\mathcal{L}_b, \mathcal{L}_s$ |
| CFM, OT path (Lipman '23) | $\mathcal{N}(0, I)$ | one-sided linear, $\alpha = 1 - (1-\sigma_{\min}) t$, $\beta = t$ | $0$ | $b$ via cond. FM loss $\equiv \mathcal{L}_b$ |
| DDPM (Ho '20) | $\mathcal{N}(0, I)$ | one-sided linear, $\alpha = \sqrt{1 - \bar\alpha_t}$, $\beta = \sqrt{\bar\alpha_t}$ (discrete) | implicit | $\hat\epsilon_\theta \equiv \hat\eta_z$ via $\mathcal{L}_{\eta_z}$ |
| VP-SBDM (Song '21) | $\mathcal{N}(0, I)$ | one-sided linear, $\alpha(t) = \sqrt{1 - t^2}$, $\beta(t) = t$ after $t = e^{-\tau}$ | $\varepsilon_t > 0$ | score $s_\theta$ via $\mathcal{L}_s$ (DSM) |
| FM, general base (Tong '23) | arbitrary | two-sided linear, $\gamma \equiv 0$ or small | $0$ | $b$ via cond. FM $\equiv \mathcal{L}_b$ |
| Stochastic bridges (Peluchetti '22) | arbitrary | $I + \sqrt{2a}\, B_t$ (Brownian bridge) $\equiv \gamma = \sqrt{2a t (1-t)}$ | implicit | $b$ via Doob-$h$ |
| Stochastic localization | $\delta_0$ | after time change: one-sided $\alpha = 1$, $\beta = t$, $\gamma = \sqrt{t}$ | $1$ | $\mathbb{E}[x_1 \mid x_t] \equiv \eta_1$ |
| Schrödinger bridge | arbitrary | $I^\star$ optimized via Thm. 41 max-min | $> 0$ fixed | $(b, s)$ at $I^\star$ |

**What unifies all rows.** The interpolant paper supplies a single recipe: *pick the path $(I, \gamma)$, write down $\mathcal{L}_b$ and/or $\mathcal{L}_{\eta_z}$, fit, and choose $\varepsilon$ at sampling time.* The next section walks through each row in detail.

---

## 4. Each method as a special case

### 4.1 Continuous normalizing flows / FFJORD (Chen et al. '18)

A CNF parameterizes a velocity field $b_\theta(t, x)$ and *defines* $\rho(t)$ as the law of $\dot X_t = b_\theta(t, X_t)$ with $X_0 \sim \mathcal{N}(0, I)$. Training maximizes $\log p_\theta(x_1)$ via the continuous change-of-variables formula: $\log p_\theta(x_1) = \log \mathcal{N}(X_0; 0, I) - \int_0^1 \nabla\!\cdot b_\theta(\tau, X_\tau)\,d\tau$, with $X_\cdot$ obtained by simulating the ODE backward from $x_1$.

**Map to interpolants.** CNF is the case $\rho_0 = \mathcal{N}(0, I)$, one-sided, with $\gamma \equiv 0$, $\varepsilon = 0$, generator the probability flow $\dot X_t = b(t, X_t)$. The *model class is identical*; what differs is training. CNFs use MLE through the ODE adjoint (which requires simulating the ODE every step and a Hutchinson estimator for $\nabla \cdot b$). Interpolants use the simulation-free loss $\mathcal{L}_b$, whose minimizer is the $b$ that generates whatever $\rho(t)$ the chosen $J$ prescribes. So: same flows, drastically cheaper training, at the cost of fixing $\rho(t)$ a priori.

### 4.2 Rectified flow (Liu et al. '23b) and the original Albergo–V.E. '23 interpolant

Rectified flow takes $\rho_0, \rho_1$ arbitrary, $\nu = \rho_0 \otimes \rho_1$, and the linear path $x_t = (1-t) x_0 + t x_1$, no latent. It learns $b_\theta(t, x)$ by minimizing $\mathbb{E}\|b_\theta(t, x_t) - (x_1 - x_0)\|^2$, then samples $\dot X_t = b_\theta$.

**Map.** Two-sided spatially linear with $\alpha = 1 - t$, $\beta = t$, $\gamma \equiv 0$. The squared loss above is exactly $\mathcal{L}_b$ (rewriting $\tfrac{1}{2}\|b\|^2 - (x_1 - x_0) \cdot b = \tfrac{1}{2}\|b - (x_1 - x_0)\|^2 - \tfrac{1}{2}\|x_1 - x_0\|^2$; the data-only term is constant in $\theta$).

The *rectification* step ("ReFlow") runs the learned flow to produce paired $(x_0, X_{t=1}(x_0))$, redefines $\nu$ as that joint, refits. **Theorem 47** of the paper proves that under unconstrained $b$ this leaves the end-to-end map unchanged — only straightening trajectories, which is useful for cheaper integration but adds bias only if $b$ is constrained (e.g. to a gradient field, in which case iterating converges to the OT map). The original A.–V.E. '23 paper allows arbitrary $I$ but still no latent variable; the 2025 paper's main novelty over '23 is the addition of the $\gamma(t) z$ term.

### 4.3 Conditional flow matching (Lipman et al. '23)

CFM defines a *conditional* probability path $p_t(x \mid x_1) = \mathcal{N}(x;\, \mu_t(x_1),\, \sigma_t^2 I_d)$ between $\mathcal{N}(0, I)$ at $t = 0$ and a tight bump at $x_1$ at $t = 1$. The conditional vector field that pushes this path forward is, by direct computation,

$$u_t(x \mid x_1) = \dot\mu_t(x_1) + \frac{\dot\sigma_t}{\sigma_t}\bigl(x - \mu_t(x_1)\bigr).$$

The CFM loss is $\mathbb{E}_{t, x_1, p_t(\cdot \mid x_1)}\|v_\theta(t, x) - u_t(x \mid x_1)\|^2$; Lipman et al. show its minimizer is the *marginal* velocity field $u_t(x) = \mathbb{E}[u_t(x \mid x_1) \mid x_t = x]$.

**Map.** The Gaussian conditional sample is $x_t = \sigma_t z + \mu_t(x_1)$ with $z \sim \mathcal{N}(0, I)$. With $\mu_t(x_1) = \beta(t) x_1$ (the "OT path" uses $\beta(t) = t$, $\sigma_t = 1 - (1-\sigma_{\min}) t$), this is exactly the *one-sided spatially-linear* interpolant $x_t^{\mathrm{os, lin}} = \alpha(t) z + \beta(t) x_1$ with $\alpha = \sigma_t$. The conditional vector field is the per-sample integrand of $\mathcal{L}_b$:

$$\partial_t I + \dot\gamma z\,\big|_{(\dagger)} \;=\; \dot\beta(t) x_1 + \dot\alpha(t) z \;=\; \dot\mu_t(x_1) + \frac{\dot\sigma_t}{\sigma_t}\bigl(x_t - \mu_t(x_1)\bigr) \;=\; u_t(x_t \mid x_1),$$

where the middle equality uses $z = (x_t - \mu_t(x_1)) / \sigma_t$. So CFM-loss $\equiv \mathcal{L}_b$ on the same path, and CFM's marginal vector field equals the interpolant velocity $b$. Tong et al. '23's extension to non-Gaussian $\rho_0$ corresponds to the two-sided linear interpolant $\alpha(t) x_0 + \beta(t) x_1$.

### 4.4 Score-based diffusion (VP-SBDM, Song et al. '21)

Variance-preserving SBDM runs the OU forward process $dZ_\tau = -Z_\tau\,d\tau + \sqrt{2}\,dW_\tau$ from $Z_0 = x_1 \sim \rho_1$, with marginal $Z_\tau \mid x_1 \sim \mathcal{N}(e^{-\tau} x_1, (1 - e^{-2\tau}) I_d)$ for $\tau \in [0, \infty)$. Training fits $s_\theta \approx \nabla \log p_\tau$ by denoising score matching:

$$\mathbb{E}\|s_\theta(\tau, Z_\tau) + (Z_\tau - e^{-\tau} x_1) / (1 - e^{-2\tau})\|^2.$$

Sampling reverses the SDE on a truncated interval $[\tau_0, T]$ — both endpoints introducing bias.

**Map.** Reparameterize $t = e^{-\tau} \in (0, 1]$, group the noise:

$$Z_{\tau = -\log t} \;=\; e^{-\tau} x_1 + \sqrt{1 - e^{-2\tau}}\, z \;=\; t\, x_1 + \sqrt{1 - t^2}\, z \;=\; \alpha(t) z + \beta(t) x_1, \qquad \alpha = \sqrt{1 - t^2},\ \beta = t.$$

This is one-sided linear with $\alpha, \beta$ specified. Once you write $b$ in terms of the score (Sec. 5.1), $b(t, x) = t s(t, x) + \eta_1^{\mathrm{os}}(t, x)$, which is *nonsingular at $t = 0$*. The standard reverse SDE on the same time interval is $dZ_t^B = t^{-1} Z_t^B\,dt + \cdots$ — that $1/t$ is what forces truncation in original SBDM. **Conclusion:** VP-SBDM is one-sided linear interpolant with the SBDM-specific $(\alpha, \beta)$, and the truncation bias is an artifact of working at the SDE level instead of the interpolant level. Score-matching loss = $\mathcal{L}_s$.

**Why the finite-time framework contains an "infinite-time" process.** This looks paradoxical at first: the OU clock $\tau \in [0, \infty)$ only reaches $\mathcal{N}(0, I)$ asymptotically, yet the interpolant lives on the bounded interval $[0, 1]$. The resolution is that $[0, \infty)$ is homeomorphic to $(0, 1]$ via $t = e^{-\tau}$, which compresses "$\tau = \infty$" into the boundary point $t = 0$. The conditional marginal becomes $\mathcal{N}(t x_1, (1 - t^2) I_d)$, which equals $\mathcal{N}(0, I) = \rho_0$ *exactly* at $t = 0$. Reparameterizing the SDE itself introduces a $1/t$ singularity in the drift (which is why standard SBDM truncates), but the probability-flow velocity $b = t s + \eta_1^{\mathrm{os}}$ stays bounded because the SDE singularity cancels against the score singularity. So the framework does not contain an infinite-time process — it contains the *same path of marginals* that OU generates, written in a finite-time coordinate system.

### 4.5 DDPM (Ho et al. '20)

DDPM is the discrete-time variant: pick $\bar\alpha_t \in [0, 1]$, sample $x_t = \sqrt{\bar\alpha_t}\, x_1 + \sqrt{1 - \bar\alpha_t}\, \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$, train $\hat\epsilon_\theta(x_t, t)$ to predict $\epsilon$:

$$\mathcal{L}_{\mathrm{DDPM}}(\theta) = \mathbb{E}_{t, x_1, \epsilon}\,\|\hat\epsilon_\theta(x_t, t) - \epsilon\|^2.$$

**Map.** Identify $\beta(t) = \sqrt{\bar\alpha_t}$, $\alpha(t) = \sqrt{1 - \bar\alpha_t}$, $\epsilon \equiv z$, $\hat\epsilon_\theta \equiv \hat\eta_z$. Then DDPM's loss is $\mathbb{E}\|\hat\eta_z\|^2 - 2\mathbb{E}[z \cdot \hat\eta_z] + \mathbb{E}\|z\|^2 = 2\mathcal{L}_{\eta_z} + \text{const}$, i.e. *exactly* the denoiser objective up to a constant. Sampling is Euler–Maruyama on the corresponding SDE. So DDPM = denoiser-trained one-sided linear interpolant in the discrete-time limit.

### 4.6 Stochastic bridges (Peluchetti '22; I²SB; Liu et al. '22)

These methods replace the latent $\gamma(t) z$ with a Brownian bridge process $B_t \sim \mathcal{N}(0, t(1-t) I_d)$ pinned at zero at both endpoints: $x_t^d = I(t, x_0, x_1) + \sqrt{2a}\, B_t$ (Def. 30 of the paper). Training requires Doob's $h$-transform to characterize the drift — generally complicated for non-Gaussian endpoints.

**Map.** Since $B_t \overset{\text{single-time}}{=} \sqrt{t(1-t)}\, z$ for $z \sim \mathcal{N}(0, I_d)$, the law of $x_t^d$ *at any fixed $t$* matches $(\star)$ with $\gamma(t) = \sqrt{2a t(1-t)}$. Hence **both formulations yield the same $b$, $s$, and generative model**, but interpolants do so with an explicit conditional expectation $b = \mathbb{E}[\partial_t I + \dot\gamma z \mid x_t]$ — no $h$-transform required. Bridges are an implementation detail; interpolants subsume them.

### 4.7 Stochastic localization (Eldan; El Alaoui–Montanari)

Localization observes $y_t = t\, x_1 + W_t$ for a Wiener process $W$, then samples $x_1 \sim \rho_1$ via the backward SDE driven by the posterior mean $\mathbb{E}[x_1 \mid y_t]$.

**Map.** At any fixed $t$, $y_t \overset{d}{=} t x_1 + \sqrt{t}\, z$. This is the one-sided spatially-linear interpolant with $\alpha(t) = \sqrt{t}$, $\beta(t) = t$ on $[0, T]$ (rescaled to $[0, 1]$ by a time change). The posterior-mean target is exactly the conditional expectation $\eta_1^{\mathrm{os}}(t, x) = \mathbb{E}[x_1 \mid x_t^{\mathrm{os}} = x]$, and its $L^2$ regression is the $\eta_1$ branch of the interpolant losses (Eq. 4.6).

### 4.8 Schrödinger bridges (entropy-regularized OT)

The Schrödinger bridge problem is

$$\min_{(\rho, u)} \int_0^1\!\!\int |u|^2 \rho\,dx\,dt \quad \text{s.t.} \quad \partial_t \rho + \nabla\!\cdot\!(u \rho) = \varepsilon \Delta \rho,\ \rho(0) = \rho_0,\ \rho(1) = \rho_1$$

for fixed $\varepsilon > 0$. **Theorem 41** of the paper shows it is the max-min of the same quadratic loss over $(\hat I, \hat u)$:

$$\max_{\hat I} \min_{\hat u}\; \int_0^1 \mathbb{E}\!\left[\tfrac{1}{2}|\hat u|^2 - \partial_t \hat I \cdot \hat u + (\dot\gamma - \varepsilon \gamma^{-1}) z \cdot \hat u\right] dt.$$

**Map.** Schrödinger bridges = optimal $I^\star$ paired with the FPE drift $b_F = b + \varepsilon s$, all inside the same interpolant skeleton. The crucial observation is that this extra optimization over $I$ is *not required* for unbiased generation: *any* fixed $I$ with the right boundary conditions already yields exact transport.

### 4.9 Denoising / Tweedie iteration (Sec. 5.2)

For the one-sided linear interpolant, $\mathbb{E}[x_1 \mid x_t^{\mathrm{os}}] = \beta^{-1}(t)\bigl(x_t^{\mathrm{os}} - \alpha(t) \eta_z^{\mathrm{os}}\bigr)$ — Tweedie's identity / SURE. **Theorem 45** shows that iterating this with infinitesimal step is a consistent integrator of the probability flow ODE associated with $b$. So the classical "denoise-and-iterate" approach (Simoncelli, Kadkhodaie, etc.) is exactly ODE sampling in the interpolant picture.

---

## 5. What is genuinely new (not in the methods above)

**(a) The latent $\gamma(t) z$ as a separate knob.** Setting $\gamma > 0$ convolves $\rho(t)$ with $\mathcal{N}(0, \gamma^2 I)$ at every $t$, killing spurious intermediate modes (Fig. 4) and globally smoothing both $b$ and $s$ — *at the deterministic level*, before any sampler stochasticity. The choice $\gamma = \sqrt{a t (1 - t)}$ has $\gamma \dot\gamma \not\to 0$ at the endpoints (Rem. 42), so $b$ inherits endpoint-score information for free.

**(b) Decoupling path from sampler.** The sampler $\varepsilon(t)$ is a free post-training parameter that does not change $\rho(t)$. Existing methods conflate the two — SBDM's noise schedule plays both roles.

**(c) ODE/SDE asymmetry (Lemmas 21–22, Thm. 23).** For the TE, $\mathrm{KL}(\rho(1) \| \hat\rho(1))$ involves a Fisher gap that is *not* bounded by $\|\hat b - b\|^2$. For the FPE with $\varepsilon > 0$ the same expression has an extra $-\varepsilon \int |\nabla \log \rho - \nabla \log \hat\rho|^2$ that absorbs the Fisher term, giving

$$\mathrm{KL}(\rho(1) \| \hat\rho(1)) \;\le\; \frac{1}{4\varepsilon}\int_0^1\!\!\int |\hat b_F - b_F|^2 \rho\,dx\,dt.$$

Optimizing $\mathcal{L}_b, \mathcal{L}_s$ jointly thus controls $\mathrm{KL}(\rho_1 \| \hat\rho_1)$ for the SDE model with optimum at $\varepsilon^\star = \sqrt{(\mathcal{L}_b\text{-gap}) / (\mathcal{L}_s\text{-gap})}$. **No analogous bound exists for $\varepsilon = 0$.**

**(d) Practical recipes.** Antithetic sampling $x_t^\pm = I \pm \gamma z$ kills the $1/\gamma$ variance blow-up in the score loss near $t = 0, 1$; learning $\eta_z$ avoids it entirely. The likelihood / cross-entropy formula for the SDE model (Thm. 26, Cor. 28) is the natural counterpart to the change-of-variables formula for ODE flows.

---

## 6. Discussion prompts

- **Theory–practice gap on ODE vs. SDE.** Theorem 23 strongly favors SDE samplers, but Karras et al. 2022 match SDE quality with ODEs once the score is well learned. Are the bounds loose, or is the gap only visible in the under-trained regime?
- **Two jobs for $\gamma$.** It (i) smooths $\rho(t)$ and (ii) sets endpoint-score information in $b$. Is there a principled criterion for choosing it, or is it always empirical?
- **Coupling $\nu$.** Experiments use $\nu = \rho_0 \otimes \rho_1$. Data-aware couplings (minibatch OT, etc.) give a free path toward Schrödinger bridges without solving the max-min in Thm. 41 — at what cost?
- **What *cannot* be obtained as a special case?** Discrete-state diffusion, Riemannian flows, equivariant models — interpolants live in $\mathbb{R}^d$ with absolutely continuous densities. Where does that hurt?
- **Unbiased SBDM.** The math in Sec. 5.1 says the truncation bias of standard SBDM is artificial. Why hasn't this reformulation propagated to top-of-the-line diffusion training?
- **Distillation.** Rectification and consistency models (Rem. 50) both fall out of this picture. Is there an interpolant-native distillation theorem with quantitative single-step bounds?

**Talking arc.** (1) Setup: $(I, \gamma, \nu, \varepsilon)$ design space and the three quadratic losses. (2) Pick CFM and VP-SBDM/DDPM as the two "derivation" walkthroughs — they are the dominant living competitors and their reductions are the most informative. (3) Pivot to what's new: latent $\gamma z$, decoupling, ODE/SDE asymmetry. (4) Open questions, especially the empirical mismatch with SDE-preference and the future of distillation.

---

## Acronym cheat sheet

| Term | Meaning |
|---|---|
| ODE | ordinary differential equation |
| SDE | stochastic differential equation |
| TE | transport (continuity) equation |
| FPE | Fokker–Planck equation |
| KL | Kullback–Leibler divergence |
| OU | Ornstein–Uhlenbeck (process) |
| OT | optimal transport |
| MLE | maximum likelihood estimation |
| SURE | Stein's unbiased risk estimator |
| i.i.d. | independent and identically distributed |
| DSM | denoising score matching (Vincent '11) |
| DDPM | Denoising Diffusion Probabilistic Models (Ho et al. '20) |
| DDIM | Denoising Diffusion Implicit Models (Song et al. '21) |
| SBDM | score-based diffusion model (Song et al. '21) |
| VP-SBDM | variance-preserving SBDM |
| EDM | Elucidating Diffusion Models (Karras et al. '22) |
| FM | flow matching (Lipman et al. '23) |
| CFM | conditional flow matching |
| CNF | continuous normalizing flow |
| FFJORD | free-form Jacobian of reversible dynamics (CNF training; Grathwohl et al.) |
| I²SB | image-to-image Schrödinger bridge (Liu et al. '23) |
| A.–V.E. | Albergo & Vanden-Eijnden ('23, original interpolant) |
