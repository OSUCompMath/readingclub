# From RNNs to Transformers

This document summarizes the main ideas from the presentation **“From RNNs to transformers”**. It is intended as a concise conceptual overview of how modern sequence models evolved from recurrent architectures to transformers, and why transformers became so important in language modeling. :contentReference[oaicite:1]{index=1}

---

## Contents

1. [Big picture](#big-picture)
2. [Language modeling setup](#language-modeling-setup)
3. [Why RNNs run into difficulty](#why-rnns-run-into-difficulty)
4. [The transformer architecture](#the-transformer-architecture)
5. [Self-attention in words](#self-attention-in-words)
6. [Why positional encoding is needed](#why-positional-encoding-is-needed)
7. [Two approaches to positional information](#two-approaches-to-positional-information)
8. [Main lessons](#main-lessons)
9. [Directions highlighted in the presentation](#directions-highlighted-in-the-presentation)
10. [References mentioned in the slides](#references-mentioned-in-the-slides)

---

## Big picture

The presentation describes the progression

- **RNNs**
- **LSTMs**
- **GRUs**
- **transformers**

as a sequence of increasingly effective neural architectures for learning from ordered data such as text. :contentReference[oaicite:2]{index=2}

The central issue is that sequence models must capture dependence across many tokens. Earlier recurrent models propagate information step by step through the sequence, which can make optimization difficult. In particular, repeated parameter sharing across many steps can contribute to **vanishing gradients**, making long-range dependencies harder to learn. Transformers address this by allowing more direct interaction between different parts of the sequence. :contentReference[oaicite:3]{index=3}

---

## Language modeling setup

A language model works with a vocabulary of possible tokens and takes as input a prompt of the form

$$
y = [y_1,\dots,y_n].
$$

Its task is to model the distribution of the next token given the earlier ones:

$$
p(y_{n+1}\mid y_n,\dots,y_1).
$$

In neural language models, the prompt is first embedded into a vector-valued representation, and then a neural network produces a hidden representation

$$
z = f_\theta(x),
$$

where \(x\) denotes the embedded input sequence. The next-token probabilities are then produced by a softmax layer:

$$
\operatorname{softmax}(Wz)
=
\frac{\exp(W^\top z)}
{\sum_{k\in[N_T]} \exp\big((W^\top z)_k\big)}.
$$

Here:

- \(N_T\) is the vocabulary size,
- \(d\) is the hidden dimension,
- \(W \in \mathbb{R}^{N_T \times d}\),
- \(x \in \mathbb{R}^{d \times n}\). :contentReference[oaicite:4]{index=4}

So at a high level, a language model does three things:

1. embed the prompt,
2. transform that representation through a neural architecture,
3. convert the output into a probability distribution over the next token.

---

## Why RNNs run into difficulty

Recurrent neural networks process a sequence one token at a time, carrying forward a hidden state. This gives them a natural sequential structure, but it also creates a long chain of repeated transformations.

That repeated chain can make gradients shrink as they are propagated backward through time. This is one reason RNNs may struggle to learn dependencies between tokens that are far apart. LSTMs and GRUs were designed to improve this situation using gating mechanisms, but they still retain the basic recurrent viewpoint. :contentReference[oaicite:5]{index=5}

Transformers take a different approach: instead of passing information only through recurrence, they allow each token to interact directly with the others through **attention**.

---

## The transformer architecture

The presentation describes the transformer as a composition of layers

$$
f_{\theta^L}\circ \cdots \circ f_{\theta^1} : \mathbb{R}^{n\times d} \to \mathbb{R}^{n\times d}.
$$

If the input sequence is written as

$$
[x_1,\dots,x_n],
$$

then a transformer layer produces an output sequence

$$
[z_1,\dots,z_n].
$$

For each position \(i\), the layer output is built from attention, normalization, residual structure, and a feedforward nonlinear map. A representative formula from the slides is

$$
z_i
=
N_2\!\left(
N_1(x_i + u_i^{(0)}) + W_2 \operatorname{ReLU}(W_1 u_i)
\right),
$$

where \(N_1\) and \(N_2\) are layer normalizations. :contentReference[oaicite:6]{index=6}

The key term is the self-attention contribution

$$
u_i^{(0)}
=
\sum_{h=1}^H
W^{(h)}
\sum_{j=1}^n
\operatorname{softmax}_j
\!\left(
\frac{1}{\sqrt{k}}
\left\langle Q^{(h)}x_i,\; K^{(h)}x_j \right\rangle
\right)
V^{(h)}x_j.
$$

Here:

- \(H\) is the number of attention heads,
- \(Q^{(h)}\), \(K^{(h)}\), and \(V^{(h)}\) are the **query**, **key**, and **value** maps,
- \(k\) is the head dimension,
- the softmax weights determine how strongly token \(i\) attends to token \(j\). :contentReference[oaicite:7]{index=7}

---

## Self-attention in words

Self-attention can be understood in three steps.

### 1. Each token asks a question

For token \(x_i\), the query vector

$$
q_i = Q^{(h)}x_i
$$

encodes what information that token is looking for.

### 2. Each token advertises what it contains

For token \(x_j\), the key vector

$$
k_j = K^{(h)}x_j
$$

encodes how relevant that token is to different kinds of queries.

### 3. Relevant information is aggregated

The value vector

$$
v_j = V^{(h)}x_j
$$

is the actual content contributed by token \(j\). The similarity score

$$
\left\langle q_i, k_j \right\rangle
$$

measures how much token \(i\) should pay attention to token \(j\). After scaling and applying softmax, these scores become weights that sum to 1, and token \(i\) receives a weighted average of the value vectors from the full sequence.

This is the mechanism that lets transformers model long-range interactions directly.

---

## Why positional encoding is needed

A pure attention mechanism does not inherently know the order of tokens. If the same collection of input vectors were permuted, the output would permute in the same way. In other words, the basic attention mechanism is **permutation equivariant**. :contentReference[oaicite:8]{index=8}

But language depends strongly on order. The meaning of a sentence changes when words are rearranged. So transformers need an additional mechanism to inject positional information.

---

## Two approaches to positional information

### 1. Sinusoidal positional encoding

The original transformer paper introduced additive positional vectors \(p_i \in \mathbb{R}^d\) with coordinates

$$
(p_i)_{2j} = \sin\!\left(\frac{i}{R^{2j/d}}\right),
\qquad
(p_i)_{2j+1} = \cos\!\left(\frac{i}{R^{2j/d}}\right),
$$

where \(R\) is a hyperparameter, often taken to be \(10{,}000\). The token representation is then modified via

$$
x_i \mapsto x_i + p_i.
$$

The basic idea is that different coordinates oscillate at different frequencies, so position is encoded across many scales. :contentReference[oaicite:9]{index=9}

### 2. Rotary / relative positional ideas

The slides also mention a more modern viewpoint in which position is incorporated through rotations in the query-key interaction. Schematically, one modifies the inner product so that it depends on the relative position

$$
m = i-j.
$$

This can be written in a form involving a rotation matrix \(R_m\), so that positional information is built into the attention score itself rather than simply added to the token embedding. :contentReference[oaicite:10]{index=10}

This relative-position perspective is often more natural for long-context sequence modeling.

---

## Main lessons

The main conceptual points of the presentation are the following.

### Transformers replace recurrence with attention

Instead of passing information step by step along the sequence, transformers let each token directly interact with all others.

### This helps with long-range dependencies

Because the path from one token to another is much shorter than in recurrent models, transformers can more easily represent distant interactions.

### Position still matters

Attention alone does not encode order, so positional information must be introduced explicitly.

### The architecture combines linear and nonlinear effects

The attention mechanism mixes information across the sequence, while feedforward layers add nonlinear processing at each position. :contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12}

---

## Directions highlighted in the presentation

The final slide points to several broader research directions:

- **continuous limits of transformers**,  
- **training dynamics**,  
- **RLHF** (reinforcement learning from human feedback). :contentReference[oaicite:13]{index=13}

These directions reflect three different kinds of questions:

1. how transformers behave mathematically in large-scale or limiting regimes,
2. how they are optimized and why training succeeds,
3. how they can be aligned with human preferences after pretraining.

---

## References mentioned in the slides

The presentation cites or mentions the following references:

1. **Dive into Deep Learning**
2. **Ashish Vaswani et al., _Attention Is All You Need_**, NeurIPS 2017
3. **John Thickstun, _Transformers and Maximum Likelihood Speech Models_**, 2021 technical report / lecture notes. :contentReference[oaicite:14]{index=14}

---

## One-paragraph summary

Transformers are neural sequence models designed to overcome some of the main limitations of recurrent architectures. In language modeling, the goal is to predict the next token from the previous ones. Earlier models such as RNNs, LSTMs, and GRUs approached this through recurrence, but transformers instead use self-attention, allowing each token to interact directly with the entire sequence. Because attention alone does not encode token order, positional encoding is added, either through sinusoidal embeddings or more modern relative/rotary mechanisms. This combination of global interaction, nonlinear local processing, and explicit positional structure is what made transformers the dominant architecture in modern language modeling. :contentReference[oaicite:15]{index=15}

---

## Minimal takeaway

If you remember only one thing, it is this:

> **RNNs process sequences step by step; transformers process sequences through attention, letting every token look at every other token.**
