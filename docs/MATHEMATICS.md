# The Mathematics of FerricML

This document explains the mathematical foundations of FerricML, focusing on the **Ahead-of-Time Automatic Differentiation (AOT-AD)** and the compiler-driven approach to gradients.

## 1. Automatic Differentiation (AD)

Automatic Differentiation is a set of techniques to numerically evaluate the derivative of a function specified by a computer program. FerricML primarily uses **Reverse-Mode AD**, also known as backpropagation.

### 1.1 The Chain Rule
The core of AD is the chain rule. If we have a composite function $y = f(g(x))$, its derivative with respect to $x$ is:
$$ \frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} \quad \text{where } u = g(x) $$

In a computational graph, each node $i$ computes an output $v_i$ from its parents $Pa(i)$. To compute the gradient of the loss $L$ with respect to $v_i$ (denoted as $\bar{v}_i$):
$$ \bar{v}_i = \sum_{j \in Children(i)} \bar{v}_j \frac{\partial v_j}{\partial v_i} $$

## 2. Ahead-of-Time (AOT) vs. Tape-Based AD

### 2.1 Tape-Based AD (e.g., PyTorch)
In tape-based systems, every operation is recorded on a "tape" (a dynamic data structure) during the forward pass. The backward pass then traverses this tape in reverse.
- **Pros:** Extremely flexible, supports dynamic control flow.
- **Cons:** High runtime overhead, limited opportunities for compiler optimization.

### 2.2 AOT-AD (FerricML)
FerricML's RMLC compiler symbolicly differentiates the code before execution.
1. **Symbolic Expression:** The forward pass is represented as a symbolic expression $E$.
2. **Derivative Generation:** RMLC computes the symbolic derivative $\frac{\partial E}{\partial x}$.
3. **Joint Optimization:** The compiler optimizes the combined graph. For example, if $v_i$ is used in the backward pass but not needed after the forward pass, the compiler can optimize the memory allocation to reflect this.

## 3. Differentiation of Common Operations

FerricML provides efficient symbolic gradients for all core operations.

### 3.1 Matrix Multiplication
Given $Y = W X$, the gradients are:
- $\bar{W} = \bar{Y} X^T$
- $\bar{X} = W^T \bar{Y}$

### 3.2 Convolution
The gradient of a convolution $Y = K * X$ involves transposed convolutions:
- $\bar{K} = \text{Conv}(\text{padding}(X), \bar{Y})$
- $\bar{X} = \text{Deconv}(K, \bar{Y})$

## 4. Optimization Theory

RMLC uses several mathematical techniques to optimize the execution graph:

### 4.1 Operator Fusion
Using the property of associativity, RMLC can fuse multiple element-wise operations. For example, $y = \text{ReLU}(ax + b)$ is computed in a single GPU kernel to minimize memory bandwidth usage.

### 4.2 Symbolic Simplification
RMLC uses algebraic identities to simplify graphs.
- $x + 0 \rightarrow x$
- $x \cdot 1 \rightarrow x$
- $\log(\exp(x)) \rightarrow x$

## 5. Numerical Stability

To prevent issues like vanishing or exploding gradients, FerricML implements:
- **Gradient Clipping:** Restricting the norm of gradients.
- **Epsilon Injection:** Adding a small value $\epsilon$ to denominators (e.g., in BatchNorm or Adam) to prevent division by zero.
- **Numerically Stable Loss Functions:** Implementing `LogSoftmax` and `BCEWithLogits` to avoid precision loss in exponential calculations.
