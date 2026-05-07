# $\Phi_\alpha$-Clustering: KMeans and Fuzzy C-Means under $\Phi_\alpha$-Geometry

This repository implements **$\Phi_\alpha$-KMeans** and **$\Phi_\alpha$-FCM**, two extensions of classical clustering algorithms based on a **non-Euclidean geometry induced by a signed power transformation**.

---

## Overview

We consider the transformation:

\[
\Phi_\alpha(x) = \mathrm{sign}(x) |x|^\alpha
\]
which induces the dissimilarity:
\[
d_{\Phi_\alpha}(x, y) = \|\Phi_\alpha(x) - \Phi_\alpha(y)\|^{1/\alpha}
\]

This deformation modifies the geometry of the data space, allowing more flexibility than standard Euclidean clustering.

---

## Implementations

### 1. $\Phi_\alpha$-KMeans
- File: `PhiKmeans.py`
- Hard clustering method
- Equivalent to KMeans when \( \alpha = 1 \)
- Performs clustering in the transformed space

---

### 2. $\Phi_\alpha$-FCM ($\Phi_\alpha$ Fuzzy C-Means)
- File: `PhiFCM.py`
- Soft clustering with memberships \( u_{ik} \)
- Controlled by:
  - \( \alpha \): geometry parameter
  - \( m \): fuzzifier
- Reduces to KMeans when \( m \to 1 \) and $\alpha=1$

---

## Illustration

### $\Phi_\alpha$-KMeans (Hard clustering, $\alpha=0.6$)

![Phi-KMeans](part2_selected_hard_alpha_0.600.png)

---

### Φα-FCM (Fuzzy clustering, $m = 1.5$, $\alpha=0.6$)

![Phi-FCM](part2_selected_fuzzy_m_1.50_alpha_0.600.png)

---

### Minimal Requirements

```python
Python >= 3.8
PyTorch
NumPy
Scikit-learn
SciPy
Matplotlib (for visualization)

