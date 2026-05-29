# 🔢 Handwritten Digit Classification — MNIST

Building a neural network **from scratch using only NumPy**, then extending it to a **Convolutional Neural Network (CNN) with Keras** to classify handwritten digits from the MNIST dataset — achieving **98.69% accuracy**.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Part 1 — Neural Network from Scratch](#part-1--neural-network-from-scratch)
- [Part 2 — CNN with Keras (Extended)](#part-2--cnn-with-keras-extended)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)

---

## 📌 Overview

This project is a two-part deep dive into neural networks for image classification:

- **Part 1 (`Wids_project_3.ipynb`)** — Implements a complete neural network engine from scratch using only NumPy. Every component — linear layers, activation functions, loss function, backpropagation, and SGD — is hand-coded without any ML library.
- **Part 2 (`Wids_project_Extended.ipynb`)** — Extends the solution using Keras to build a CNN with 5-fold cross-validation, achieving **98.69% mean accuracy** on the MNIST test set.

---

## 📁 Project Structure

```
Handwritten-Digit-Classification/
│
├── Wids_project_3.ipynb          # Neural network built from scratch (NumPy only)
├── Wids_project_Extended.ipynb   # CNN with Keras + K-Fold cross-validation
└── README.md
```

---

## Part 1 — Neural Network from Scratch

### 🧠 Architecture

A fully modular neural network engine implemented from scratch in NumPy, following the **linear + activation module** pattern:

```
Input (784)  →  Linear(784→128)  →  ReLU
             →  Linear(128→64)   →  ReLU
             →  Linear(64→10)    →  SoftMax
                                  →  NLL Loss
```

### ⚙️ Modules Implemented

| Module | Description |
|---|---|
| `Linear` | Forward: `Z = Wᵀ·A + W₀` — Backward: computes `dL/dW`, `dL/dW₀`, `dL/dA` |
| `Tanh` | Activation: `tanh(Z)` — Backward: `1 - tanh²(Z)` |
| `ReLU` | Activation: `max(0, Z)` — Backward: pass gradient where `Z > 0` |
| `SoftMax` | Output activation with numerical stability (`exp(Z - max(Z))`) |
| `NLL` | Negative Log-Likelihood loss with one-hot encoded labels |
| `Sequential` | Chains modules, runs forward/backward pass, applies SGD steps |

### 🔁 Training Loop (SGD)

```python
# Build the network
net = Sequential([
    Linear(784, 128), ReLU(),
    Linear(128, 64),  ReLU(),
    Linear(64, 10),   SoftMax()
], NLL())

# Train with Stochastic Gradient Descent
for each iteration:
    1. Sample a random training point (Xt, Yt)
    2. One-hot encode the label
    3. Forward pass → prediction
    4. Compute NLL loss
    5. Backward pass → compute gradients
    6. SGD step → update all weights
```

### 📐 Mathematical Foundation

**Linear layer forward:**
$$z_j = \sum_{i=1}^{m} x_i W_{i,j} + W_{0j}$$

**Backpropagation — weight gradient:**
$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial Z} \cdot A^T$$

**NLL Loss:**
$$\mathcal{L} = -\frac{1}{N} \sum_{i} \sum_{k} y_{ik} \log(\hat{y}_{ik} + \epsilon)$$

---

## Part 2 — CNN with Keras (Extended)

### 🏗️ CNN Architecture

```
Input (28×28×1)
    │
    ▼
Conv2D(32 filters, 3×3, ReLU, he_uniform)
    │
    ▼
MaxPooling2D(2×2)
    │
    ▼
Flatten
    │
    ▼
Dense(100, ReLU, he_uniform)
    │
    ▼
Dense(10, Softmax)
    │
    ▼
Output: digit class (0–9)
```

### 🔄 Training Setup

| Setting | Value |
|---|---|
| Optimizer | SGD (lr=0.01, momentum=0.9) |
| Loss | Categorical Cross-Entropy |
| Epochs | 10 per fold |
| Batch Size | 32 |
| Validation | 5-Fold Cross-Validation |
| Pixel Normalization | Divide by 255 → range [0, 1] |

---

## 📊 Results

### Part 1 — Neural Network from Scratch
| Component | Detail |
|---|---|
| Dataset | MNIST (CSV format) |
| Architecture | 784 → 128 → 64 → 10 |
| Training | Stochastic Gradient Descent |
| Framework | NumPy only |

### Part 2 — CNN (Extended)

| Fold | Accuracy |
|---|---|
| Fold 1 | 98.475% |
| Fold 2 | 98.767% |
| Fold 3 | 98.525% |
| Fold 4 | 98.883% |
| Fold 5 | 98.800% |
| **Mean** | **98.69%** |
| **Std Dev** | **0.161%** |

> 🏆 The CNN achieves **98.69% mean accuracy** with very low variance (±0.16%) across all 5 folds — demonstrating strong generalization.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)

| Part | Libraries Used |
|---|---|
| Part 1 (from scratch) | `numpy`, `pandas`, `matplotlib`, `sklearn` |
| Part 2 (CNN) | `tensorflow`, `keras`, `numpy`, `sklearn`, `matplotlib` |

---

## ▶️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/pawanahirwa/Handwritten-Digit-Classification-Using-Neural-Network-on-MNIST-dataset.git
cd Handwritten-Digit-Classification-Using-Neural-Network-on-MNIST-dataset
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

### 3. Download MNIST dataset
Get the CSV version from [Kaggle MNIST](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) and update the file path in the notebook:
```python
Mnist = pd.read_csv('path/to/your/mnist.csv')
```

### 4. Run the notebooks

**Part 1 — From Scratch:**
```bash
jupyter notebook Wids_project_3.ipynb
```

**Part 2 — CNN (Extended):**
```bash
jupyter notebook Wids_project_Extended.ipynb
```

> **Note:** Part 2 uses `tensorflow.keras.datasets.mnist` which auto-downloads the dataset — no manual download needed for the extended notebook.

---

## 👤 Author

**Pawan Singh Ahirwar**
- GitHub: [@pawanahirwa](https://github.com/pawanahirwa)
- Affiliation: IRCC, IIT Bombay
