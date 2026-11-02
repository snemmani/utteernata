# PyTorch Zero-to-Fluent Guide (For TensorFlow Engineers)

Welcome to PyTorch. If you know Python and basic TensorFlow/NumPy, this guide is designed to make you completely self-sufficient in raw PyTorch in **72 hours**.

---

## 1. The Core Philosophy Shift

* In **TensorFlow (especially 1.x and early 2.x)**, you declared a computational graph first, then executed it inside a session or tape.
* In **PyTorch**, everything is **imperative ("eager execution" by default)**. A PyTorch tensor is just a NumPy array that lives on a GPU and remembers what math was done to it so it can calculate derivatives backwards.

```python
# NumPy
import numpy as np
a = np.array([1.0, 2.0])
b = np.array([3.0, 4.0])
c = a + b

# PyTorch (looks almost identical!)
import torch
a = torch.tensor([1.0, 2.0])
b = torch.tensor([3.0, 4.0])
c = a + b
```

---

## 2. The 3 Primitives You Must Master

### Primitive 1: Tensor Device Management (Using Your RTX 5070)
PyTorch requires you to be explicit about where data lives (CPU RAM vs. GPU VRAM). Operations between a CPU tensor and a GPU tensor will throw an error.

```python
import torch

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create tensor directly on GPU
x = torch.zeros((4, 128), device=device)

# Or move existing tensor to GPU
y = torch.randn(4, 128).to(device)

# Fast matrix multiplication on your RTX 5070
z = x @ y.T  # Shape: (4, 4)
```

### Primitive 2: Autograd (`requires_grad=True` & `.backward()`)
To train any model, you need gradients of the loss with respect to the parameters: $\frac{\partial \mathcal{L}}{\partial W}$.

```python
import torch

# 1. Weights: require gradients
w = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)

# 2. Input data: does NOT require gradients
x = torch.tensor([3.0])

# 3. Forward calculation: y = w * x + b = 2*3 + 1 = 7
y = w * x + b

# 4. Target and Loss: loss = (y - target)^2 = (7 - 10)^2 = (-3)^2 = 9
target = torch.tensor([10.0])
loss = (y - target) ** 2

# 5. Backward pass: computes d(loss)/dw and d(loss)/db
loss.backward()

# 6. Gradients are now populated directly on w.grad and b.grad!
# d(loss)/dw = 2 * (w*x + b - target) * x = 2 * (-3) * 3 = -18
print(w.grad)  # tensor([-18.])
# d(loss)/db = 2 * (w*x + b - target) * 1 = 2 * (-3) * 1 = -6
print(b.grad)  # tensor([-6.])
```

> [!CAUTION]
> **Gradient Accumulation Trap**: In PyTorch, calling `loss.backward()` **adds** gradients to `param.grad` instead of overwriting them! This allows multi-step gradient accumulation, but you MUST call `optimizer.zero_grad()` before computing gradients for the next batch.

### Primitive 3: The `nn.Module` Architecture Skeleton
Every layer, block, and model in PyTorch inherits from `torch.nn.Module`.

```python
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # 1. Define layers (trainable parameters live here)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 2. Define data flow
        h = self.fc1(x)
        h = self.activation(h)
        out = self.fc2(h)
        return out

# Instantiate and move to GPU
model = SimpleMLP(input_dim=10, hidden_dim=32, output_dim=1).cuda()
```

---

## 3. The Canonical PyTorch Training Loop

In Keras you called `model.fit(x, y, epochs=10)`. In PyTorch, you write the explicit 5-step loop (which gives you 100% control over training dynamics, clipping, and custom losses):

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Setup
model = SimpleMLP(10, 32, 1).cuda()
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# Training loop
for epoch in range(10):
    # Dummy batch
    inputs = torch.randn(64, 10).cuda()
    targets = torch.randn(64, 1).cuda()

    # Step 1: Forward pass
    outputs = model(inputs)

    # Step 2: Compute loss
    loss = criterion(outputs, targets)

    # Step 3: Clear old gradients! (Crucial)
    optimizer.zero_grad()

    # Step 4: Backward pass (compute gradients)
    loss.backward()

    # Optional: Gradient clipping to prevent exploding gradients
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Step 5: Update weights using the optimizer
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

---

## 4. The 5 Essential Tensor Operations for LLM Builders

All LLM architectures (Attention, KV Cache, RoPE) come down to these 5 tensor manipulation functions:

### 1. `.view()` vs `.reshape()`
Reshaping tensor dimensions without copying memory when contiguous:
```python
x = torch.randn(2, 4, 16)  # [batch, seq_len, hidden_dim]
# Split hidden_dim into 4 heads of dimension 4
y = x.view(2, 4, 4, 4)     # [batch, seq_len, num_heads, head_dim]
```

### 2. `.transpose()` and `.permute()`
Swapping dimensions. Essential for multi-head attention to bring `num_heads` next to `batch`:
```python
# x shape: [batch, seq_len, num_heads, head_dim]
# We want: [batch, num_heads, seq_len, head_dim] for parallel attention
x_transposed = x.permute(0, 2, 1, 3)
```

### 3. `.unsqueeze()` and `.squeeze()`
Adding or removing a dimension of size 1:
```python
mask = torch.tensor([1, 1, 0, 1])     # Shape: (4)
mask_2d = mask.unsqueeze(0)           # Shape: (1, 4)
mask_3d = mask.unsqueeze(0).unsqueeze(1) # Shape: (1, 1, 4) (broadcastable over batch and heads)
```

### 4. Batched Matrix Multiplication (`@` or `torch.matmul`)
Multiplies the last two dimensions across all leading batch/head dimensions:
```python
Q = torch.randn(2, 8, 16, 64)  # [batch, heads, seq_len_q, head_dim]
K = torch.randn(2, 8, 16, 64)  # [batch, heads, seq_len_k, head_dim]
# Q @ K.transpose(-2, -1) -> [2, 8, 16, 16] (Attention scores matrix)
scores = Q @ K.transpose(-2, -1)
```

### 5. `torch.no_grad()`
Disables gradient calculation during evaluation or inference, saving huge amounts of VRAM:
```python
model.eval()  # Put layers like Dropout/BatchNorm into eval mode
with torch.no_grad():
    predictions = model(test_data)
```

---

## 5. Next Exercise: Scaled Dot-Product Attention

Once you run `src/day01_tensors_autograd.py`, you will be ready to build Scaled Dot-Product Attention:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
where $M$ is the causal mask preventing attention to future tokens.
