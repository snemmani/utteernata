"""
Day 1: PyTorch Tensors, Autograd & First Training Loop
Author: Shankar Nag Nemmani (Utteernata Project)

Run this script to:
1. Verify PyTorch sees your NVIDIA RTX 5070 (CUDA support).
2. Understand PyTorch tensor mechanics vs NumPy/TensorFlow.
3. Observe Autograd computing derivatives automatically.
4. Execute your first raw PyTorch training loop.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim

def step_1_check_environment():
    print("=" * 60)
    print("STEP 1: Hardware & Environment Verification")
    print("=" * 60)
    print(f"Python version : {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available : {cuda_available}")

    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Active GPU     : {gpu_name} ({gpu_mem:.2f} GB VRAM)")
        device = torch.device("cuda")
    else:
        print("WARNING: CUDA not detected by PyTorch. Falling back to CPU.")
        device = torch.device("cpu")
    print(f"Selected device: {device}\n")
    return device


def step_2_tensor_basics(device):
    print("=" * 60)
    print("STEP 2: Tensors on Device")
    print("=" * 60)
    # Create tensors on device
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)

    # Matrix multiplication (@)
    c = a @ b
    print("Matrix A:\n", a.cpu())
    print("Matrix B:\n", b.cpu())
    print("Matrix C = A @ B:\n", c.cpu())
    print(f"Tensor lives on: {c.device}\n")


def step_3_autograd_in_action():
    print("=" * 60)
    print("STEP 3: Autograd & Derivative Tracking")
    print("=" * 60)
    # We want to find gradient of Loss w.r.t weights w
    # Let y = w * x + b
    # Loss = (y - target)^2
    w = torch.tensor([3.0], requires_grad=True)
    b = torch.tensor([1.0], requires_grad=True)
    x = torch.tensor([2.0])
    target = torch.tensor([15.0])

    # Forward pass: y = 3 * 2 + 1 = 7
    y = w * x + b
    # Loss: (7 - 15)^2 = (-8)^2 = 64
    loss = (y - target) ** 2

    print(f"Weights w: {w.item()}, Bias b: {b.item()}, Input x: {x.item()}")
    print(f"Predicted y: {y.item()}, Target: {target.item()}, Loss: {loss.item()}")

    # Backward pass: compute gradients
    loss.backward()

    # Analytical gradients:
    # d(Loss)/dw = 2 * (w*x + b - target) * x = 2 * (7 - 15) * 2 = -32.0
    # d(Loss)/db = 2 * (w*x + b - target) * 1 = 2 * (7 - 15) * 1 = -16.0
    print(f"Computed d(Loss)/dw (w.grad): {w.grad.item()} (Expected: -32.0)")
    print(f"Computed d(Loss)/db (b.grad): {b.grad.item()} (Expected: -16.0)\n")


class TinyMLP(nn.Module):
    """A 2-layer Neural Network with GELU activation."""
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x):
        return self.net(x)


def step_4_training_loop(device):
    print("=" * 60)
    print("STEP 4: Complete PyTorch Training Loop")
    print("=" * 60)
    # Synthetic problem: learn y = 2*x1 - 3*x2 + 0.5
    torch.manual_seed(42)
    X = torch.randn(200, 2, device=device)
    y_true = 2.0 * X[:, 0:1] - 3.0 * X[:, 1:2] + 0.5

    # Model, Loss, Optimizer
    model = TinyMLP(in_features=2, hidden_features=16, out_features=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.05)

    print("Training TinyMLP for 50 epochs...")
    for epoch in range(1, 51):
        # 1. Forward
        y_pred = model(X)
        loss = criterion(y_pred, y_true)

        # 2. Backward
        optimizer.zero_grad()  # Essential: clear accumulated grads
        loss.backward()

        # 3. Update weights
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/50 | MSE Loss: {loss.item():.6f}")

    print("\nTraining completed successfully! Model parameters converged.")
    print("=" * 60)


if __name__ == "__main__":
    device = step_1_check_environment()
    step_2_tensor_basics(device)
    step_3_autograd_in_action()
    step_4_training_loop(device)
