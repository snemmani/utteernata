"""
Day 2: Industrial-Grade CausalSelfAttention (nn.Module)
Author: Shankar Nag Nemmani (Utteernata Project)

In this lesson, we assemble all the Day 1 fundamentals into a production-grade
PyTorch module — the exact implementation used inside GPT-2, nanoGPT, and Llama.

Key Engineering Patterns:
1. Fused QKV Linear Projection (nn.Linear(C, 3 * C))
2. Persistent Buffers via self.register_buffer()
3. Attention & Residual Dropout
4. Output Projection (nn.Linear(C, C))
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        """
        Args:
            n_embd: Total embedding dimension (C). e.g., 64 (toy), 768 (GPT-2), 4096 (Llama 3).
            n_head: Number of attention heads (nh). Must divide n_embd evenly.
            block_size: Maximum context length (T_max) the causal mask can handle.
            dropout: Dropout probability for regularization.
        """
        super().__init__()
        assert n_embd % n_head == 0, f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # 1. FUSED QKV PROJECTION:
        # Instead of 3 separate linear layers (q_proj, k_proj, v_proj),
        # we compute all three in ONE single matrix multiplication: C -> 3 * C.
        # This doubles GPU memory throughput and reduces kernel launch overhead!
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)

        # 2. OUTPUT PROJECTION:
        # After concatenating the heads back together, this linear layer
        # mixes the heads and projects them back into the residual stream.
        self.c_proj = nn.Linear(n_embd, n_embd)

        # 3. REGULARIZATION:
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # 4. CAUSAL MASK BUFFER:
        # Why register_buffer?
        # - It is NOT a trainable parameter (no gradients).
        # - BUT it needs to automatically move to the GPU when you call model.cuda().
        # - It gets saved in the model's state_dict() automatically.
        # Shape: (1, 1, block_size, block_size) -> broadcastable across Batch and Heads!
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("bias", mask.view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, C)
        Returns:
            Output tensor of shape (B, T, C) enriched with context.
        """
        B, T, C = x.shape
        assert C == self.n_embd, f"Input channel dim {C} does not match model {self.n_embd}"
        assert T <= self.bias.size(-1), f"Input sequence length {T} exceeds max block size {self.bias.size(-1)}"

        # -------------------------------------------------------------
        # STEP 1: Fused Projection & Head Splitting
        # -------------------------------------------------------------
        # Project: (B, T, C) -> (B, T, 3 * C)
        qkv = self.c_attn(x)
        
        # Split into 3 equal chunks of size C along dimension 2:
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape & Transpose from (B, T, C) -> (B, nh, T, hs)
        # Using the exact Drill 2 & Drill 3 operations we mastered!
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # -------------------------------------------------------------
        # STEP 2: Scaled Dot-Product Attention & Causal Masking
        # -------------------------------------------------------------
        # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        # Scale by 1 / sqrt(head_dim) to keep variance at 1.0!
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale

        # Apply Causal Mask: slice mask up to current sequence length T
        # Where mask is 0, replace with -infinity so softmax outputs 0.0
        scores = scores.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        # Softmax over columns (dim=-1) so each ROW sums to 1.0!
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # -------------------------------------------------------------
        # STEP 3: Context Blending (weights @ V)
        # -------------------------------------------------------------
        # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = weights @ v

        # -------------------------------------------------------------
        # STEP 4: Head Reassembly & Output Projection
        # -------------------------------------------------------------
        # Reassemble heads: (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection back to residual stream
        y = self.resid_dropout(self.c_proj(y))
        return y


def run_verification():
    print("=" * 65)
    print("DAY 2 VERIFICATION: Testing CausalSelfAttention Module")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Hyperparameters for a toy attention block
    B = 2           # 2 sentences in batch
    T = 6           # 6 tokens per sentence
    C = 64          # Embedding dimension (channels)
    nh = 4          # 4 attention heads (each head_dim = 16)
    block_size = 32 # Maximum context length

    # 1. Instantiate module
    attn = CausalSelfAttention(n_embd=C, n_head=nh, block_size=block_size, dropout=0.1).to(device)

    # 2. Count parameters
    total_params = sum(p.numel() for p in attn.parameters() if p.requires_grad)
    print(f"CausalSelfAttention parameter count: {total_params:,}")
    # c_attn: (C * 3C) weights + 3C biases = (64 * 192) + 192 = 12,480
    # c_proj: (C * C) weights + C biases   = (64 * 64) + 64   = 4,160
    # Total = 16,640 parameters

    # 3. Create dummy input tensor
    x = torch.randn(B, T, C, device=device, requires_grad=True)
    print(f"\nInput tensor shape  : {x.shape} (B={B}, T={T}, C={C})")

    # 4. Execute Forward Pass
    y = attn(x)
    print(f"Output tensor shape : {y.shape} (Matches input (B, T, C)!)")

    # 5. Execute Backward Pass (Verify autograd gradients flow cleanly!)
    loss = y.sum()
    loss.backward()

    print(f"\nAutograd Check:")
    print(f"- Input x.grad is populated: {x.grad is not None} (Shape: {x.grad.shape})")
    print(f"- c_attn.weight.grad exists: {attn.c_attn.weight.grad is not None} (Shape: {attn.c_attn.weight.grad.shape})")
    print(f"- c_proj.weight.grad exists: {attn.c_proj.weight.grad is not None} (Shape: {attn.c_proj.weight.grad.shape})")
    print("\nSUCCESS: Production-grade CausalSelfAttention module is fully functional!")
    print("=" * 65)


if __name__ == "__main__":
    run_verification()
