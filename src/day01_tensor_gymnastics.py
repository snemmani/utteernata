"""
Day 1 (Part 2): Tensor Gymnastics — The 5 Operations That Power Transformers
Author: Shankar Nag Nemmani (Utteernata Project)

Goal: Master the exact tensor manipulations used in Multi-Head Attention (MHA)
so that dimension errors never slow you down.

Run:
    python src/day01_tensor_gymnastics.py
"""

import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running Tensor Gymnastics on: {device}")
if device.type == "cuda":
    print(f"Device Name: {torch.cuda.get_device_name(0)}\n")


def drill_1_the_anatomy_of_llm_tensors():
    """
    DRILL 1: The Standard Transformer Tensor Notation
    B = Batch size (how many independent sequences we process at once)
    T = Sequence length / Time (number of tokens in the prompt/context)
    C = Channel / Embedding dimension (size of each token's vector, e.g. 768 for GPT-2, 4096 for Llama 3)
    """
    print("=" * 65)
    print("DRILL 1: The Standard (B, T, C) Representation")
    print("=" * 65)
    B, T, C = 2, 4, 16  # 2 sentences, 4 tokens each, embedding size 16
    x = torch.randn(B, T, C, device=device)

    print(f"Input tensor x shape: {x.shape} -> (Batch={B}, SeqLen={T}, Channels={C})")
    print(f"Token vector for sentence 0, token 0:\n{x[0, 0, :8]} ... (showing first 8 values)\n")
    return x, B, T, C


def drill_2_splitting_heads_view():
    """
    DRILL 2: Splitting Embedding dimension into Multiple Attention Heads.
    Instead of 1 big attention mechanism of size C (16), we use n_head (4)
    parallel heads of size head_dim = C / n_head (4).
    """
    print("=" * 65)
    print("DRILL 2: Multi-Head Splitting with .view()")
    print("=" * 65)
    B, T, C = 2, 4, 16
    n_head = 4
    head_dim = C // n_head  # 16 // 4 = 4

    # Let Q be the Query projection of shape (B, T, C)
    Q = torch.randn(B, T, C, device=device)

    # We reshape (B, T, C) -> (B, T, n_head, head_dim)
    Q_split = Q.view(B, T, n_head, head_dim)
    print(f"Original Q shape : {Q.shape}")
    print(f"Q after .view()  : {Q_split.shape} -> (Batch, SeqLen, NumHeads, HeadDim)\n")
    return Q_split, n_head, head_dim


def drill_3_the_transpose_trick():
    """
    DRILL 3: Swapping Dimensions for Parallel Attention
    In PyTorch, batched matrix multiplication (`@`) operates on the LAST TWO dimensions.
    To calculate attention between all pairs of tokens (T x T) independently for each head,
    we MUST swap NumHeads and SeqLen so NumHeads acts as an outer batch dimension.

    Target shape: (B, NumHeads, SeqLen, HeadDim)
    """
    print("=" * 65)
    print("DRILL 3: Swapping Dimensions with .transpose() & .contiguous()")
    print("=" * 65)
    B, T, n_head, head_dim = 2, 4, 4, 4
    Q_split = torch.randn(B, T, n_head, head_dim, device=device)

    # Swap dimension 1 (SeqLen) and dimension 2 (NumHeads)
    # (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
    Q_transposed = Q_split.transpose(1, 2)
    print(f"Before transpose : {Q_split.shape} -> [B, T, nh, hs]")
    print(f"After transpose  : {Q_transposed.shape} -> [B, nh, T, hs]")

    # Check memory layout:
    # Transposing changes how strides are indexed without moving memory, making it non-contiguous.
    print(f"Is tensor contiguous in memory? {Q_transposed.is_contiguous()}")
    # .contiguous() rearranges the memory layout in RAM/VRAM so subsequent .view() calls never fail:
    Q_contiguous = Q_transposed.contiguous()
    print(f"After .contiguous(): {Q_contiguous.is_contiguous()}\n")
    return Q_transposed


def drill_4_attention_matrix_multiplication():
    """
    DRILL 4: Calculating Token-to-Token Attention Scores (Q @ K^T)
    Q shape: (B, nh, T, hs)
    K shape: (B, nh, T, hs)

    We need to multiply each token's Query with all tokens' Keys.
    To do this, we transpose the last two dimensions of K:
    K.transpose(-2, -1) shape: (B, nh, hs, T)

    Now: (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
    The result is a (T x T) square matrix representing how much token i attends to token j!
    """
    print("=" * 65)
    print("DRILL 4: Attention Matrix Multiplication (Q @ K.T)")
    print("=" * 65)
    B, nh, T, hs = 2, 4, 4, 4
    Q = torch.randn(B, nh, T, hs, device=device)
    K = torch.randn(B, nh, T, hs, device=device)

    # Transpose only the last two dimensions (-2 and -1)
    K_t = K.transpose(-2, -1)
    print(f"Q shape     : {Q.shape}")
    print(f"K.T shape   : {K_t.shape}")

    # Scaled Dot-Product: divide by sqrt(d_k) to prevent vanishing gradients in softmax
    scores = (Q @ K_t) / (hs ** 0.5)
    print(f"Scores shape: {scores.shape} -> (Batch, NumHeads, Token_i, Token_j)")
    print(f"Attention map for Batch 0, Head 0:\n{scores[0, 0].cpu().detach().round(decimals=2)}\n")
    return scores, T


def drill_5_causal_masking_and_softmax():
    """
    DRILL 5: Causal Masking (Preventing Looking into the Future)
    In autoregressive generation (GPT, Llama), token t can only attend to tokens <= t.
    We create a lower-triangular matrix:
    [ [1, 0, 0, 0],
      [1, 1, 0, 0],
      [1, 1, 1, 0],
      [1, 1, 1, 1] ]
    Where mask == 0, we fill with -infinity. When softmax is applied, exp(-inf) = 0!
    """
    print("=" * 65)
    print("DRILL 5: Causal Masking & Softmax")
    print("=" * 65)
    B, nh, T = 2, 4, 4
    scores = torch.randn(B, nh, T, T, device=device)

    # 1. Create lower triangular mask
    mask = torch.tril(torch.ones(T, T, device=device))
    print(f"Causal Mask (1=allowed, 0=forbidden):\n{mask.cpu()}\n")

    # 2. Fill 0s with -inf
    masked_scores = scores.masked_fill(mask == 0, float("-inf"))
    print(f"Masked scores (Batch 0, Head 0):\n{masked_scores[0, 0].cpu().round(decimals=2)}\n")

    # 3. Softmax over the last dimension (dim=-1)
    attn_weights = F.softmax(masked_scores, dim=-1)
    print(f"Attention Weights (Probabilities sum to 1.0 across rows):\n{attn_weights[0, 0].cpu().round(decimals=3)}")
    print(f"Row sums: {attn_weights[0, 0].sum(dim=-1).cpu()}\n")

    # 4. Multiply with Values V to get final context representation
    V = torch.randn(B, nh, T, 4, device=device)
    out = attn_weights @ V  # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    print(f"Output of attention before reassembly: {out.shape}")

    # 5. Reassemble heads back to (B, T, C)
    # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
    out = out.transpose(1, 2).contiguous().view(B, T, nh * 4)
    print(f"Final MHA Output tensor shape: {out.shape} -> exactly matches original (B, T, C)!\n")
    print("=" * 65)
    print("ALL 5 DRILLS COMPLETED! You now know the exact mechanics of Multi-Head Attention.")
    print("=" * 65)


if __name__ == "__main__":
    drill_1_the_anatomy_of_llm_tensors()
    drill_2_splitting_heads_view()
    drill_3_the_transpose_trick()
    drill_4_attention_matrix_multiplication()
    drill_5_causal_masking_and_softmax()
