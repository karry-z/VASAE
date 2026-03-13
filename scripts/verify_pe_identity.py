"""Verify that PE = WE^T (WE WE^T)^† WE ≈ I_d for GPT-2's embedding matrix.

Mathematical identity: WE^T (WE WE^T)^† WE = WE^+ WE  (pseudoinverse times WE).
This is the orthogonal projector onto row(WE) in R^d.
If rank(WE) = d, then PE = I_d exactly.

We avoid the expensive pinv of the V×V matrix by using pinv(WE) directly (d×V),
which internally only needs the SVD of the smaller dimension.
"""

import torch
from transformers import GPT2LMHeadModel


def main():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    WE = model.transformer.wte.weight.detach().float()  # (V, d) = (50257, 768)
    print(f"WE shape: {WE.shape}")

    # Rank via SVD
    S = torch.linalg.svdvals(WE)
    tol = S[0] * max(WE.shape) * torch.finfo(WE.dtype).eps
    rank = (S > tol).sum().item()
    print(f"Rank of WE: {rank}  (d={WE.shape[1]})")
    print(f"Smallest singular value: {S[-1]:.6e}")
    print(f"Condition number: {S[0]/S[-1]:.2f}")

    # PE = WE^+ @ WE  — equivalent to WE^T (WE WE^T)^† WE, but much faster
    WE_pinv = torch.linalg.pinv(WE)  # (d, V)
    PE = WE_pinv @ WE  # (d, d)

    I = torch.eye(WE.shape[1])
    diff = PE - I
    frob = torch.linalg.norm(diff, "fro").item()
    maxabs = diff.abs().max().item()

    print(f"||PE - I||_F  = {frob:.6e}")
    print(f"max|PE - I|   = {maxabs:.6e}")
    print(f"Conclusion: PE {'≈' if frob < 1e-3 else '≠'} I")


if __name__ == "__main__":
    main()
