"""

Transformer Classifier with Manual Autograd

Author: Ifigeneia Apostolopoulou

==========================================================
Every core step in the forward pass is implemented as a custom
``torch.autograd.Function`` with hand-derived gradients.

Custom Function inventory
-------------------------
Attention stack
  LinearProjectionFunction  X @ W for Q/K/V projections
  ScaledDotProductScores    Q·Kᵀ / √d
  SoftmaxFunction           row-wise stable softmax (Jacobian form)
  AttentionOutputFunction   softmax_weights · V

Head / loss
  MeanPoolFunction          mean over sequence dimension
  LinearHeadFunction        pooled @ W_O + b_O
  SigmoidFunction           σ(z) = 1 / (1 + e^{-z})
  BCELossFunction           −mean[ y·log(p) + (1−y)·log(1−p) ]

This file exposes a single unittest demo that:
1. trains the model on a toy sequence classification task,
2. prints the initial and final train/eval losses,
3. plots both train and eval loss,
4. asserts that the train loss decreases.

"""

import math
import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD


# ---------------------------------------------------------------------------
# 1. Data generation
# ---------------------------------------------------------------------------

def generate_sequence_data(num_samples=1500, seq_len=6, d_model=4, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((num_samples, seq_len, d_model)).astype(np.float32)
    # One random token is shifted by a large offset; label = sign of that offset
    anomaly_pos = rng.integers(0, seq_len, size=num_samples)
    offsets = rng.choice([-3.0, 3.0], size=num_samples).astype(np.float32)
    for i in range(num_samples):
        X[i, anomaly_pos[i], 0] += offsets[i]
    y = (offsets > 0).astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(y)
    
# ---------------------------------------------------------------------------
# 2. Custom autograd.Function building blocks
# ---------------------------------------------------------------------------

class LinearProjectionFunction(torch.autograd.Function):
    """
    Batched linear projection over the last dimension.

    Forward:
        Y[n, l, k] = Σ_d X[n, l, d] * W[d, k]

    Shapes:
        X : (N, L, D_in)
        W : (D_in, D_out)
        Y : (N, L, D_out)

    Backward:
        ∂L/∂X[n,l,d] =  Σ_k ∂L/∂Y[n,l,k] * ∂Y[n,l,k]/∂X[n,l,d] = Σ_k ∂L/∂Y[n,l,k] * W[d,k]
        ∂L/∂W[d,k]   = Σ_{n,l} ∂L/∂Y[n,l,k] * ∂Y[n,l,k]/∂W[d,k] =  Σ_{n,l} X[n,l,d] * ∂L/∂Y[n,l,k]
    """

    @staticmethod
    def forward(ctx, X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        Y = torch.einsum("nld,dk->nlk", X, W)
        ctx.save_for_backward(X, W)
        return Y

    @staticmethod
    def backward(ctx, grad_Y: torch.Tensor):
        X, W = ctx.saved_tensors

	# grad_Y: (N, L, D_out), W.t: (D_out, D_in) 
        grad_X = torch.einsum("nlk,kd->nld", grad_Y, W.t()) # (N, L, D_in)
        
        # X: (N, L, D_in), grad_Y: (N, L, D_out)
        grad_W = torch.einsum("nld,nlk->dk", X, grad_Y) # (D_in, D_out)

        return grad_X, grad_W


class ScaledDotProductScores(torch.autograd.Function):
    """
    Forward : scores = Q · Kᵀ / √d      shape (N, L, L)
    
    Shapes:
        Q      : (N, L, D)
        K      : (N, L, D)
        scores : (N, L, L)

    Backward:
        ∂L/∂Q[n,q,d] = Σ_k ∂L/∂scores[n,q,k] * ∂scores[n,q,k]/∂Q[n,q,d] = Σ_k ∂L/∂scores[n,q,k] * K[n,k,d] / √d
        ∂L/∂K[n,k,d] = Σ_q ∂L/∂scores[n,q,k] * ∂scores[n,q,k]/∂K[n,k,d] = Σ_q ∂L/∂scores[n,q,k] * Q[n,q,d] / √d
    """

    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        d = Q.shape[-1]
        scale = math.sqrt(d)

        scores = torch.einsum("nqd,nkd->nqk", Q, K) / scale
        ctx.save_for_backward(Q, K)
        ctx.scale = scale
        return scores

    @staticmethod
    def backward(ctx, grad_scores: torch.Tensor):
        Q, K = ctx.saved_tensors
        scale = ctx.scale

 	# grad_scores: (N, L, L), K: (N, L, D)
        grad_Q = torch.einsum("nqk,nkd->nqd", grad_scores, K) / scale # (N, L, D)
        
        # grad_scores: (N, L, L), Q: (N, L, D)
        grad_K = torch.einsum("nqk,nqd->nkd", grad_scores, Q) / scale # (N, L, D)

        return grad_Q, grad_K


class SoftmaxFunction(torch.autograd.Function):
    """
    Row-wise softmax over the last axis.
    
    Forward:
        s[n, q, k] = exp(logits[n, q, k]) / Σ_j exp(logits[n, q, j])

    Backward:
        ∂L/∂logits[n,q,k]
        = Σ_j ∂L/∂s[n,q,j] * ∂s[n,q,j]/∂logits[n,q,k]

        with softmax Jacobian
            ∂s[n,q,j]/∂logits[n,q,k] = s[n,q,j] * (1_{j=k} - s[n,q,k])

        so
            ∂L/∂logits[n,q,k]
            = s[n,q,k] * ( ∂L/∂s[n,q,k] - Σ_j ∂L/∂s[n,q,j] * s[n,q,j] )
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor) -> torch.Tensor:
        shifted = logits - logits.amax(dim=-1, keepdim=True)
        exp_z = shifted.exp()
        s = exp_z / exp_z.sum(dim=-1, keepdim=True)
        ctx.save_for_backward(s)
        return s

    @staticmethod
    def backward(ctx, grad_s: torch.Tensor):
        (s,) = ctx.saved_tensors
        
        # grad_s: (N, L, L), s: (N, L, L)
        # dot = Σ_j ∂L/∂s[n,q,j] * s[n,q,j]
        dot = (grad_s * s).sum(dim=-1, keepdim=True) #  (N,L,1)
        
        grad_logits = s * (grad_s - dot) #  # (N, L, L)
        return grad_logits


class AttentionOutputFunction(torch.autograd.Function):
    """
    Forward:
        out[n,q,d] = Σ_k attn_weights[n,q,k] * V[n,k,d]
        
    Shapes:
	attn_weights : (N, L, L)
	V       : (N, L, D)
	out     : (N, L, D)

    Backward:
        ∂L/∂weights[n,q,k] = Σ_d ∂L/∂out[n,q,d] * ∂out[n,q,d]/∂weights[n,q,k] = Σ_d ∂L/∂out[n,q,d] * V[n,k,d]
        ∂L/∂V[n,k,d]       = Σ_q ∂L/∂out[n,q,d] * ∂out[n,q,d]/∂V[n,k,d] = Σ_q weights[n,q,k] * ∂L/∂out[n,q,d]
    """
    

    @staticmethod
    def forward(ctx, attn_weights: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        out = torch.einsum("nqk,nkd->nqd", attn_weights, V)
        ctx.save_for_backward(attn_weights, V)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        attn_weights, V = ctx.saved_tensors

        # grad_out: (N, L, D), V: (N, L, D)
        grad_weights = torch.einsum("nqd,nkd->nqk", grad_out, V)  # (N, L, L)

        # weights: (N, L, L), grad_out: (N, L, D)
        grad_V = torch.einsum("nqk,nqd->nkd", attn_weights, grad_out)  # (N, L, D)

        return grad_weights, grad_V

class MeanPoolFunction(torch.autograd.Function):
    """
    Forward:
        pooled[n, d] = (1/L) * Σ_l attn_output[n, l, d]
        shape (N, d_model)

    Backward:
        ∂L/∂attn_output[n,l,d] = ∂L/∂pooled[n,d] * ∂pooled[n,d]/∂attn_output[n,l,d] = ∂L/∂pooled[n,d] * (1/L)
    """

    @staticmethod
    def forward(ctx, attn_output: torch.Tensor) -> torch.Tensor:
        L = attn_output.shape[1]
        pooled = attn_output.sum(dim=1) / L
        ctx.L = L
        return pooled

    @staticmethod
    def backward(ctx, grad_pooled: torch.Tensor):
        L = ctx.L
        
        # grad_pooled: (N, D)
        grad_attn_output = (grad_pooled.unsqueeze(1) / L).expand(-1, L, -1).clone() # (N, L, D)
        return grad_attn_output


class LinearHeadFunction(torch.autograd.Function):
    """
    Forward : logits = pooled @ W_O + b_O      shape (N, D_out)

    Shapes:
        pooled : (N, D_in)
        W_O    : (D_in, D_out)
        b_O    : (1, D_out)
        logits : (N, D_out)

    Backward:
        ∂L/∂pooled[n,d]   = Σ_k ∂L/∂logits[n,k] * ∂logits[n,k]/∂pooled[n,d] = Σ_k ∂L/∂logits[n,k] * W_O[d,k]
        ∂L/∂W_O[d,k]      = Σ_n ∂L/∂logits[n,k] * ∂logits[n,k]/∂W_O[d,k] = Σ_n pooled[n,d] * ∂L/∂logits[n,k]
        ∂L/∂b_O[1,k]      = Σ_n ∂L/∂logits[n,k]
    """

    @staticmethod
    def forward(
        ctx,
        pooled: torch.Tensor,
        W_O: torch.Tensor,
        b_O: torch.Tensor,
    ) -> torch.Tensor:
        logits = pooled @ W_O + b_O
        ctx.save_for_backward(pooled, W_O)
        return logits

    @staticmethod
    def backward(ctx, grad_logits: torch.Tensor):
        pooled, W_O = ctx.saved_tensors

        # grad_logits: (N, D_out), W_O: (D_in, D_out)
        grad_pooled = torch.einsum("nk,dk->nd", grad_logits, W_O)  # (N, D_in)

        # pooled.t(): (D_in, N), grad_logits: (N, D_out)
        grad_W_O = torch.einsum("nd,nk->dk", pooled, grad_logits)  # (D_in, D_out)

        # grad_logits: (N, D_out)
        grad_b_O = grad_logits.sum(dim=0, keepdim=True)  # (1, D_out)

        return grad_pooled, grad_W_O, grad_b_O


class SigmoidFunction(torch.autograd.Function):
    """
    Elementwise sigmoid.
    
    Forward:
        p = σ(z) = 1 / (1 + e^{-z})

    Backward:
        ∂L/∂z = ∂L/∂p * ∂p/∂z

        with sigmoid derivative
            ∂p/∂z = p * (1 - p)

        so
            ∂L/∂z = ∂L/∂p * p * (1 - p)
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor) -> torch.Tensor:
        p = 1.0 / (1.0 + torch.exp(-z))
        ctx.save_for_backward(p)
        return p

    @staticmethod
    def backward(ctx, grad_p: torch.Tensor):
        (p,) = ctx.saved_tensors
        grad_z = grad_p * p * (1.0 - p)
        return grad_z


class BCELossFunction(torch.autograd.Function):
    """
    Forward:
        loss = -(1/N) * Σ_n [ y_n log(p_n) + (1-y_n) log(1-p_n) ]

    Backward:
        grad_p[n]
          = grad_loss * (1/N) * [ -y_n/p_n + (1-y_n)/(1-p_n) ]
    """

    @staticmethod
    def forward(ctx, p: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        p_c = p.clamp(eps, 1.0 - eps)
        loss = -(y_true * p_c.log() + (1.0 - y_true) * (1.0 - p_c).log()).mean()
        ctx.save_for_backward(p_c, y_true)
        return loss

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor):
        p, y = ctx.saved_tensors
        N = p.shape[0]
        grad_p = grad_loss * (1.0 / N) * (-y / p + (1.0 - y) / (1.0 - p))
        return grad_p, None


# ---------------------------------------------------------------------------
# 3. TransformerClassifier (nn.Module)
# ---------------------------------------------------------------------------

class TransformerClassifier(nn.Module):
    """
    Two stacked single-head self-attention blocks
    followed by mean-pool -> sigmoid classifier.

    Parameters
    ----------
    d_model : int
    seq_len : int  (stored for reference; not strictly needed at runtime)
    """

    def __init__(self, d_model: int = 4, seq_len: int = 3, seed: int = 42) -> None:
        super().__init__()
        torch.manual_seed(seed)

        # ------------------------------------------------------------------
        # Attention block 1 parameters
        # ------------------------------------------------------------------
        # Q1 / K1 / V1 projection matrices  (d_model × d_model)
        self.W_Q1 = nn.Parameter(torch.randn(d_model, d_model) * 0.1)
        self.W_K1 = nn.Parameter(torch.randn(d_model, d_model) * 0.1)
        self.W_V1 = nn.Parameter(torch.randn(d_model, d_model) * 0.1)

        # ------------------------------------------------------------------
        # Attention block 2 parameters
        # ------------------------------------------------------------------
        # Q2 / K2 / V2 projection matrices  (d_model × d_model)
        self.W_Q2 = nn.Parameter(torch.randn(d_model, d_model) * 0.1)
        self.W_K2 = nn.Parameter(torch.randn(d_model, d_model) * 0.1)
        self.W_V2 = nn.Parameter(torch.randn(d_model, d_model) * 0.1)

        # ------------------------------------------------------------------
        # Output classifier  (d_model → 1)
        # ------------------------------------------------------------------
        self.W_O = nn.Parameter(torch.randn(d_model, 1) * 0.1)
        self.b_O = nn.Parameter(torch.zeros(1, 1))

        self.d_model = d_model
        self.seq_len = seq_len

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X : (N, L, d_model)
        Returns y_pred : (N,) probabilities in (0, 1)
        """

        # ==============================================================
        # 1st attention block
        # ==============================================================

        # 1. Project X to Q1, K1, V1   (N, L, d_model)
        Q1 = LinearProjectionFunction.apply(X, self.W_Q1)
        K1 = LinearProjectionFunction.apply(X, self.W_K1)
        V1 = LinearProjectionFunction.apply(X, self.W_V1)

        # 2. Scaled dot-product scores  (N, L, L)
        scores1 = ScaledDotProductScores.apply(Q1, K1)

        # 3. Attention weights via custom softmax  (N, L, L)
        attn_weights1 = SoftmaxFunction.apply(scores1)

        # 4. Weighted sum of V1  (N, L, d_model)
        H1 = AttentionOutputFunction.apply(attn_weights1, V1)

        # ==============================================================
        # 2nd attention block
        # ==============================================================

        # 5. Project H1 to Q2, K2, V2   (N, L, d_model)
        Q2 = LinearProjectionFunction.apply(H1, self.W_Q2)
        K2 = LinearProjectionFunction.apply(H1, self.W_K2)
        V2 = LinearProjectionFunction.apply(H1, self.W_V2)

        # 6. Scaled dot-product scores  (N, L, L)
        scores2 = ScaledDotProductScores.apply(Q2, K2)

        # 7. Attention weights via custom softmax  (N, L, L)
        attn_weights2 = SoftmaxFunction.apply(scores2)

        # 8. Weighted sum of V2  (N, L, d_model)
        H2 = AttentionOutputFunction.apply(attn_weights2, V2)

        # ==============================================================
        # Head
        # ==============================================================

        # 9. Mean-pool across sequence  (N, d_model)
        pooled = MeanPoolFunction.apply(H2)

        # 10. Linear projection  (N, 1)
        logits = LinearHeadFunction.apply(pooled, self.W_O, self.b_O)

        # 11. Sigmoid  (N, 1) → (N,)
        y_pred = SigmoidFunction.apply(logits).squeeze(-1)

        return y_pred


# ---------------------------------------------------------------------------
# 4. Training helpers
# ---------------------------------------------------------------------------

def binary_cross_entropy(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return BCELossFunction.apply(y_pred, y_true)


def train(
    model: TransformerClassifier,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_eval: torch.Tensor,
    y_eval: torch.Tensor,
    epochs: int = 1000,
    lr: float = 0.01,
    print_every: int = 50,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """
    Full-batch gradient descent.

    Returns:
        train_loss_history, eval_loss_history,
        train_acc_history,  eval_acc_history
    """
    optimizer = SGD(model.parameters(), lr=lr)
    train_loss_history: list[float] = []
    eval_loss_history: list[float] = []
    train_acc_history: list[float] = []
    eval_acc_history: list[float] = []

    for epoch in range(epochs):
        # -------------------------
        # Train step
        # -------------------------
        model.train()
        optimizer.zero_grad()

        y_pred_train = model(X_train)
        train_loss = binary_cross_entropy(y_pred_train, y_train)
        train_loss.backward()
        optimizer.step()

        train_loss_val = train_loss.item()
        train_loss_history.append(train_loss_val)

        with torch.no_grad():
            train_preds = (y_pred_train > 0.5).float()
            train_acc = (train_preds == y_train).float().mean().item()
            train_acc_history.append(train_acc)

        # -------------------------
        # Eval step
        # -------------------------
        model.eval()
        with torch.no_grad():
            y_pred_eval = model(X_eval)
            eval_loss = binary_cross_entropy(y_pred_eval, y_eval)
            eval_loss_val = eval_loss.item()
            eval_loss_history.append(eval_loss_val)

            eval_preds = (y_pred_eval > 0.5).float()
            eval_acc = (eval_preds == y_eval).float().mean().item()
            eval_acc_history.append(eval_acc)

        if epoch % print_every == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:>4}/{epochs}  "
                f"train_loss={train_loss_val:.6f}  "
                f"eval_loss={eval_loss_val:.6f}  "
                f"train_acc={train_acc:.2%}  "
                f"eval_acc={eval_acc:.2%}"
            )

    return (
        train_loss_history,
        eval_loss_history,
        train_acc_history,
        eval_acc_history,
    )

# ---------------------------------------------------------------------------
# 5. Single unittest demo
# ---------------------------------------------------------------------------

class TestTransformerSingleDemo(unittest.TestCase):
    """Single demo test: train the model and plot train/eval loss."""

    def test_demo_train_and_eval_loss(self) -> None:
        X, y = generate_sequence_data(
            num_samples=600,
            seq_len=3,
            d_model=4,
            seed=42,
        )

        X_train, y_train = X[:500], y[:500]
        X_eval, y_eval = X[500:], y[500:]

        model = TransformerClassifier(d_model=4, seq_len=3, seed=123)

        train_history, eval_history, train_acc_history, eval_acc_history = train(
            model,
            X_train,
            y_train,
            X_eval,
            y_eval,
            epochs=1000,
            lr=0.01,
            print_every=50,
        )

        print(f"\nInitial train loss: {train_history[0]:.6f}")
        print(f"Final train loss:   {train_history[-1]:.6f}")
        print(f"Initial eval loss:  {eval_history[0]:.6f}")
        print(f"Final eval loss:    {eval_history[-1]:.6f}")

        with torch.no_grad():
            train_preds = (model(X_train) > 0.5).float()
            eval_preds = (model(X_eval) > 0.5).float()
            train_acc = (train_preds == y_train).float().mean().item()
            eval_acc = (eval_preds == y_eval).float().mean().item()

        print(f"Train accuracy: {train_acc:.2%}")
        print(f"Eval accuracy:  {eval_acc:.2%}")

	# Loss plot
        plt.figure(figsize=(8, 5))
        plt.plot(train_history, label="Train loss", linewidth=3.0)
        plt.plot(eval_history, label="Eval loss", linewidth=3.0)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train vs Eval Loss")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Accuracy plot
        plt.figure(figsize=(8, 5))
        plt.plot(train_acc_history, label="Train accuracy", linewidth=3.0)
        plt.plot(eval_acc_history, label="Eval accuracy", linewidth=3.0)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Train vs Eval Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        

        self.assertLess(train_history[-1], train_history[0])


# ---------------------------------------------------------------------------
# 6. Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
