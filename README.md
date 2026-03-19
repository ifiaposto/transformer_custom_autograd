# transformer_custom_autograd
A minimal PyTorch/Colab walkthrough of a transformer-style classifier where each forward-pass step is implemented with custom `torch.autograd.Function` and hand-derived backward passes.
# Transformer Classifier — Custom Autograd from Scratch

A minimal PyTorch/Colab walkthrough of a transformer-style classifier where each forward-pass step is implemented with custom `torch.autograd.Function` and hand-derived backward passes.

This notebook is meant to build intuition for what autograd is doing under the hood in attention-based models, and to serve as a compact reference for writing custom backward functions in PyTorch.

## What’s inside

- manual autograd for core transformer-style operations
- custom forward and backward passes for:
  - linear projections
  - scaled dot-product attention scores
  - softmax
  - attention-weighted sum
  - mean pooling
  - linear head
  - sigmoid
  - binary cross-entropy loss
- inline gradient derivations and comments
- a simple synthetic sequence-classification task
- training curves / accuracy visualization

## Why this repo

Most transformer examples rely entirely on PyTorch’s built-in autograd. This notebook takes the opposite approach: implement the graph one operation at a time and explicitly define how gradients flow backward.

It is useful if you want to:

- understand autograd at a lower level
- sanity-check gradient flow through attention blocks
- learn the structure of custom `torch.autograd.Function`
- experiment with customized backward passes

## Files
- `transformer_autograd.py`
- `transformer_custom_autograd.ipynb` — main Colab  notebook

## Run

Open the notebook locally or in Colab and run all cells.  
Dependencies are minimal:

```bash
pip install torch numpy matplotlib
