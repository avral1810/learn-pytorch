# PyTorch Tutorials

This folder is organized as a progression. Run the scripts in order.

Each script also has a companion reading handout:

- Source notes: `tutorials/notes/`
- Generated PDFs: `tutorials/pdfs/`
- Downloaded lesson images: `tutorials/assets/`

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision matplotlib
```

Verify PyTorch:

```bash
python3 -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)"
```

## Learning path

1. `00_tensors_and_autograd.py`
   Learn tensors, shapes, gradients, and a basic optimization loop.
2. `01_shapes_gradients_and_reshape.py`
   Learn shape mechanics, broadcasting, `reshape`, `view`, `unsqueeze`, `squeeze`, `requires_grad`, `no_grad`, and `detach`.
3. `02_linear_regression_from_scratch.py`
   Learn regression using raw tensors, manual loss, and manual parameter updates.
4. `03_linear_regression_with_nn_module.py`
   Learn the same regression problem using `nn.Module`, `DataLoader`, and an optimizer.
5. `04_logistic_regression_from_scratch.py`
   Learn binary classification using logits, sigmoid, and BCE from scratch.
6. `05_logistic_regression_with_nn_module.py`
   Learn PyTorch's standard binary-classification workflow with `BCEWithLogitsLoss`.
7. `06_manual_mlp_from_scratch.py`
   Build a small neural net with raw parameter tensors and manual forward passes.
8. `07_basic_nn_mlp_with_nn_module.py`
   Build a small MLP classifier with `nn.Sequential` after seeing the from-scratch version.
9. `08_cnn_basics.py`
   Understand convolutions, pooling, and a tiny CNN.
10. `09_vision_classifier.py`
   Train a CNN on generated image patterns and practice vision workflow.
11. `14_basic_rnn_sequence_classifier.py`
   Learn the core recurrent-network idea with embeddings, hidden state, and a sequence classifier built from the final RNN summary.
12. `10_lstm_sequence_classifier.py`
   Learn sequence modeling with embeddings, hidden state, and LSTM output shapes.
13. `11_transformer_basics.py`
   Build a minimal Transformer encoder classifier.
14. `12_toy_gan.py`
   Learn generator-vs-discriminator training on a tiny 1D distribution.
15. `13_device_cpu_to_mps.py`
   Learn how to move the same code from CPU to Apple Metal (`mps`).

## Suggested rhythm

- Read one file top to bottom.
- Run it.
- Change one thing at a time: learning rate, hidden size, batch size, epochs.
- Print tensor shapes until they feel obvious.
- Read the note PDF before each script run, especially for lesson `01`.

## What to focus on

- Fundamentals: tensors, gradients, parameters, loss, optimization
- Classical ML in PyTorch: linear regression, logistic regression
- From scratch learning: raw parameter tensors before `nn.Module`
- Neural nets: forward pass, activation functions, overfitting, train vs eval
- CNNs: channels, kernels, feature maps, pooling
- Sequence models: embeddings, hidden state, LSTM intuition
- RNNs: recurrence, final hidden summaries, and sequence order
- Vision: image tensors, augmentations, normalization, classifier pipeline
- Transformers: embeddings, positional information, attention intuition
- Generative models: GAN basics, adversarial training, `detach()` in practice
- Devices: CPU first, then switch to `mps`

## Companion PDFs

If you update the text notes and want to rebuild the PDFs:

```bash
bash tutorials/build_pdfs.sh
```

Each PDF now includes:
- a sourced image or diagram for the lesson
- the written explanation
- exercises or self-check questions
