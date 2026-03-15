import torch
from torch import nn


def get_device() -> torch.device:
    # Prefer CUDA if available, then Apple Metal, otherwise CPU.
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    device = get_device()
    print("Using device:", device)

    # Move the model to the chosen device once.
    model = nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    ).to(device)

    # Create tensors directly on the same device as the model.
    x = torch.randn(8, 4, device=device)
    y = torch.randint(0, 2, (8,), device=device)

    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)

    # Backpropagation works the same way on CPU, CUDA, or MPS.
    loss.backward()

    print("Forward and backward pass succeeded on", device)
    print("When you are ready for Metal, replace CPU tensors/model with `.to('mps')` or a device variable.")


if __name__ == "__main__":
    main()
