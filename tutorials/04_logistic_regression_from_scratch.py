import torch


def make_data(n: int = 400) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(12)
    # Each example has 2 input features.
    x = torch.randn(n, 2)

    # Create a linear boundary plus noise.
    scores = 2.0 * x[:, 0] - 1.2 * x[:, 1] + 0.3 * torch.randn(n)
    # unsqueeze(1) gives labels shape (n, 1), which matches the model output shape.
    y = (scores > 0).float().unsqueeze(1)
    return x, y


def sigmoid(z: torch.Tensor) -> torch.Tensor:
    # Sigmoid converts any real value into a number between 0 and 1.
    # That lets us interpret the result as a probability for class 1.
    return 1 / (1 + torch.exp(-z))


def binary_cross_entropy(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Clamp prevents log(0), which would produce infinities.
    eps = 1e-7
    preds = preds.clamp(eps, 1 - eps)
    return -(targets * preds.log() + (1 - targets) * (1 - preds).log()).mean()


def main() -> None:
    x, y = make_data()

    # weight shape (2, 1) means:
    # 2 input features -> 1 output logit.
    weight = torch.randn(2, 1, requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    learning_rate = 0.1

    for epoch in range(120):
        # logits are raw scores, not probabilities yet.
        logits = x @ weight + bias
        # probs are probabilities after the sigmoid squashes logits into (0, 1).
        probs = sigmoid(logits)
        loss = binary_cross_entropy(probs, y)

        # backward() fills weight.grad and bias.grad.
        loss.backward()

        # The update step itself is not part of the model.
        with torch.no_grad():
            weight -= learning_rate * weight.grad
            bias -= learning_rate * bias.grad

        # Clear accumulated gradients before the next iteration.
        weight.grad.zero_()
        bias.grad.zero_()

        if epoch % 20 == 0 or epoch == 119:
            # Threshold 0.5 is the simplest way to turn probabilities into class ids.
            preds = (probs >= 0.5).float()
            acc = (preds == y).float().mean().item()
            print(f"epoch={epoch:03d} loss={loss.item():.4f} acc={acc:.3f}")


if __name__ == "__main__":
    main()
