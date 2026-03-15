import torch


def make_data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)

    # We create a simple line with noise:
    # y = 3.5x + 1.2 + noise
    x = torch.linspace(-2, 2, steps=200).unsqueeze(1)
    noise = 0.25 * torch.randn_like(x)
    y = 3.5 * x + 1.2 + noise
    return x, y


def main() -> None:
    x, y = make_data()

    # x shape is (200, 1):
    # 200 training examples, 1 feature per example.
    # y shape is also (200, 1) because we want one target value per example.

    # These are raw learnable tensors, not an nn.Module yet.
    # We are doing the core math ourselves so you can see what PyTorch tracks.
    weight = torch.randn(1, 1, requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)

    learning_rate = 0.05

    for epoch in range(100):
        # Forward pass from scratch:
        # prediction = x @ weight + bias
        # Shapes:
        # (200, 1) @ (1, 1) -> (200, 1)
        # bias shape (1,) then broadcasts across all 200 rows.
        preds = x @ weight + bias

        # Mean squared error from scratch.
        # preds - y gives one error value per example.
        # Squaring makes large errors hurt more.
        # mean() turns all example errors into one scalar loss.
        loss = ((preds - y) ** 2).mean()

        # Compute gradients d(loss)/d(weight) and d(loss)/d(bias).
        # After this line:
        # weight.grad and bias.grad hold the slopes telling us
        # how to move those parameters to reduce the loss.
        loss.backward()

        # Parameter updates should not become part of the gradient graph.
        with torch.no_grad():
            weight -= learning_rate * weight.grad
            bias -= learning_rate * bias.grad

        # Gradients accumulate in PyTorch, so clear them every step.
        weight.grad.zero_()
        bias.grad.zero_()

        if epoch % 10 == 0 or epoch == 99:
            print(
                f"epoch={epoch:03d} loss={loss.item():.4f} "
                f"weight={weight.item():.3f} bias={bias.item():.3f}"
            )

    print("Final learned line is approximately: y = weight * x + bias")


if __name__ == "__main__":
    main()
