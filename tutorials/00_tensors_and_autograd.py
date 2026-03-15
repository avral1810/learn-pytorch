import torch


def main() -> None:
    print("PyTorch tensors and autograd")

    # A tensor is PyTorch's core data structure.
    # You can think of it as a multi-dimensional array with extra features:
    # shape tracking, mathematical operations, and optional gradient support.
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    print("x shape:", x.shape)
    print("y shape:", y.shape)

    # Elementwise addition keeps the same shape.
    print("x + y:\n", x + y)

    # Matrix multiplication combines rows of x with columns of y.
    print("x @ y:\n", x @ y)

    # This is a learnable scalar parameter.
    # requires_grad=True tells PyTorch to track how the loss depends on w.
    w = torch.tensor(2.0, requires_grad=True)

    for step in range(10):
        # The target is to make 3 * w close to 9, so w should move toward 3.
        loss = (w * 3 - 9) ** 2

        # backward() computes d(loss)/d(w) using autograd.
        loss.backward()

        # We update the parameter manually while gradients are temporarily disabled.
        with torch.no_grad():
            w -= 0.1 * w.grad

        print(f"step={step:02d} w={w.item():.4f} loss={loss.item():.4f}")

        # Gradients accumulate by default, so clear them before the next step.
        w.grad.zero_()

    print("Expected w is close to 3.0 because 3 * w should approach 9.")


if __name__ == "__main__":
    main()
