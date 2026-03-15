import torch


def print_header(title: str) -> None:
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def shape_examples() -> None:
    print_header("1. Shapes")

    scalar = torch.tensor(5.0)
    vector = torch.tensor([1.0, 2.0, 3.0])
    matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    cube = torch.arange(24.0).reshape(2, 3, 4)

    # The number of axes is called the rank or number of dimensions.
    print("scalar shape:", scalar.shape, "ndim:", scalar.ndim)
    print("vector shape:", vector.shape, "ndim:", vector.ndim)
    print("matrix shape:", matrix.shape, "ndim:", matrix.ndim)
    print("cube shape:", cube.shape, "ndim:", cube.ndim)


def reshape_examples() -> None:
    print_header("2. Reshape, View, Unsqueeze, Squeeze")

    x = torch.arange(12.0)
    print("original x:", x)
    print("original shape:", x.shape)

    # reshape changes how the same values are arranged.
    # 12 numbers can be reorganized into 3 rows and 4 columns.
    x_3x4 = x.reshape(3, 4)
    print("reshape(3, 4):\n", x_3x4)

    # view is similar to reshape, but has stricter memory-layout requirements.
    x_2x6 = x.view(2, 6)
    print("view(2, 6):\n", x_2x6)

    # unsqueeze inserts a dimension of size 1.
    column = x.unsqueeze(1)
    print("unsqueeze(1) shape:", column.shape)

    # squeeze removes dimensions whose size is 1.
    back_to_flat = column.squeeze(1)
    print("squeeze(1) shape:", back_to_flat.shape)

    # -1 tells PyTorch to infer that dimension automatically.
    x_auto = x.reshape(3, -1)
    print("reshape(3, -1) shape:", x_auto.shape)


def broadcasting_examples() -> None:
    print_header("3. Broadcasting")

    matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    bias = torch.tensor([10.0, 20.0, 30.0])

    # Broadcasting means PyTorch can expand compatible dimensions
    # without you manually copying data.
    result = matrix + bias
    print("matrix shape:", matrix.shape)
    print("bias shape:", bias.shape)
    print("matrix + bias:\n", result)


def matmul_examples() -> None:
    print_header("4. Matrix Multiplication")

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    c = a @ b

    print("a shape:", a.shape)
    print("b shape:", b.shape)
    print("a @ b:\n", c)


def grad_examples() -> None:
    print_header("5. requires_grad and backward()")

    # This tensor is a learnable parameter because gradients are enabled.
    w = torch.tensor(2.0, requires_grad=True)
    x = torch.tensor(4.0)
    target = torch.tensor(11.0)

    prediction = w * x
    loss = (prediction - target) ** 2

    print("prediction:", prediction.item())
    print("loss:", loss.item())

    # backward computes the derivative of loss with respect to w.
    loss.backward()
    print("gradient stored in w.grad:", w.grad.item())


def no_grad_and_detach_examples() -> None:
    print_header("6. torch.no_grad() and detach()")

    w = torch.tensor(3.0, requires_grad=True)
    y = w * 5
    print("y requires_grad:", y.requires_grad)

    # no_grad temporarily disables graph tracking.
    with torch.no_grad():
        z = w * 5
    print("z requires_grad after no_grad:", z.requires_grad)

    # detach creates a tensor that shares value data but is disconnected
    # from the computation graph.
    detached = y.detach()
    print("detached requires_grad:", detached.requires_grad)

    # This is the pattern used during optimization updates.
    w.grad = None
    loss = (w * 2 - 7) ** 2
    loss.backward()
    print("gradient before update:", w.grad.item())

    with torch.no_grad():
        w -= 0.1 * w.grad

    print("updated w:", w.item())


def main() -> None:
    shape_examples()
    reshape_examples()
    broadcasting_examples()
    matmul_examples()
    grad_examples()
    no_grad_and_detach_examples()


if __name__ == "__main__":
    main()
