import torch
from torch import nn


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Convolutional layers learn local patterns like edges or textures.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            # Pooling halves height and width while keeping the strongest responses.
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # After the feature extractor, flatten the feature maps and classify them.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def main() -> None:
    # Shape format for image batches is:
    # (batch_size, channels, height, width)
    batch = torch.randn(4, 1, 28, 28)
    model = TinyCNN()

    print("Input shape:", batch.shape)

    x = batch
    for layer in model.features:
        x = layer(x)
        # Printing shapes after each layer is the fastest way to learn CNNs.
        print(f"{layer.__class__.__name__:>10} -> {tuple(x.shape)}")

    logits = model(batch)
    print("Output logits shape:", logits.shape)
    print("Interpretation: batch size 4, 3 class scores each")


if __name__ == "__main__":
    main()
