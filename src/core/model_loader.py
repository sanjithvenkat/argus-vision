from pathlib import Path

import torch

from src.models.simple_cnn import SimpleCNN


def load_colored_mnist_model(checkpoint_path, device="cpu"):
    """
    Load the Colored-MNIST benchmark model from a state_dict checkpoint.
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    model = SimpleCNN(num_classes=10)

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model