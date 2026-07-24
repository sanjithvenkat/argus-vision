import torch

from src.core.model_loader import load_colored_mnist_model


CHECKPOINT = (
    "benchmarks/colored_mnist/checkpoints/"
    "baseline_shortcut.pth"
)


model = load_colored_mnist_model(CHECKPOINT)

print("Checkpoint loaded successfully.")
print(model)


dummy_input = torch.randn(1, 3, 28, 28)

with torch.no_grad():
    output = model(dummy_input)


print("\nInput shape:", dummy_input.shape)
print("Output shape:", output.shape)

assert output.shape == (1, 10)

print("\nARGUS MODEL LOADER TEST PASSED")