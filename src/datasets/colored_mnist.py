"""
Colored-MNIST dataset construction and loading.

Reusable extraction of the FINAL/CORRECTED shortcut-learning dataset from
the original Colored-MNIST research project. Produces a dataset where:

- Easy set: color correlates with digit label at `bias` probability (default 0.99).
- Hard set: color is deterministically inverted per digit (HARD_COLOR_MAP),
  so a model relying on color will fail on this split.
- Both foreground (digit stroke) AND background carry color information —
  background is tinted at 50% brightness of the foreground color, so color
  is a global, low-cost signal rather than a stroke-only cue.
"""

import random

import numpy as np
import torch
from torch.utils.data import Dataset

# One canonical color per digit for the Easy (biased) set.
COLOR_MAP = {
    0: (220, 20, 20),     # Red
    1: (20, 200, 20),     # Green
    2: (20, 20, 220),     # Blue
    3: (220, 220, 20),    # Yellow
    4: (180, 20, 180),    # Magenta
    5: (20, 200, 200),    # Cyan
    6: (220, 120, 20),    # Orange
    7: (120, 20, 220),    # Purple
    8: (20, 120, 220),    # Sky Blue
    9: (220, 20, 120),    # Pink
}

# Deterministic swapped mapping for the Hard set — digit d gets the
# canonical color of digit (d+5) % 10. Makes the bias perfectly measurable.
HARD_COLOR_MAP = {
    0: COLOR_MAP[5],
    1: COLOR_MAP[6],
    2: COLOR_MAP[7],
    3: COLOR_MAP[8],
    4: COLOR_MAP[9],
    5: COLOR_MAP[0],
    6: COLOR_MAP[1],
    7: COLOR_MAP[2],
    8: COLOR_MAP[3],
    9: COLOR_MAP[4],
}

COLOR_NAMES = {
    0: "Red", 1: "Green", 2: "Blue", 3: "Yellow", 4: "Magenta",
    5: "Cyan", 6: "Orange", 7: "Purple", 8: "Sky Blue", 9: "Pink",
}


def add_color_variation(color, variation=20):
    """Small per-channel noise so images aren't pixel-identical, while
    keeping color identity clear."""
    r, g, b = color
    r = int(np.clip(r + np.random.randint(-variation, variation + 1), 0, 255))
    g = int(np.clip(g + np.random.randint(-variation, variation + 1), 0, 255))
    b = int(np.clip(b + np.random.randint(-variation, variation + 1), 0, 255))
    return (r, g, b)


def colorize(img_array, digit, is_hard=False, bias=0.99):
    """
    Colorize a single 28x28 grayscale MNIST image.

    Easy set (is_hard=False): canonical color with probability `bias`,
        a random other digit's color otherwise.
    Hard set (is_hard=True): always the deterministic HARD_COLOR_MAP
        inversion for that digit.

    Background = fg_color // 2 (50% brightness) so color floods every
    pixel, not just the digit stroke — this is what makes color the
    cheapest available signal for the optimizer.
    """
    rgb = np.zeros((28, 28, 3), dtype=np.uint8)

    if not is_hard:
        if random.random() < bias:
            base_color = COLOR_MAP[digit]
        else:
            base_color = COLOR_MAP[random.choice([d for d in range(10) if d != digit])]
    else:
        base_color = HARD_COLOR_MAP[digit]

    fg_color = add_color_variation(base_color, variation=20)
    bg_color = (fg_color[0] // 2, fg_color[1] // 2, fg_color[2] // 2)

    # MNIST strokes are ~200-255, background is ~0-30; 128 cleanly separates them.
    foreground_mask = img_array > 128
    rgb[foreground_mask] = fg_color
    rgb[~foreground_mask] = bg_color

    return rgb


def build_dataset(mnist_data, is_hard=False, bias=0.99):
    """Colorize an entire MNIST split (a torchvision MNIST dataset of
    (PIL image, label) pairs) into (images, labels) numpy arrays."""
    images, labels = [], []
    for i, (img, label) in enumerate(mnist_data):
        img_array = np.array(img)
        images.append(colorize(img_array, label, is_hard=is_hard, bias=bias))
        labels.append(label)
        if i % 10000 == 0:
            print(f"  {i}/{len(mnist_data)}")
    return np.array(images), np.array(labels)


class ColoredMNIST(Dataset):
    """Wraps colorized (N, H, W, C) uint8 arrays as a torch Dataset,
    converting to (N, C, H, W) float tensors normalized to [0, 1]."""

    def __init__(self, images, labels):
        self.images = torch.FloatTensor(images).permute(0, 3, 1, 2) / 255.0
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]