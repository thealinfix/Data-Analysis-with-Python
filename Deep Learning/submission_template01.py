import numpy as np
import torch
import re
from torch import nn

# Task 1: Save the PyTorch version
version = torch.__version__

# Task 2: Create a model with the specified architecture
def create_model():
    model = nn.Sequential(
        nn.Linear(784, 256, bias=True),
        nn.ReLU(),
        nn.Linear(256, 16, bias=True),
        nn.ReLU(),
        nn.Linear(16, 10, bias=True)
    )
    return model

# Task 3: Count the number of parameters in a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Tests
model = create_model()
for param in model.parameters():
    nn.init.constant_(param, 1.)

assert torch.allclose(model(torch.ones((1, 784))), torch.ones((1, 10)) * 3215377.), 'Incorrect model structure'

small_model = nn.Linear(128, 256)
assert count_parameters(small_model) == 128 * 256 + 256, 'Incorrect parameter count for small_model'

medium_model = nn.Sequential(*[nn.Linear(128, 32, bias=False), nn.ReLU(), nn.Linear(32, 10, bias=False)])
assert count_parameters(medium_model) == 128 * 32 + 32 * 10, 'Incorrect parameter count for medium_model'

print("Seems fine!")
