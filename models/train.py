import torch
import torchvision
from ResNet50 import res50

"""
Backbone weights pretrained on iNaturalist from BBN paper
"""

# Import the weights
import sys

# Print all directories in the Python search path
for index, path in enumerate(sys.path):
    print(f"Directory {index}: {path}")

weights_path = "models/pretrained/iNaturalist2018_res50_180epoch.pth"
model = res50(pretrained_model=weights_path)
print(model)
