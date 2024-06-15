import torch
import torchvision
from ResNet50 import res50

"""
Backbone weights pretrained on iNaturalist from BBN paper
"""

# Import the weights
weights_path = "ConceptWhitening/models/pretrained/iNaturalist_180epoch_best.pth"
model = res50(pretrain_model=weights_path)
print(model)
