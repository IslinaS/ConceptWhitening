import torch
import torchvision
from models.ResNet50 import res50

"""
Backbone weights pretrained on iNaturalist from BBN paper
"""

# Import the weights
weights_path = "models/iNaturalist_180epoch_best.pth"
inat_weights = torch.load(weights_path)

# Open the model
model = res50(pretrain=False)
model.load_model(pretrain=weights_path)    
print(model)    