import torch
import torch.nn as nn
import numpy as np
import random

from collections import OrderedDict
from typing import Type
from models.IterNorm import IterNormRotation as CWLayer

"""
Code adapted from BBN.
"""


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(True)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        if stride != 1 or self.expansion * planes != inplanes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )
        else:
            self.downsample = None
        self.relu = nn.ReLU(True)

    def forward(self, x, region=None, orig_x_dim=None):
        # The isinstance see if we are passing to a CWLayer. If we are, send the region
        out = self.conv1(x)
        out = (self.bn1(out, X_redact_coords=region, orig_x_dim=orig_x_dim)
               if isinstance(self.bn1, CWLayer) else self.bn1(out))
        out = self.relu1(out)

        out = self.conv2(out)
        out = (self.bn2(out, X_redact_coords=region, orig_x_dim=orig_x_dim)
               if isinstance(self.bn2, CWLayer) else self.bn2(out))
        out = self.relu2(out)

        out = self.conv3(out)
        out = (self.bn3(out, X_redact_coords=region, orig_x_dim=orig_x_dim)
               if isinstance(self.bn3, CWLayer) else self.bn3(out))

        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block_type: Type[BottleNeck],
        num_blocks,
        high_to_low,
        num_classes=200,
        last_layer_stride=2,
        whitened_layers=[[0], [0], [0], [0]],
        cw_lambda=0.1,
        activation_mode="pool_max",
        pretrain_loc=None,
        vanilla_pretrain=True  # When true, expects no concept whitening modules
    ):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.block = block_type
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(num_blocks[0], 64)
        self.layer2 = self._make_layer(
            num_blocks[1], 128, stride=2
        )
        self.layer3 = self._make_layer(
            num_blocks[2], 256, stride=2
        )
        self.layer4 = self._make_layer(
            num_blocks[3], 512, stride=last_layer_stride,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block_type.expansion, num_classes)

        if pretrain_loc and vanilla_pretrain:
            self.load_model(pretrain=pretrain_loc)

        # The architecture is structured as [3, 4, 6, 4], stored in num_blocks
        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        self.cw_layers: list[CWLayer] = []
        self.BN_DIM = [64, 128, 256, 512]  # This was what was given in the pretrained model

        self.num_low_level = sum([len(high_to_low[high_concept]) for high_concept in high_to_low])
        self.num_high_level = len(high_to_low)

        self.concept_matrix = self._generate_concept_matrix(high_to_low)

        for i in range(4):
            for whitened_layer in whitened_layers[i]:
                # All params in train_params.yaml can be changed
                new_cw_layer = CWLayer(
                    num_features=self.BN_DIM[i],
                    activation_mode=activation_mode,
                    concept_mat=self.concept_matrix,
                    latent_mappings=self._generate_latent_mappings(high_to_low, latent_dim=self.BN_DIM[i]),
                    cw_lambda=cw_lambda
                )
                self.layers[i][whitened_layer].bn1 = new_cw_layer
                self.cw_layers.append(new_cw_layer)

        if pretrain_loc and not vanilla_pretrain:
            self.load_model(pretrain=pretrain_loc)

    def change_mode(self, mode):
        """
        Change the training mode for each whitened layer.
        If mode = -1, no update for gradient matrix G.
                = 0 to k-1, the column index of gradient matrix G that needs to be updated i.e.
                            the index of the current concept while minimizing concept alignment loss.
        """
        for cw_layer in self.cw_layers:
            cw_layer.mode = mode

    def update_rotation_matrix(self):
        """
        Update the rotation matrix R using accumulated gradient matrix G for each whitened layer.
        """
        for cw_layer in self.cw_layers:
            cw_layer.update_rotation_matrix()

    def load_model(self, pretrain):
        print("Loading backbone pretrain model from {}......".format(pretrain))
        model_dict = self.state_dict()
        pretrain_dict: dict = torch.load(pretrain)
        pretrain_dict = pretrain_dict["state_dict"] if "state_dict" in pretrain_dict else pretrain_dict

        new_dict = OrderedDict()

        for key, value in pretrain_dict.items():
            key: str
            if "cb_block" in key or "rb_block" in key:
                continue
            if key.startswith("module"):
                key = key[7:]
            if "fc" not in key and "classifier" not in key:
                key = key.replace("backbone.", "")
                new_dict[key] = value

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("Backbone model has been loaded......")

    def _make_layer(self, num_block, planes, stride=1):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for now_stride in strides:
            layers.append(
                self.block(
                    self.inplanes, planes, stride=now_stride
                )
            )
            self.inplanes = planes * self.block.expansion
        return nn.Sequential(*layers)

    def _generate_concept_matrix(self, high_to_low: dict):
        """
        Generate a concept indicator matrix, which is a square 0-1 matrix. Each (i, j)-th entry is 1 if
        the i-th and j-th low level concepts belong to the same high level concept, and 0 otherwise.
        The concepts are indexed based on their order in `low_level.json`, but translated to start from index 0.
        This matrix is used to train for concept whitening loss by the CWLayer.

        For example, the concept indicator matrix
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [0, 0, 1, 1],
         [0, 0, 1, 1]]
        signifies that concepts 0 and 1 are in the same high level concept, and so are concepts 2 and 3.

        Params:
        -------
        - high_to_low (dictionary): Mapping from high level concept to low level concept

        Returns:
        --------
        - torch.Tensor: The concept indicator matrix
        """
        # Create an empty concept matrix of size num_low_level x num_low_level
        concept_matrix = torch.zeros((self.num_low_level, self.num_low_level), dtype=torch.int)

        # Populate the concept matrix
        for indices in high_to_low.values():
            for i in indices:
                for j in indices:
                    concept_matrix[i, j] = 1

        return concept_matrix
    
    def _generate_latent_mappings(self, high_to_low: dict, latent_dim):
        random.seed(42)

        indices = np.linspace(0, latent_dim, self.num_high_level + 1, dtype=int)
        partitions = [range(indices[i], indices[i + 1]) for i in range(self.num_high_level)]
        latent_mappings = {}

        for i, high_concept in enumerate(high_to_low.keys()):
            for low_concept in high_to_low[high_concept]:
                latent_mappings[low_concept] = random.choice(partitions[i])

        return latent_mappings

    def forward(self, x, region=None, orig_x_dim=None):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool(out)

        # Process layers potentially containing CWLayer instances
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                out = block(out, region=region, orig_x_dim=orig_x_dim)

        # FC layer to predict the class
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


def res50(whitened_layers, high_to_low, cw_lambda, activation_mode="pool_max",
          pretrained_model=None, vanilla_pretrain=True):
    return ResNet(
        BottleNeck,
        [3, 4, 6, 3],
        last_layer_stride=2,
        whitened_layers=whitened_layers,
        high_to_low=high_to_low,
        cw_lambda=cw_lambda,
        activation_mode=activation_mode,
        pretrain_loc=pretrained_model,
        vanilla_pretrain=vanilla_pretrain
    )
