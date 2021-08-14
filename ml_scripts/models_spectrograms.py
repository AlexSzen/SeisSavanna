#!/usr/bin/env python3

"""Define architecture for spectrogram classification."""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class PretrainedCNN(nn.Module):
    """CNN pretrained on Imagenet."""

    def __init__(self, num_classes, class_counts, init_bias=False, arch="resnet", pretrained=True):
        """ Add classification head to pretrained CNN."""

        super(PretrainedCNN, self).__init__()

        
        self.model = models.squeezenet1_0(pretrained=True)
        self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        self.model.num_classes = num_classes


        
    def forward(self, x):
        return self.model(x)
        

