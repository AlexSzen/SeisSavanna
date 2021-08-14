#!/usr/bin/env python3

""" Defines ML architectures that will be used in train.py """

import torch
import torch.nn as nn

class cnn1(nn.Module):
    """ Typical CNN for binary classification"""

    def __init__(self, num_classes):
        """ Defines layers, hardcoded size at the moment."""

        super(cnn1, self).__init__()


        self.activation = nn.LeakyReLU()

        self.conv1 = nn.Conv1d(3, 32, kernel_size = 11, padding = 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.downconv1 = nn.Conv1d(32, 64, kernel_size = 4, stride = 2, padding = 1)

        self.conv2 = nn.Conv1d(64, 64, kernel_size = 11, padding = 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.downconv2 = nn.Conv1d(64, 128, kernel_size = 4, stride = 2, padding = 1)

        self.conv3 = nn.Conv1d(128, 128, kernel_size = 11, padding = 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.downconv3 = nn.Conv1d(128, 256, kernel_size = 4, stride = 2, padding = 1)

        self.conv4 = nn.Conv1d(256, 256, kernel_size = 3, padding = 1)
        self.bn4 = nn.BatchNorm1d(256)
        self.downconv4 = nn.Conv1d(256, 256, kernel_size = 4, stride = 2, padding = 1)
        
        self.conv5 = nn.Conv1d(256, 256, kernel_size = 3, padding = 1)
        self.bn5 = nn.BatchNorm1d(256)
        self.downconv5 = nn.Conv1d(256, 256, kernel_size = 4, stride = 2, padding = 1)


        self.conv6 = nn.Conv1d(256, 256, kernel_size = 1, padding = 1)
        self.bn6 = nn.BatchNorm1d(256)

        self.conv7 = nn.Conv1d(256, num_classes, kernel_size = 1, padding = 1)
        self.bn7 = nn.BatchNorm1d(num_classes)

        self.model = nn.Sequential(
                    self.conv1, self.bn1, self.activation, self.downconv1,
                    self.conv2, self.bn2, self.activation, self.downconv2,
                    self.conv3, self.bn3, self.activation, self.downconv3,
                    self.conv4, self.bn4, self.activation, self.downconv4,
                    self.conv5, self.bn5, self.activation, self.downconv5,
                    self.conv6, self.bn6, self.activation,
                    self.conv7, self.bn7, self.activation 
                    )

    def forward(self, x):

        out=self.model(x)
        #global averaging over "spatial" (time here) dimension, following all-cnn paper
        out=out.mean(2)
        return out

class cnn_simple(nn.Module):
    """Simple conv-maxpool cnn"""

    def __init__(self, num_classes):
        """ Defines layers, hardcoded size at the moment."""

        super(cnn_simple, self).__init__()


        self.activation = nn.LeakyReLU()

        self.conv1 = nn.Conv1d(3, 64, kernel_size = 3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.mp1 = nn.MaxPool1d(3, stride=2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size = 3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.mp2 = nn.MaxPool1d(3, stride=2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size = 3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.mp3 = nn.MaxPool1d(3, stride=2)

        self.conv4 = nn.Conv1d(256, 256, kernel_size = 3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.mp4 = nn.MaxPool1d(3, stride=2)
        
        self.conv5 = nn.Conv1d(256, 256, kernel_size = 3, padding=1)
        self.bn5 = nn.BatchNorm1d(256)
        self.mp5 = nn.MaxPool1d(3, stride=2)

        self.features = nn.Sequential(
                    self.conv1, self.bn1, self.activation, self.mp1,
                    self.conv2, self.bn2, self.activation, self.mp2,
                    self.conv3, self.bn3, self.activation, self.mp3,
                    self.conv4, self.bn4, self.activation, self.mp4,
                    self.conv5, self.bn5, self.activation, self.mp5,
                    )
        
        self.avgpool = nn.AdaptiveAvgPool1d(16)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(4096, 1024) 
        self.fc2 = nn.Linear(1024, 256) 
        self.fc3 = nn.Linear(256, num_classes) 
         
        self.classifier = nn.Sequential(self.dropout, self.fc1, self.activation,
                                        self.dropout, self.fc2, self.activation,
                                        self.fc3) 
    def forward(self, x):
        
        flat_ftrs =  torch.flatten(self.avgpool(self.features(x)),start_dim=1)
        return self.classifier(flat_ftrs)

class cnn_simple2(nn.Module):
    """Simple conv-maxpool cnn"""

    def __init__(self, num_classes):
        """ Defines layers, hardcoded size at the moment."""

        super(cnn_simple2, self).__init__()


        self.activation = nn.LeakyReLU()

        self.conv1 = nn.Conv1d(3, 32, kernel_size = 3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.mp1 = nn.MaxPool1d(3, stride=2)

        self.conv2 = nn.Conv1d(32, 32, kernel_size = 3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.mp2 = nn.MaxPool1d(3, stride=2)

        self.conv3 = nn.Conv1d(32, 32, kernel_size = 3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.mp3 = nn.MaxPool1d(3, stride=2)

        self.conv4 = nn.Conv1d(32, 32, kernel_size = 3, padding=1)
        self.bn4 = nn.BatchNorm1d(32)
        self.mp4 = nn.MaxPool1d(3, stride=2)
        
        self.conv5 = nn.Conv1d(32, 32, kernel_size = 3, padding=1)
        self.bn5 = nn.BatchNorm1d(32)
        self.mp5 = nn.MaxPool1d(3, stride=2)

        self.features = nn.Sequential(
                    self.conv1, self.mp1, self.bn1, self.activation,
                    self.conv2, self.mp2, self.bn2, self.activation, 
                    self.conv3, self.mp3, self.bn3, self.activation,
                    self.conv4, self.mp4, self.bn4, self.activation,
                    self.conv5, self.mp5, self.bn5, self.activation,
                    )
        
        self.avgpool = nn.AdaptiveAvgPool1d(16)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(512, 128) 
        self.fc2 = nn.Linear(128, num_classes) 
         
        self.classifier = nn.Sequential(self.fc1, self.activation,
                                        self.dropout, self.fc2,
                                        ) 
    def forward(self, x):
        
        flat_ftrs =  torch.flatten(self.avgpool(self.features(x)),start_dim=1)
        return self.classifier(flat_ftrs)


class mlp(nn.Module):
    """
    Multi layer perceptron
    """

    def __init__(self, D_in, D_out):
        """
        Defines the layers of the network.

        D_in: lengt of input trace
        D_out: length of output trace
        """

        super(mlp, self).__init__()



        self.model = nn.Sequential(
                     nn.Linear(D_in, 512), nn.Dropout(p=0.5), nn.Sigmoid(),
#                     nn.Linear(512, 512), nn.Dropout(p=0.5), nn.LeakyReLU(),
                     nn.Linear(512, D_out)
                     )


    def forward(self, x):
        """
        Forward pass through the network.

        x: low frequency input
        """
        x = x[:,0,:]
        x = torch.flatten(x, start_dim=1)
        output = self.model(x)
        return output

class lr(nn.Module):
    """
    Multi layer perceptron
    """

    def __init__(self, D_in, D_out):
        """
        Defines the layers of the network.

        D_in: lengt of input trace
        D_out: length of output trace
        """

        super(lr, self).__init__()



        self.model = nn.Sequential(
                     nn.Linear(D_in, D_out)
                     )


    def forward(self, x):
        """
        Forward pass through the network.

        x: low frequency input
        """

        output = self.model(x[:,0,:]).mean(1)
        return output

