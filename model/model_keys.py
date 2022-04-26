import torch
from torch import nn


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        self.fc = nn.Linear()
        

    def forward(self, x):

