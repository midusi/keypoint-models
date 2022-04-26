import torch
from torch import nn


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        self.fc = nn.Linear()
        

    def forward(self, x):
        

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = KeypointModel().to(device)
print(model)