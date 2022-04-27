from typing import List
from torch import Tensor, nn, stack
from torch.nn.functional import relu
import torch

from model.PositionalEncoding import PositionalEncoding


class KeypointModel(nn.Module):

    def __init__(self, inp_max_len: int, tgt_max_len: int):
        super(KeypointModel, self).__init__()

        # in_features is the result of flattening the input of (x,y,c).(k1, ..., k42)
        self.fc = nn.Linear(in_features=126, out_features=64)
        kernel_size = 5
        self.conv1d = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=kernel_size)
        self.pe = PositionalEncoding(emb_size=32, max_len=(inp_max_len - kernel_size + 1))
        self.transformer = nn.Transformer()
        

    def forward(self, inp: List[Tensor], tgt: Tensor):
        # flatten and apply fc frame by frame, then stack the frames and permute dims for conv
        x = stack([relu(self.fc(torch.flatten(frame))) for frame in inp]).permute(1,0)
        # unsqueeze adds dimention representing the batch CHEQUEAR
        x = self.conv1d(x.unsqueeze(0))[0].permute(1,0)
        # unsqueeze adds dimention representing the batch CHEQUEAR
        x = x.unsqueeze(1)
        x = self.pe(x)
        #x = self.transformer(x)

        return x