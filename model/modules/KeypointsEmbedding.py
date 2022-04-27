from typing import List
from torch import Tensor, nn, stack
from torch.nn.functional import relu
import torch


class KeypointsEmbedding(nn.Module):

    def __init__(self,
                keys_amount: int,
                kernel_size: int = 5,
                emb_size: int = 64,
                keys_initial_emb_size: int = 128,
                ):
        super(KeypointsEmbedding, self).__init__()

        # in_features is the result of flattening the input of (x,y,c).(k1, ..., k42)
        self.fc = nn.Linear(in_features=keys_amount*3, out_features=keys_initial_emb_size)
        self.conv1d = nn.Conv1d(in_channels=keys_initial_emb_size, out_channels=emb_size, kernel_size=kernel_size)
        

    def forward(self, src: List[Tensor]):
        # flatten and apply fc frame by frame, then stack the frames and permute dims for conv
        src_emb = stack([relu(self.fc(torch.flatten(frame))) for frame in src]).permute(1,0)
        # unsqueeze adds dimention representing the batch CHEQUEAR
        src_emb = self.conv1d(src_emb.unsqueeze(0))[0].permute(1,0)
        # unsqueeze adds dimention representing the batch CHEQUEAR
        src_emb = src_emb.unsqueeze(1)
        return src_emb
