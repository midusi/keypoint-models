from typing import Callable, List
import torch
from torch import Tensor

from torch.nn.utils.rnn import pad_sequence
from type_hints import ClipSample, KeypointModelSample


def get_keypoint_model_collate_fn(pad_idx: int) -> Callable[[List[ClipSample]], KeypointModelSample]:
    # adds padding to target and normalizes keypoints
    def collate_fn(batch: List[ClipSample]) -> KeypointModelSample:
        xs, ys = Tensor(), Tensor()
        for _, keypoints, _ in batch:
            for frame in keypoints:
                torch.cat([xs, frame[0]])
                torch.cat([ys, frame[1]])
        keypoint_mean = Tensor([[torch.mean(xs)],[torch.mean(ys)],[0]])
        keypoint_std = Tensor([[torch.std(xs)],[torch.std(ys)],[1]])
        src_batch, tgt_batch = [], []
        for clip, keypoints, label in batch:
            src_batch.append([(frame-keypoint_mean)/keypoint_std for frame in keypoints])
            tgt_batch.append(label)
        tgt_batch = pad_sequence(tgt_batch, padding_value=pad_idx)
        return src_batch, tgt_batch
    return collate_fn