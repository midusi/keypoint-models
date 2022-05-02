from typing import Callable, List

from torch.nn.utils.rnn import pad_sequence
from type_hints import ClipSample, KeypointModelSample


def get_keypoint_model_collate_fn(pad_idx: int) -> Callable[[List[ClipSample]], KeypointModelSample]:
    # function to collate data samples into batch tensors
    def collate_fn(batch: List[ClipSample]) -> KeypointModelSample:
        src_batch, tgt_batch = [], []
        for clip, keypoints, label in batch:
            src_batch.append(keypoints)
            tgt_batch.append(label)
        tgt_batch = pad_sequence(tgt_batch, padding_value=pad_idx)
        return src_batch, tgt_batch
    return collate_fn