from math import ceil

from typing import Sequence, List, Callable
from type_hints import Box, T, KeypointData

import torch
from torch import Tensor
from torchvision.transforms.functional import crop, resize


def get_roi_selector_transform(height: int, width: int) -> Callable[[Tensor], Tensor]:
    '''Given height and width, returns a frame-level transform that crops a given roi from the frame and resizes it to to the desired values keeping the aspect ratio and padding with zeros if necessary'''
    def roi_selector_transform(img: Tensor, box: Box) -> Tensor:
        img = crop(img, int(box['y1']),int(box['x1']),int(box['height']),int(box['width']))
        pad = torch.zeros(3, height, width, dtype=torch.uint8)
        if (box['height'] - height) > (box['width'] - width):
            new_width = int(box['width']*height/box['height'])
            img = resize(img, [height, new_width])
            pad[:, :, int((width - new_width)/2):-int((width - new_width)/2) - (1 if (width - new_width) % 2 == 1 else 0)] = img
        else:
            new_height = int(box['height']*width/box['width'])
            img = resize(img, [new_height, width])
            pad[:, int((height - new_height)/2):-int((height - new_height)/2) - (1 if (height - new_height) % 2 == 1 else 0), :] = img
        return pad
    return roi_selector_transform

def get_frames_reduction_transform(max_frames: int) -> Callable[[Sequence[T]], List[T]]:
    '''Given the desired frame amount, returns a transform that reduces amount of frames of sequence to max_frames'''
    def frames_reduction_transform(clip: Sequence[T]) -> List[T]:
        frames = []
        for frame in [c for (i,c) in enumerate(clip) if (i%(ceil(len(clip)/max_frames)) == 0)]:
            frames.append(frame)
        if len(frames) < max_frames:
            for _ in range(max_frames - len(frames)):
                frames.append(frames[-1])
        return frames
    return frames_reduction_transform

def keypoint_format_transform(keypoint_data: KeypointData) -> Tensor:
    return Tensor([[
        k for k in keypoint_data['keypoints'] if (k%3) == i
    ] for i in range(3)])
