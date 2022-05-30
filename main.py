import os
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torchvision.transforms import Compose
import wandb

from data.LSA_Dataset import LSA_Dataset
from data.transforms import (
    get_frames_reduction_transform,
    get_keypoint_format_transform,
    get_text_to_tensor_transform,
    keypoint_norm_to_center_transform,
    interpolate_keypoints,
    keypoints_norm_to_nose
)
from model.KeypointModel import KeypointModel
from train import train
from type_hints import ModelCheckpoint


def __main__():
    root = '/mnt/data/datasets/cn_sordos_db/data/cuts'
    max_frames = 75
    batch_size = 128
    keypoints_to_use = [i for i in range(94, 136)]
    use_only_res = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_PATH = Path("checkpoints/" if not use_only_res else "checkpoints/res/")
    CHECKPOINT_PATH.mkdir(exist_ok=True)

    #wandb.init(project="all-db-train", entity="pedroodb")

    keypoints_transform = Compose([
        get_frames_reduction_transform(max_frames),
        interpolate_keypoints
    ])
    keypoints_transform_each = Compose([
        get_keypoint_format_transform(keypoints_to_use),
        keypoints_norm_to_nose
    ])

    print("Loading train dataset")
    train_dataset = LSA_Dataset(
        root,
        mode = "train",
        use_only_res = use_only_res,
        load_videos = False,
        keypoints_transform = keypoints_transform,
        keypoints_transform_each = keypoints_transform_each
    )
    label_transform = get_text_to_tensor_transform(train_dataset.get_token_idx("<bos>"), train_dataset.get_token_idx("<eos>"))
    train_dataset.label_transform = label_transform
    
    print("Loading test dataset")
    test_dataset = LSA_Dataset(
        root,
        mode="test",
        use_only_res = use_only_res,
        load_videos = False,
        keypoints_transform = keypoints_transform,
        keypoints_transform_each = keypoints_transform_each
    )
    test_dataset.label_transform = label_transform
    
    print("Loading model")
    if not os.listdir(CHECKPOINT_PATH):
        torch.manual_seed(0)
        # adds 2 to max_seq_len for <bos> and <eos> tokens
        model = KeypointModel(max_frames, train_dataset.max_tgt_len + 2, len(keypoints_to_use), len(train_dataset.vocab)).to(DEVICE)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        checkpoint = None
    else:
        checkpoint: Optional[ModelCheckpoint] = torch.load(sorted((CHECKPOINT_PATH.glob('*.tar')), reverse=True)[0])
        model = KeypointModel(max_frames, train_dataset.max_tgt_len + 2, len(keypoints_to_use), len(train_dataset.vocab)).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])

    checkpoint = train(train_dataset, test_dataset, model, 10, batch_size, DEVICE, checkpoint)

    torch.save(checkpoint, CHECKPOINT_PATH / f"checkpoint_{checkpoint['epoch']}_epochs.tar")

__main__()