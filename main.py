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
    root = '/mnt/data/datasets/LSA-T/data/cuts'
    max_frames = 75
    batch_size = 128
    keypoints_to_use = [i for i in range(94, 136)]
    words_min_freq = 5
    confidence_threshold = 0.5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_PATH = Path("checkpoints/")
    CHECKPOINT_PATH.mkdir(exist_ok=True)

    print(DEVICE)
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
        words_min_freq = words_min_freq,
        signer_confidence_threshold = confidence_threshold,
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
        words_min_freq = words_min_freq,
        signer_confidence_threshold = confidence_threshold,
        load_videos = False,
        keypoints_transform = keypoints_transform,
        keypoints_transform_each = keypoints_transform_each
    )
    test_dataset.label_transform = label_transform
    
    print("Loading model")
    if not os.listdir(CHECKPOINT_PATH):
        torch.manual_seed(0)
        # adds 2 to max_seq_len for <bos> and <eos> tokens
        model = KeypointModel(max_frames, train_dataset.max_label_len + 2, len(keypoints_to_use), len(train_dataset.vocab)).to(DEVICE)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        checkpoint = None
    else:
        #checkpoint: Optional[ModelCheckpoint] = torch.load(sorted((CHECKPOINT_PATH.glob('*.tar')), reverse=True)[0])
        checkpoint: Optional[ModelCheckpoint] = torch.load(CHECKPOINT_PATH / "checkpoint_20_epochs_5_min_freq_05_conf_threshold.tar")
        model = KeypointModel(max_frames, train_dataset.max_label_len + 2, len(keypoints_to_use), len(train_dataset.vocab)).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    print(checkpoint)
    checkpoint = train(train_dataset, test_dataset, model, 8, batch_size, DEVICE, checkpoint)

    torch.save(checkpoint, CHECKPOINT_PATH / f"checkpoint_{checkpoint['epoch']}_epochs_{words_min_freq}_min_freq_{str(confidence_threshold).replace('.', '')}_conf_threshold.tar")

__main__()