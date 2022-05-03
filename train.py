from pathlib import Path
from timeit import default_timer as timer

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from data.LSA_Dataset import LSA_Dataset
from data.transforms import (
    get_frames_reduction_transform,
    get_keypoint_format_transform,
    get_text_to_tensor_transform,
    keypoint_norm_to_center_transform
)
from model.KeypointModel import KeypointModel
from helpers.create_mask import create_mask
from data.collate_fn import get_keypoint_model_collate_fn


def train_epoch(model, optimizer, dataset, batch_size, collate_fn, loss_fn, device):
    model.train()
    losses = 0
    train_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

    for i, (src, tgt) in enumerate(train_dataloader):
        print(i, len(train_dataloader))
        src = [[frame.to(device) for frame in each] for each in src]
        #src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, dataset.vocab.__getitem__("<pad>"), device)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)

def evaluate(model, dataset, batch_size, collate_fn, loss_fn, device):
    model.eval()
    losses = 0

    val_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

    for src, tgt in val_dataloader:
        src = [[frame.to(device) for frame in each] for each in src]
        #src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, dataset.vocab.__getitem__("<pad>"), device)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)

def train():
    root = '/mnt/data/datasets/cn_sordos_db/data/cuts'
    load_videos = False
    load_keypoints = True
    max_frames = 75
    batch_size = 128
    keypoints_to_use = [i for i in range(94, 136)]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_PATH = Path("checkpoints/")
    CHECKPOINT_PATH.mkdir(exist_ok=True)

    print("Loading train dataset")
    dataset = LSA_Dataset(
        root,
        mode="train",
        load_videos = load_videos,
        load_keypoints = load_keypoints,
        keypoints_transform = Compose([
            keypoint_norm_to_center_transform,
            get_frames_reduction_transform(max_frames)
        ]),
        keypoints_transform_each = get_keypoint_format_transform(keypoints_to_use)
        )
    dataset.label_transform = get_text_to_tensor_transform(dataset.vocab.__getitem__("<bos>"), dataset.vocab.__getitem__("<eos>"))
    
    print("Loading test dataset")
    dataset_test = LSA_Dataset(
        root,
        mode="test",
        load_videos = load_videos,
        load_keypoints = load_keypoints,
        keypoints_transform = Compose([
            keypoint_norm_to_center_transform,
            get_frames_reduction_transform(max_frames)
        ]),
        keypoints_transform_each = get_keypoint_format_transform(keypoints_to_use)
        )
    dataset_test.label_transform = get_text_to_tensor_transform(dataset_test.vocab.__getitem__("<bos>"), dataset_test.vocab.__getitem__("<eos>"))
    
    print("Loading model")
    torch.manual_seed(0)
    # adds 2 to max_seq_len for <bos> and <eos> tokens
    model = KeypointModel(max_frames, max(dataset.max_seq_len, dataset_test.max_seq_len) + 2, len(keypoints_to_use), len(dataset.vocab)).to(DEVICE)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=dataset.vocab.__getitem__("<pad>"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    collate_fn = get_keypoint_model_collate_fn(dataset.vocab.__getitem__("<pad>"))

    NUM_EPOCHS = 10
    train_loss_hist, val_loss_hist = [], []
    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        train_loss = train_epoch(model, optimizer, dataset, batch_size, collate_fn, loss_fn, DEVICE)
        train_loss_hist.append(train_loss)
        end_time = timer()
        val_loss = evaluate(model, dataset_test, batch_size, collate_fn, loss_fn, DEVICE)
        val_loss_hist.append(val_loss)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_loss_hist': train_loss_hist,
            'val_loss_hist': val_loss_hist
            }, CHECKPOINT_PATH)

train()