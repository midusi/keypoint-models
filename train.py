from timeit import default_timer as timer

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from data.LSA_Dataset import LSA_Dataset
from data.transforms import get_frames_reduction_transform, get_keypoint_format_transform, get_text_to_tensor_transform
from model.KeypointModel import KeypointModel
from helpers.create_mask import create_mask


def get_collate_fn(pad_idx):
    # function to collate data samples into batch tensors
    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for clip, keypoints, label in batch:
            src_batch.append(clip if keypoints is None else (keypoints if clip is None else (clip, keypoints)))
            tgt_batch.append(label)
        tgt_batch = pad_sequence(tgt_batch, padding_value=pad_idx)
        return src_batch, tgt_batch
    return collate_fn

def train_epoch(model, optimizer, dataset, batch_size, collate_fn, loss_fn, device):
    model.train()
    losses = 0
    train_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

    for src, tgt in train_dataloader:
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
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

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
    batch_size = 64
    keypoints_to_use = [i for i in range(94, 136)]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = LSA_Dataset(
        root,
        load_videos = load_videos,
        load_keypoints = load_keypoints,
        keypoints_transform = get_frames_reduction_transform(max_frames),
        keypoints_transform_each = get_keypoint_format_transform(keypoints_to_use)
        )
    dataset.label_transform = get_text_to_tensor_transform(dataset.vocab.__getitem__("<bos>"), dataset.vocab.__getitem__("<eos>"))
    
    torch.manual_seed(0)

    model = KeypointModel(max_frames, dataset.max_seq_len, len(keypoints_to_use), len(dataset.vocab)).to(DEVICE)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=dataset.vocab.__getitem__("<pad>"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    collate_fn = get_collate_fn(dataset.vocab.__getitem__("<pad>"))

    train_epoch(model, optimizer, dataset, batch_size, collate_fn, loss_fn, DEVICE)

    NUM_EPOCHS = 2

    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        train_loss = train_epoch(model, optimizer, dataset, batch_size, collate_fn, loss_fn, DEVICE)
        end_time = timer()
        val_loss = evaluate(model)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

train()