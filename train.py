import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


from data.LSA_Dataset import LSA_Dataset
from data.transforms import get_frames_reduction_transform, get_keypoint_format_transform, get_text_to_tensor_transform
from model.KeypointModel import KeypointModel


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, pad_idx, device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def get_collate_fn(vocab, tokenizer, text_to_tensor, pad_idx):
    # function to collate data samples into batch tensors
    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(src_sample)
            tgt_batch.append(text_to_tensor(vocab(tokenizer(tgt_sample))))
        src_batch = pad_sequence(src_batch, padding_value=pad_idx)
        tgt_batch = pad_sequence(tgt_batch, padding_value=pad_idx)
        return src_batch, tgt_batch
    return collate_fn

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
    
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    torch.manual_seed(0)

    transformer = KeypointModel(max_frames, dataset.max_seq_len, len(keypoints_to_use), len(dataset.vocab)).to(DEVICE)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=dataset.vocab.__getitem__("<pad>"))

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

