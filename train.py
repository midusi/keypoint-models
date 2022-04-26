from torch.utils.data import DataLoader

from data.LSA_Dataset import LSA_Dataset
from data.transforms import get_frames_reduction_transform


def train():
    root = '/mnt/data/datasets/cn_sordos_db/data/cuts'
    load_videos = False
    load_keypoints = True
    max_frames = 75
    batch_size = 64

    dataset = LSA_Dataset(
        root,
        load_videos = load_videos,
        load_keypoints = load_keypoints,
        keypoints_transform = get_frames_reduction_transform(max_frames)
        )
    
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)