import json
from pathlib import Path

from typing import Callable, Optional, Generator, Literal, Union, Iterator
from type_hints import ClipData, ClipSample, SignerData, KeypointData, KEYPOINT_FORMAT

from torch import stack, Tensor
from torch.utils.data import Dataset
from torchvision.io import VideoReader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from helpers.get_clip_paths import get_clip_paths
from data.train_test_helpers import split_train_test, load_train_test, store_samples_to_csv


def yield_tokens(samples: list[Path], tokenizer: Callable[[str], list[str]]) -> Generator[list[str], None, None]:
    for sample in samples:
        with sample.open() as data_file:
            data: ClipData = json.load(data_file)
            yield tokenizer(data['label'])

class LSA_Dataset(Dataset):

    def __init__(self,
            root: str,
            mode: Literal["train", "test"],
            load_videos: bool = True,
            load_keypoints: bool = True,
            use_only_res: bool = False,
            label_as_idx: bool = True,
            frame_transform: Optional[Callable[[Tensor], Tensor]] = None,
            video_transform: Optional[Callable[[list[Tensor]], list[Tensor]]] = None,
            keypoints_transform: Optional[Callable[[list[KeypointData]], list[KeypointData]]] = None,
            keypoints_transform_each: Optional[Callable[[KeypointData], KEYPOINT_FORMAT]] = None,
            label_transform: Optional[Callable[[list[int]], Tensor]] = None
        ) -> None:

        self.mode = mode
        train_path = Path(root) / ("train.csv" if not use_only_res else "train_res.csv")
        test_path = Path(root) / ("test.csv" if not use_only_res else "test_res.csv")
        if train_path.exists() and test_path.exists():
            self.train_samples, self.test_samples = load_train_test(train_path, test_path)
        else:
            self.train_samples, self.test_samples = split_train_test(Path(root), use_only_res)
            store_samples_to_csv(train_path, self.train_samples)
            store_samples_to_csv(test_path, self.test_samples)
        self.tokenizer: Callable[[str], list[str]] = get_tokenizer('spacy', language='es_core_news_lg')
        special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.vocab = build_vocab_from_iterator(yield_tokens(self.train_samples, self.tokenizer),
                                                min_freq=1,
                                                specials=special_symbols,
                                                special_first=True)
        # by default returns <unk> index
        self.vocab.set_default_index(0)
        self.max_tgt_len = max(map(len, yield_tokens(self.train_samples + self.test_samples, self.tokenizer)))
        self.load_videos = load_videos
        self.load_keypoints = load_keypoints
        self.label_as_idx = label_as_idx
        self.frame_transform = frame_transform
        self.video_transform = video_transform
        self.keypoints_transform = keypoints_transform
        self.keypoints_transform_each = keypoints_transform_each
        self.label_transform = label_transform

    def __len__(self) -> int:
        return len(self.train_samples if self.mode == "train" else self.test_samples)

    def __getitem__(self, index: int) -> ClipSample:
        paths = get_clip_paths((self.train_samples if self.mode == "train" else self.test_samples)[index])
        with paths['json'].open() as data_file:
            data: ClipData = json.load(data_file)
        # label stores a list of the token indices for the corresponding label
        label: Union[list[str], list[int]] = self.tokenizer(data['label'])
        if self.label_as_idx:
            label = self.vocab(label)

        with paths['signer'].open() as signer_file:
            signer: SignerData = json.load(signer_file)
        
        if self.load_keypoints:
            if self.keypoints_transform_each is not None:
                keypoints = list(map(self.keypoints_transform_each, signer['keypoints']))
            keypoints = self.keypoints_transform(keypoints) if self.keypoints_transform is not None else keypoints
                
        if self.load_videos:
            clip: list[Tensor] = list(map(lambda frame: frame['data'], VideoReader(str(paths['mp4']), "video")))
            clip = self.video_transform(clip) if self.video_transform is not None else clip
            if self.frame_transform is not None:
                clip = list(map(lambda f: self.frame_transform(f, signer['roi']), clip))
            out_clip = stack(clip)

        return (
            out_clip if self.load_videos else None,
            keypoints if self.load_keypoints else None,
            label if self.label_transform is None else self.label_transform(label))
    
    def __iter__(self) -> Iterator[ClipSample]:
        for i in range(self.__len__()):
            yield self.__getitem__(i)
    
    def get_token_idx(self, token: str) -> int:
        return self.vocab.__getitem__(token)
