import json
from pathlib import Path

from typing import Callable, Optional, Generator, Literal, Union, Iterable, Iterator
from type_hints import ClipData, ClipSample, SignerData, KeypointData, KEYPOINT_FORMAT

from torch import stack, Tensor
from torch.utils.data import Dataset
from torchvision.io import VideoReader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab

from helpers.get_clip_paths import get_clip_paths
from helpers.get_score import get_score
from data.train_test_helpers import split_train_test, load_train_test, store_samples_to_csv


def yield_tokens(samples: Iterable[Path], tokenizer: Callable[[str], list[str]]) -> Generator[list[str], None, None]:
    for sample in samples:
        with sample.open() as data_file:
            data: ClipData = json.load(data_file)
            yield tokenizer(data['label'])

def sample_contains_oov(data_path: Path, vocab: Vocab, tokenizer: Callable[[str], list[str]]) -> bool:
    with open(data_path) as data_file:
        data: ClipData = json.load(data_file)
        return not all(map(vocab.__contains__, tokenizer(data['label'])))

def sample_above_confidence_threshold(data_path: Path, threshold: float) -> bool:
    with open(get_clip_paths(data_path)['signer']) as signer_file:
        line = ""
        while ']' not in line:
            line += signer_file.readline()
        line = line[:-2] + '}'
        signer_data = json.loads(line)
        return get_score(signer_data['scores']) >= threshold


class LSA_Dataset(Dataset):

    def __init__(self,
            root: str,
            mode: Literal["train", "test"],
            load_videos: bool = True,
            load_keypoints: bool = True,
            words_min_freq: int = 1,
            signer_confidence_threshold: float = .5,
            max_label_len: Optional[int] = None,
            label_as_idx: bool = True,
            frame_transform: Optional[Callable[[Tensor], Tensor]] = None,
            video_transform: Optional[Callable[[list[Tensor]], list[Tensor]]] = None,
            keypoints_transform: Optional[Callable[[list[KeypointData]], list[KeypointData]]] = None,
            keypoints_transform_each: Optional[Callable[[KeypointData], KEYPOINT_FORMAT]] = None,
            label_transform: Optional[Callable[[list[int]], Tensor]] = None
        ) -> None:

        self.root = Path(root)
        self.mode = mode
        self.load_videos = load_videos
        self.load_keypoints = load_keypoints
        self.words_min_freq = words_min_freq
        self.signer_confidence_threshold = signer_confidence_threshold
        self.label_as_idx = label_as_idx
        self.frame_transform = frame_transform
        self.video_transform = video_transform
        self.keypoints_transform = keypoints_transform
        self.keypoints_transform_each = keypoints_transform_each
        self.label_transform = label_transform

        train_path = self.root / f"train_min_freq_{words_min_freq}_threshold_{str(signer_confidence_threshold).replace('.','')}.csv"
        test_path = self.root / f"test_min_freq_{words_min_freq}_threshold_{str(signer_confidence_threshold).replace('.','')}.csv"
        video_paths = map(lambda p: Path(str(p.resolve())[:-3] + "json"), self.root.glob('**/*.mp4'))
        
        special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.tokenizer: Callable[[str], list[str]] = get_tokenizer('spacy', language='es_core_news_lg')
        self.vocab = build_vocab_from_iterator(yield_tokens(video_paths, self.tokenizer),
                                                            min_freq = words_min_freq,
                                                            specials = special_symbols,
                                                            special_first = True)
        # by default returns <unk> index
        self.vocab.set_default_index(0)
        
        if train_path.exists() and test_path.exists():
            self.train_samples, self.test_samples = load_train_test(train_path, test_path)
        else:
            self.train_samples, self.test_samples = split_train_test(self.root, lambda path:
                (not sample_contains_oov(path, self.vocab, self.tokenizer))
                and (sample_above_confidence_threshold(path, self.signer_confidence_threshold) if self.signer_confidence_threshold != 0 else True))
            store_samples_to_csv(train_path, self.train_samples)
            store_samples_to_csv(test_path, self.test_samples)
        self.max_label_len =  max_label_len if max_label_len is not None else max(map(len, yield_tokens(self.train_samples + self.test_samples, self.tokenizer)))

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
