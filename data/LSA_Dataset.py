import json
from pathlib import Path

from typing import Sequence, List, Iterator, Callable, Optional
from type_hints import ClipData, ClipSample, SignerData, KeypointData, T, KEYPOINT_FORMAT

from torch import stack, Tensor
from torchvision.io import VideoReader
from torchvision.datasets import VisionDataset

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from helpers.get_clip_paths import get_clip_paths


def get_bpe_tokenizer(corpus: Sequence[str]) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=['[UNK]', '[SEP]'])
    tokenizer.train_from_iterator(corpus, trainer)
    #tokenizer.post_processor = TemplateProcessing(
    #    single="$0 [SEP]",
    #    special_tokens=[("[SEP]", tokenizer.token_to_id("[SEP]"))],
    #)
    return tokenizer

class LSA_Dataset(VisionDataset):

    def __init__(self,
        root: str,
        load_videos = True,
        load_keypoints = True,
        frame_transform: Optional[Callable[[Tensor], Tensor]] = None,
        video_transform: Optional[Callable[[Sequence[T]], List[T]]] = None,
        keypoints_transform: Optional[Callable[[Sequence[T]], List[T]]] = None,
        keypoints_transform_each: Optional[Callable[[KeypointData], KEYPOINT_FORMAT]] = None
        ) -> None:

        super().__init__(root)

        # samples stores metadata's file path for all samples
        self.samples = [(clip.parent / (clip.name[:-3] + 'json')) for clip in sorted(Path(root).glob('**/*.mp4'), key=lambda p: (str(p.parent), int(str(p.name)[:-4])))]

        corpus: List[str] = []
        for sample in self.samples:
            with sample.open() as data_file:
                data: ClipData = json.load(data_file)
                corpus.append(data['label'])
        self.tokenizer: Tokenizer = get_bpe_tokenizer(corpus)

        self.load_videos = load_videos
        self.load_keypoints = load_keypoints
        self.frame_transform = frame_transform
        self.video_transform = video_transform
        self.keypoints_transform = keypoints_transform
        self.keypoints_transform_each = keypoints_transform_each

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ClipSample:
        paths = get_clip_paths(self.samples[index])
        with paths['json'].open() as data_file:
            data: ClipData = json.load(data_file)
        label = data['label']

        with paths['signer'].open() as signer_file:
            signer: SignerData = json.load(signer_file)
        
        if self.load_keypoints:
            keypoints = self.keypoints_transform(signer['keypoints']) if self.keypoints_transform is not None else signer['keypoints']
            if self.keypoints_transform_each is not None:
                keypoints = map(self.keypoints_transform_each, keypoints)
        else:
            keypoints = None
                
        if self.load_videos:
            clip: List[Tensor] = list(map(lambda frame: frame['data'], VideoReader(str(paths['mp4']), "video")))
            clip = self.video_transform(clip) if self.video_transform is not None else clip
            if self.frame_transform is not None:
                clip = list(map(lambda f: self.frame_transform(f, signer['roi']), clip))
            clip = stack(clip)
        else:
            clip = None

        return (paths, clip, keypoints, self.tokenizer.encode(label))
    
    def __iter__(self) -> Iterator[ClipSample]:
        for i in range(self.__len__()):
            yield self.__getitem__(i)
