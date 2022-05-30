from typing import Optional, Sequence, Tuple, TypedDict, TypeVar, Union, OrderedDict
from torch import Tensor


T = TypeVar('T')

KEYPOINT_FORMAT = TypeVar('KEYPOINT_FORMAT')

class ClipData(TypedDict):
    label: str
    start: float
    end: float
    video: float

class KeypointData(TypedDict):
    image_id: str
    category_id: int
    keypoints: list[float]
    score: float
    box: list[float]
    idx: list[float]

class Box(TypedDict):
    x1: float
    y1: float
    width: float
    height: float

class SignerData(TypedDict):
    scores: list[float]
    roi: Box
    keypoints: list[KeypointData]

class ModelCheckpoint(TypedDict):
    epoch: int
    model_state_dict: OrderedDict[str, Tensor]
    optimizer_state_dict: dict
    train_loss: float
    val_loss: float
    train_loss_hist: list[float]
    val_loss_hist: list[float]

ClipSample = Tuple[
    Optional[Tensor],
    Optional[Sequence[KEYPOINT_FORMAT]],
    Union[list[int], Tensor]]

KeypointModelSample = Tuple[
    list[Tensor],
    Tensor
]
