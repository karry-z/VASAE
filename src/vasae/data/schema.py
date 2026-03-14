from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class DataConfig:
    train_batchsize: int = 32
    valid_batchsize: int = 32
    test_batchsize: int = 32
    use_centralize: bool = True
    meta_path: str = ""  # deprecated
    layer_name: str = ""
    data_dir: str | Path | None = None

    def __post_init__(self):
        if self.data_dir is not None:
            self.data_dir = Path(self.data_dir)


@dataclass(frozen=True)
class LayerMeta:
    path: Path
    shape: List[int]
    dtype: str
    mean: Path


Meta = Dict[str, LayerMeta]
