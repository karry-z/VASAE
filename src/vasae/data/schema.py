from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class DataConfig:
    # Dataset source (online flow)
    dataset: str = "wikitext"
    dataset_config: str | None = "wikitext-103-raw-v1"
    text_column: str = "text"
    max_length: int = 128

    # Batch sizes (shared)
    train_batchsize: int = 32
    valid_batchsize: int = 32
    test_batchsize: int = 32

    # Offline flow (deprecated)
    data_dir: str | Path | None = None
    layer_name: str = ""
    use_centralize: bool = True

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
