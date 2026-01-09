from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    train_batchsize: int = 32
    valid_batchsize: int = 32
    test_batchsize: int = 32
    use_centralize: bool = True
    meta_path: str = ""
    layer_name: str = ""
    data_dir: Path = None
