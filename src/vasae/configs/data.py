from dataclasses import dataclass


@dataclass
class DataConfig:
    train_batchsize: int
    valid_batchsize: int
    test_batchsize: int
    use_centralize: bool
    meta_path: str = ""
    layer_name: str = ""
