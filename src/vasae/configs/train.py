from dataclasses import dataclass


@dataclass
class TrainConfig:
    # loop control
    num_epochs: int
    max_batchsize: int

    # optim
    lr: float
