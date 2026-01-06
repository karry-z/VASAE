from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class LayerMeta:
    path: Path
    shape: List[int]
    dtype: str
    mean: Path


Meta = Dict[str, LayerMeta]
