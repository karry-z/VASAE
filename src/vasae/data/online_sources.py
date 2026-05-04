import logging
from pathlib import Path

from vasae.data.schema import DataConfig

logger = logging.getLogger(__name__)


def hf_dataset_cache_name(data_cfg: DataConfig) -> str:
    return f"{data_cfg.dataset}_{data_cfg.dataset_config or 'default'}".replace(
        "/",
        "_",
    )


def hf_dataset_cache_dir(cache_root: Path | str, data_cfg: DataConfig) -> Path:
    return Path(cache_root) / ".data_cache" / hf_dataset_cache_name(data_cfg)


def load_hf_text_dataset(
    data_cfg: DataConfig,
    cache_root: Path | str,
):
    from datasets import load_dataset, load_from_disk

    data_cache_dir = hf_dataset_cache_dir(cache_root, data_cfg)
    if (data_cache_dir / "dataset_info.json").exists():
        logger.info(f"Loading cached dataset from {data_cache_dir}")
        return load_from_disk(str(data_cache_dir))

    logger.info(f"Loading dataset {data_cfg.dataset} (first run, will cache)...")
    ds = load_dataset(data_cfg.dataset, data_cfg.dataset_config, split="train")
    if data_cfg.text_column != "text":
        ds = ds.rename_column(data_cfg.text_column, "text")
    ds = ds.filter(lambda x: len(x["text"].strip()) > 50)
    data_cache_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(data_cache_dir))
    logger.info(f"Cached dataset to {data_cache_dir}")
    return ds
