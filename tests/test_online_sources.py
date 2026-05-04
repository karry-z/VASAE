from types import SimpleNamespace

from vasae.data.online_sources import (
    hf_dataset_cache_dir,
    hf_dataset_cache_name,
    load_hf_text_dataset,
)
from vasae.data.schema import DataConfig


class FakeDataset:
    column_names = ["body"]

    def __init__(self):
        self.saved_to = None
        self.renamed = None

    def rename_column(self, old, new):
        self.renamed = (old, new)
        return self

    def filter(self, fn):
        assert fn({"text": "x" * 80})
        return self

    def save_to_disk(self, path):
        self.saved_to = path


def test_hf_cache_name_and_dir():
    data_cfg = DataConfig(dataset="org/name", dataset_config=None)

    assert hf_dataset_cache_name(data_cfg) == "org_name_default"
    assert hf_dataset_cache_dir("/tmp/cache", data_cfg).as_posix() == (
        "/tmp/cache/.data_cache/org_name_default"
    )


def test_load_hf_text_dataset_loads_existing_cache(monkeypatch, tmp_path):
    data_cfg = DataConfig(dataset="wikitext", dataset_config="raw")
    cache_dir = hf_dataset_cache_dir(tmp_path, data_cfg)
    cache_dir.mkdir(parents=True)
    (cache_dir / "dataset_info.json").write_text("{}")
    cached = object()

    monkeypatch.setitem(
        __import__("sys").modules,
        "datasets",
        SimpleNamespace(
            load_from_disk=lambda path: cached,
            load_dataset=lambda *args, **kwargs: None,
        ),
    )

    assert load_hf_text_dataset(data_cfg, tmp_path) is cached


def test_load_hf_text_dataset_loads_filters_and_saves(monkeypatch, tmp_path):
    dataset = FakeDataset()
    data_cfg = DataConfig(
        dataset="wikitext",
        dataset_config="raw",
        text_column="body",
    )

    monkeypatch.setitem(
        __import__("sys").modules,
        "datasets",
        SimpleNamespace(
            load_from_disk=lambda path: None,
            load_dataset=lambda dataset_name, dataset_config, split: dataset,
        ),
    )

    assert load_hf_text_dataset(data_cfg, tmp_path) is dataset
    assert dataset.renamed == ("body", "text")
    assert dataset.saved_to == str(hf_dataset_cache_dir(tmp_path, data_cfg))
