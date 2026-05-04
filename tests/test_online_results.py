import json
from pathlib import Path

from scripts.eval.eval_sae_online import (
    infer_data_source,
    jsonable as eval_jsonable,
    parse_layer_from_dirname,
    read_training_config,
    read_training_split_config,
    save_eval_results,
)
from scripts.training.train_sae_online import (
    jsonable as train_jsonable,
    save_training_results,
)


def test_script_jsonable_converts_paths_and_sequences():
    payload = {
        "path": Path("/tmp/run"),
        "items": (Path("/tmp/a"), [Path("/tmp/b")]),
    }

    expected = {
        "path": "/tmp/run",
        "items": ["/tmp/a", ["/tmp/b"]],
    }
    assert train_jsonable(payload) == expected
    assert eval_jsonable(payload) == expected


def test_save_training_results_preserves_schema(tmp_path):
    results_path = tmp_path / "results.json"

    save_training_results(
        results_path,
        config={"corpus_dir": tmp_path},
        stopped_epoch=3,
        eval_out={"loss": 1, "variance_explained": 0.25, "label": "ok"},
    )

    saved = json.loads(results_path.read_text())
    assert saved == {
        "config": {"corpus_dir": str(tmp_path)},
        "stopped_epoch": 3,
        "last_eval": {
            "loss": 1.0,
            "variance_explained": 0.25,
            "label": "ok",
        },
    }


def test_save_eval_results_preserves_schema(tmp_path):
    results_path = tmp_path / "results_eval.json"

    save_eval_results(
        results_path,
        config={"corpus_dir": tmp_path},
        test_out={"loss": 2, "dead_rate": 0.1, "l0": 8, "alive_features": [1, 2]},
    )

    saved = json.loads(results_path.read_text())
    assert saved["config"] == {"corpus_dir": str(tmp_path)}
    assert saved["test"]["loss"] == 2.0
    assert saved["test"]["alive_features"] == [1, 2]
    assert saved["dead_rate"] == 0.1
    assert saved["l0"] == 8


def test_read_training_config_and_split_sizes(tmp_path):
    (tmp_path / "results.json").write_text(
        json.dumps(
            {
                "config": {
                    "data_source": "jsonl",
                    "train_samples": "10",
                    "eval_samples": 5,
                    "test_samples": 2,
                }
            }
        )
    )

    assert read_training_config(tmp_path)["data_source"] == "jsonl"
    assert read_training_split_config(tmp_path) == (10, 5, 2)


def test_infer_data_source_and_parse_layer():
    assert infer_data_source("hf", {"data_source": "jsonl"}) == "hf"
    assert infer_data_source("auto", {"data_source": "jsonl"}) == "jsonl"
    assert parse_layer_from_dirname("009_online_gpt2_L11_k32_a0") == 11

