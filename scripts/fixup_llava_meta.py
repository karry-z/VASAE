"""
Fix meta.json from the initial collection run (job 2680252) which used
absolute paths, no 'mean' key, and flat directory layout.

Moves .dat files into data/ subdir, updates meta.json with relative paths
and mean paths, then computes and saves means.
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np


def main():
    out_dir = Path(
        "/scratch/b5bq/pu22650.b5bq/activations_llava-hf_llava-1.5-7b-hf_lmms-lab_COCO-Caption"
    )
    meta_path = out_dir / "meta.json"

    with open(meta_path) as f:
        meta = json.load(f)

    # Create subdirs
    data_dir = out_dir / "data"
    mean_dir = out_dir / "mean"
    data_dir.mkdir(exist_ok=True)
    mean_dir.mkdir(exist_ok=True)

    new_meta = {}
    for layer, info in meta.items():
        old_path = Path(info["path"])
        fname = old_path.name
        new_dat_path = data_dir / fname

        # Move .dat file if it's still in the flat dir
        if old_path.exists() and not new_dat_path.exists():
            print(f"Moving {old_path} -> {new_dat_path}")
            shutil.move(str(old_path), str(new_dat_path))
        elif not new_dat_path.exists():
            # Maybe path is already relative
            flat_path = out_dir / fname
            if flat_path.exists():
                print(f"Moving {flat_path} -> {new_dat_path}")
                shutil.move(str(flat_path), str(new_dat_path))

        shape = tuple(info["shape"])
        dtype = info["dtype"]

        # Compute mean
        mm = np.memmap(str(new_dat_path), mode="r", dtype=dtype, shape=shape)
        mean = mm.mean(axis=0, dtype=np.float64).astype(np.float32)
        mean_fname = f"{layer.replace('.', '_')}_mean.npy"
        mean_path = mean_dir / mean_fname
        np.save(mean_path, mean)
        mean_path.chmod(0o444)
        print(f"Saved mean for {layer}: {mean_path}")

        new_meta[layer] = {
            "path": f"data/{fname}",
            "shape": list(shape),
            "dtype": dtype,
            "mean": f"mean/{mean_fname}",
        }

    # Backup old meta and write new
    shutil.copy(str(meta_path), str(meta_path) + ".old")
    with open(meta_path, "w") as f:
        json.dump(new_meta, f, indent=2)
    print(f"Updated {meta_path}")


if __name__ == "__main__":
    main()
