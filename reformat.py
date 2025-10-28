import os
import glob
import pickle
import torch
import numpy as np
from time import time
import logging
import pickle, glob, os

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s %(message)s', # The main format string
    level=logging.INFO, # Sets the minimum severity to be logged
    datefmt='%Y-%m-%d %H:%M:%S' # Sets the time format
)

# --- Configuration ---
SRC_DIR = "/mnt/data/activation_cache"        # folder with activations_*.pkl
OUT_DIR = "/mnt/data/output_npy"     # folder for .npy outputs
NUM_SAMPLES_PER_FILE = 100
os.makedirs(OUT_DIR, exist_ok=True)
logging.info(f"START")

# --- Step 1: scan pickle files ---
pkl_files = sorted(glob.glob(os.path.join(SRC_DIR, "activations_*.pkl")))
logging.info(f"Found {len(pkl_files)} pickle files in {SRC_DIR}")

# --- Step 2: infer layer names and shapes ---
logging.info(f"Reading structure from: {pkl_files[0]}")
with open(pkl_files[0], "rb") as f:
    first_batch = pickle.load(f)
first_sample = first_batch[0]
layer_shapes = {k: v.shape for k, v in first_sample.items()}
layer_dtypes = {k: np.float32 for k in first_sample.keys()}
n_layers = len(layer_shapes)
logging.info(f"Detected {n_layers} layers:")
for k, v in layer_shapes.items():
    logging.info(f"   - {k}: shape={v}")

# --- Step 3: count total examples ---
logging.info(f"Counting total examples...")
n_total = 0
for pkl_path in pkl_files:
    logging.info(f"load: {pkl_path}")
    with open(pkl_path, "rb") as f:
        batch = pickle.load(f)
    n_total += len(batch)
logging.info(f"Total examples: {n_total}")

# --- Step 4: create memmaps ---
logging.info(f"Creating memmap files...")
memmaps = {}
for layer, shape in layer_shapes.items():
    out_path = os.path.join(OUT_DIR, f"{layer.replace('.', '_')}.npy")
    full_shape = (n_total,) + shape[1:]  # remove leading batch dim
    logging.info(f"   - {layer} → {out_path}  shape={full_shape}")
    memmaps[layer] = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=layer_dtypes[layer], shape=full_shape
    )

# --- Step 5: write data incrementally ---
logging.info(f"Writing data to memmaps...")
idx = 0
start_time = time()
for file_idx, pkl_path in enumerate(pkl_files):
    t0 = time()
    with open(pkl_path, "rb") as f:
        batch = pickle.load(f)

    for i, data in enumerate(batch):
        for layer, tensor in data.items():
            logging.info(f"write batch {i}, layer {layer} ")
            arr = tensor.squeeze(0).numpy()
            memmaps[layer][idx] = arr
        idx += 1

    logging.info(f"{file_idx+1}/{len(pkl_files)}: {pkl_path} processed, "
          f"{idx}/{n_total} samples total, time={time()-t0:.2f}s")

# --- Step 6: flush and verify ---
for layer, m in memmaps.items():
    m.flush()
logging.info(f"Finished writing all data in {time()-start_time:.2f}s")
logging.info(f"Output directory: {OUT_DIR}")
