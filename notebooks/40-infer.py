"""Phase 4 inference script — runs inside the self-contained Kaggle notebook.

This file is the PAYLOAD of notebooks/40-infer.ipynb.
The builder (build_infer_ipynb.py) embeds this verbatim.
"""

import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, "/kaggle/working")

from knee.config import get as cfg
from knee.preprocess import (
    IMG_SIZE, MAX_SERIES, NUM_SLICES,
    find_data_dir, process_study, select_series,
)
from knee.dataset import prep_tensor
from knee.model import KneeModel
from knee.inference import predict, write_submission
from knee.dinov2 import vit_small, load_pretrained


def find_checkpoint():
    """Search for best_model.pt in /kaggle/input/."""
    for root, dirs, files in os.walk("/kaggle/input"):
        if "best_model.pt" in files:
            return os.path.join(root, "best_model.pt")
    raise FileNotFoundError("best_model.pt not found in /kaggle/input/")


def main():
    t0 = time.time()

    # 1. Find data
    DATA_DIR = find_data_dir()
    SERIES_ROOT = os.path.join(DATA_DIR, "test_series")
    print("data dir:", DATA_DIR)

    # 2. Read metadata
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    series_df = pd.read_csv(os.path.join(DATA_DIR, "test_series.csv"))
    series_meta = series_df.to_dict("records")
    study_series = series_df.groupby("StudyInstanceUID")

    # 3. Select series per study
    tasks = []
    for uid in test_df["StudyInstanceUID"]:
        if uid in study_series.groups:
            rows = study_series.get_group(uid).to_dict("records")
            uids = select_series(rows)
        else:
            uids = []
        tasks.append((uid, uids, SERIES_ROOT))

    print(f"test studies: {len(tasks)}")

    # 4. Preprocess with multiprocessing
    N_WORKERS = cfg("preprocessing", "n_workers", default=4)
    CHUNKSIZE = cfg("preprocessing", "chunksize", default=8)

    raw_tensors = {}
    fails = 0
    with Pool(N_WORKERS) as pool:
        for uid, ok, tensor in pool.imap(process_study, tasks, chunksize=CHUNKSIZE):
            if not ok:
                fails += 1
            raw_tensors[uid] = tensor

    print(f"preprocessing done: {len(raw_tensors)} studies, {fails} failures, "
          f"{time.time()-t0:.1f}s")

    # 5. Apply prep_tensor (normalize, replicate channels)
    uids_ordered = list(test_df["StudyInstanceUID"])
    tensors = np.stack([raw_tensors[u] for u in uids_ordered])
    tensors = prep_tensor(tensors)  # (N, 3, 8, 224, 224) uint8 -> (N, 24, 3, 224, 224) float32
    print(f"prep_tensor done: {tensors.shape}, {tensors.dtype}")

    # 6. Load model
    ckpt_path = find_checkpoint()
    print(f"checkpoint: {ckpt_path}")

    from knee.inference import load_checkpoint
    model, model_cfg = load_checkpoint(ckpt_path)
    print(f"model loaded: {model_cfg}")

    # 7. Predict
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    probs = predict(model, tensors, batch_size=32, device=device)
    print(f"predictions: {probs.shape}, range [{probs.min():.4f}, {probs.max():.4f}]")

    # 8. Write submission
    out_path = "/kaggle/working/submission.csv"
    write_submission(probs, uids_ordered, out_path)
    print(f"submission written: {out_path}")
    print(f"total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
