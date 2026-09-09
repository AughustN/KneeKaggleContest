"""Phase 1: one-pass preprocessing of the 569GB train corpus -> compact npz shards.

RUN ON KAGGLE (CPU notebook, ~1.5-2h). Output -> /kaggle/working/shards/
Then SAVE VERSION -> new Kaggle Dataset for Phase 3 training.

Input layout:  <data>/train_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
Output:        shards/part-XXXX.npz  (each ~2000 studies)
                meta.csv  (StudyInstanceUID, shard, offset, n_series_actual)
                series_selection.csv (study -> chosen SeriesInstanceUIDs, for reuse)

Design (from Phase 0 recon + Phase 2):
- 3 series/study by priority buckets, 8 slices each, 224x224 uint8
- selector identical to src/knee/preprocess.py (single source of truth)
- multiprocessing: 4 workers (Kaggle has 4 vCPUs)
- per-study tensor shape fixed (3, 8, 224, 224) so Phase 3 streams npz directly
"""

import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
for p in ("/kaggle/working", _SRC):
    if os.path.isdir(p) or p == _SRC:
        if p not in sys.path:
            sys.path.insert(0, p)

from knee.config import get as cfg  # noqa: E402
from knee.preprocess import (  # noqa: E402
    IMG_SIZE, MAX_SERIES, NUM_SLICES, find_data_dir, process_study,
    select_series,
)

SHARDS_PER_FILE = cfg("preprocessing", "shards_per_file", default=2000)
N_WORKERS = cfg("preprocessing", "n_workers", default=4)
CHUNKSIZE = cfg("preprocessing", "chunksize", default=8)

DATA_DIR = os.environ.get("KNEE_DATA_DIR", "")
if not DATA_DIR:
    try:
        DATA_DIR = find_data_dir()
    except FileNotFoundError:
        DATA_DIR = find_data_dir(".") if os.path.exists("train.csv") else "."

OUT_DIR = os.environ.get(
    "KNEE_OUT_DIR",
    os.path.join(cfg("paths", "kaggle_working", default="/kaggle/working"),
                 cfg("paths", "output_subdir", default="shards"))
)
SERIES_ROOT = os.path.join(DATA_DIR, "train_series")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    series_meta = pd.read_csv(os.path.join(DATA_DIR, "train_series.csv"))
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

    # deterministic selection per study; tasks carry series_root (picklable)
    t0 = time.time()
    selections = {}
    grouped = series_meta.groupby("StudyInstanceUID")
    tasks = []
    for uid, grp in grouped:
        uids = select_series(grp.to_dict("records"))
        selections[uid] = uids
        tasks.append((uid, uids))
    tasks = [(uid, u, SERIES_ROOT) for uid, u in tasks]
    print(f"series selection: {len(tasks)} studies in {time.time()-t0:.1f}s")

    # save selection for reuse (train+test consistency)
    sel_rows = [
        {"StudyInstanceUID": uid, "S1": u[0] if len(u) > 0 else "",
         "S2": u[1] if len(u) > 1 else "", "S3": u[2] if len(u) > 2 else ""}
        for uid, u in selections.items()
    ]
    pd.DataFrame(sel_rows).to_csv(
        os.path.join(OUT_DIR, "series_selection.csv"), index=False
    )

    meta_rows = []
    shard, in_shard, fails = 0, 0, 0
    buf = {}

    t0 = time.time()
    with Pool(N_WORKERS) as pool:
        for uid, ok, tensor in pool.imap(process_study, tasks, chunksize=CHUNKSIZE):
            if not ok or tensor is None:
                fails += 1
                tensor = np.zeros((MAX_SERIES, NUM_SLICES, IMG_SIZE, IMG_SIZE), np.uint8)
            buf[uid] = tensor
            in_shard += 1

            if in_shard == SHARDS_PER_FILE:
                _write_shard(buf, shard)
                meta_rows.extend(_meta_rows(buf, shard))
                buf = {}
                shard += 1
                in_shard = 0
                elapsed = time.time() - t0
                done = shard * SHARDS_PER_FILE
                eta = elapsed / done * (len(tasks) - done) / 60
                print(f"  shard {shard} written ({done} studies, "
                      f"{elapsed/60:.1f} min, ETA {eta:.1f} min, fails={fails})")

    if buf:
        _write_shard(buf, shard)
        meta_rows.extend(_meta_rows(buf, shard))

    pd.DataFrame(meta_rows).to_csv(os.path.join(OUT_DIR, "meta.csv"), index=False)
    print(f"done: {shard+1} shards, {len(tasks)} studies, {fails} failures, "
          f"{(time.time()-t0)/60:.1f} min")
    print(f"output: {OUT_DIR}")


def _write_shard(buf, shard):
    uids = sorted(buf)
    arr = np.stack([buf[u] for u in uids])
    np.savez_compressed(
        os.path.join(OUT_DIR, f"part-{shard:04d}.npz"),
        uids=np.array(uids),
        data=arr,
    )


def _meta_rows(buf, shard):
    return [
        {"StudyInstanceUID": u, "shard": shard, "offset": i}
        for i, u in enumerate(sorted(buf))
    ]


if __name__ == "__main__":
    main()
