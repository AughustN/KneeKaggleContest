"""Regenerate notebooks/10-preprocess.ipynb (self-contained Kaggle notebook).

Embeds src/knee/preprocess.py and config.yaml VERBATIM (json-escaped) so the
notebook can never drift from the repo. The notebook writes the package itself
on Kaggle — no repo upload needed; only the competition dataset must be attached.

Usage: python notebooks/build_preprocess_ipynb.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
NB_PATH = os.path.join(HERE, "10-preprocess.ipynb")

payload = open(
    os.path.join(REPO, "src", "knee", "preprocess.py"), encoding="utf-8"
).read()

config_payload = open(
    os.path.join(REPO, "config.yaml"), encoding="utf-8"
).read()

config_py_payload = open(
    os.path.join(REPO, "src", "knee", "config.py"), encoding="utf-8"
).read()

CELL_PACKAGE = """import os

os.makedirs("/kaggle/working/knee", exist_ok=True)

with open("/kaggle/working/knee/__init__.py", "w") as f:
    f.write('"\"\"knee: RSNA knee abnormality challenge package.\"\"\"\\n\\n__version__ = "0.2.0"\\n')

with open("/kaggle/working/knee/constants.py", "w", encoding="utf-8") as f:
    f.write(
        '\\"\\"\\"Shared constants for the knee package.\\"\\"\\"\\n\\n'
        'from typing import List\\n\\n'
        'STUDY_ID = "StudyInstanceUID"\\n\\n'
        'LABELS: List[str] = [\\n'
        '    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",\\n'
        '    "Medial OA", "Lateral OA", "PF OA",\\n'
        '    "Effusion", "Synovitis", "Baker\\'s", "Contusion", "Fracture",\\n'
        ']\\n\\n'
        'NUM_LABELS = len(LABELS)  # 12\\n'
    )

# Write config.yaml (embedded from repo root)
_CONFIG_YAML = {config_yaml!r}
with open("/kaggle/working/config.yaml", "w", encoding="utf-8") as f:
    f.write(_CONFIG_YAML)

# Write knee/config.py (embedded from repo)
_CONFIG_PY = {config_py!r}
with open("/kaggle/working/knee/config.py", "w", encoding="utf-8") as f:
    f.write(_CONFIG_PY)

print("package skeleton + config written")
"""

CELL_PAYLOAD = """# knee/preprocess.py source — embedded VERBATIM from repo src/knee/preprocess.py.
# Regenerate the notebook with notebooks/build_preprocess_ipynb.py; do not edit.
_PP_PAYLOAD = {payload!r}

with open("/kaggle/working/knee/preprocess.py", "w", encoding="utf-8") as f:
    f.write(_PP_PAYLOAD)
import py_compile
py_compile.compile("/kaggle/working/knee/preprocess.py", doraise=True)
print("knee/preprocess.py written and syntax-checked:", len(_PP_PAYLOAD), "chars")
"""

CELL_RUN = '''# --- Phase 1 run (mirrors notebooks/10-preprocess.py) ---
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, "/kaggle/working")

from knee.config import get as cfg
from knee.preprocess import (
    IMG_SIZE, MAX_SERIES, NUM_SLICES, find_data_dir, process_study, select_series,
)

SHARDS_PER_FILE = cfg("preprocessing", "shards_per_file", default=2000)
N_WORKERS = cfg("preprocessing", "n_workers", default=4)
CHUNKSIZE = cfg("preprocessing", "chunksize", default=8)

DATA_DIR = find_data_dir()
SERIES_ROOT = os.path.join(DATA_DIR, "train_series")
print("data dir:", DATA_DIR)

OUT_DIR = os.path.join(cfg("paths", "kaggle_working", default="/kaggle/working"),
                       cfg("paths", "output_subdir", default="shards"))
os.makedirs(OUT_DIR, exist_ok=True)

series_meta = pd.read_csv(os.path.join(DATA_DIR, "train_series.csv"))

t0 = time.time()
selections = {}
tasks = []
for uid, grp in series_meta.groupby("StudyInstanceUID"):
    uids = select_series(grp.to_dict("records"))
    selections[uid] = uids
    tasks.append((uid, uids))
tasks = [(uid, u, SERIES_ROOT) for uid, u in tasks]
print(f"series selection: {len(tasks)} studies in {time.time()-t0:.1f}s")

sel_rows = [
    {"StudyInstanceUID": uid, "S1": u[0] if len(u) > 0 else "",
     "S2": u[1] if len(u) > 1 else "", "S3": u[2] if len(u) > 2 else ""}
    for uid, u in selections.items()
]
pd.DataFrame(sel_rows).to_csv(os.path.join(OUT_DIR, "series_selection.csv"), index=False)


def _write_shard(buf, shard):
    uids = sorted(buf)
    arr = np.stack([buf[u] for u in uids])
    np.savez_compressed(
        os.path.join(OUT_DIR, f"part-{shard:04d}.npz"),
        uids=np.array(uids),
        data=arr,
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
            meta_rows.extend(
                {"StudyInstanceUID": u, "shard": shard, "offset": i}
                for i, u in enumerate(sorted(buf))
            )
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
    meta_rows.extend(
        {"StudyInstanceUID": u, "shard": shard, "offset": i}
        for i, u in enumerate(sorted(buf))
    )

pd.DataFrame(meta_rows).to_csv(os.path.join(OUT_DIR, "meta.csv"), index=False)
print(f"done: {shard+1} shards, {len(tasks)} studies, {fails} failures, "
      f"{(time.time()-t0)/60:.1f} min")
print("output:", OUT_DIR)
'''

CELL_SANITY = '''# sanity check
import os

import numpy as np

shards = sorted(f for f in os.listdir(OUT_DIR) if f.endswith(".npz"))
print("shards:", shards)
z = np.load(os.path.join(OUT_DIR, shards[0]))
print("shape:", z["data"].shape, "dtype:", z["data"].dtype)
print("study 0: max=", z["data"][0].max(), "mean=", z["data"][0].mean().round(1))
assert z["data"].shape[1:] == (3, 8, 224, 224)
assert z["data"].max() > 0
print("OK")
'''


def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


nb = {
    "cells": [
        md(
            "# Phase 1 — Knee preprocessing (self-contained)\n"
            "\n"
            "Run on Kaggle (CPU notebook) with **only the competition dataset attached**.\n"
            "No repo upload needed — this notebook writes the `knee` package itself\n"
            "(payload embedded verbatim from `src/knee/preprocess.py`).\n"
            "\n"
            "Output: `/kaggle/working/shards/` (part-*.npz + meta.csv + series_selection.csv)\n"
            "→ **Save Version → Save & Run All → New Dataset** named `knee-shards`.\n"
            "\n"
            "Pipeline: 3 series/study by plane/fluid priority, 8 slices/series sorted by\n"
            "ImagePositionPatient[2], 224x224 uint8, per-volume percentile normalization.\n"
            "~4.7 GB total, ~1.5-2h with 4 workers."
        ),
        code(CELL_PACKAGE.format(config_yaml=config_payload, config_py=config_py_payload)),
        code(CELL_PAYLOAD.format(payload=payload)),
        code(CELL_RUN),
        md(
            "## Next\n"
            "\n"
            "1. Quick sanity below\n"
            "2. **Save Version → Save & Run All → New Dataset** named `knee-shards`"
        ),
        code(CELL_SANITY),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"wrote {NB_PATH} ({len(payload)}-char payload embedded)")
