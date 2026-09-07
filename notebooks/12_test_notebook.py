"""Execute notebooks/10-preprocess.ipynb cells end-to-end against a simulated
Kaggle environment (temp dirs for /kaggle/input and /kaggle/working).

Multiprocessing is replaced with a serial shim: the real Pool+spawn path is
already covered by 11_test_preprocess.py; here we validate the NOTEBOOK's own
logic (payload writing, package bootstrap, find_data_dir, sharding, sanity).

Also verifies the embedded payload is byte-identical to src/knee/preprocess.py.
"""

import json
import os
import shutil
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))


class SerialPool:
    """Drop-in replacement for multiprocessing.Pool (serial, no spawn)."""

    def __init__(self, *a, **k):
        pass

    def imap(self, fn, items, chunksize=None):
        return iter([fn(t) for t in items])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def main() -> int:
    # --- simulated kaggle filesystem ---
    root = os.path.join(tempfile.gettempdir(), "nb_sim")
    if os.path.exists(root):
        shutil.rmtree(root)
    kg = os.path.join(root, "kaggle")
    os.makedirs(os.path.join(kg, "working"))
    comp = os.path.join(kg, "input", "competitions", "rsna-knee-abnormality-detection")

    # corpus source = synthetic set built by 11_test_preprocess.py
    src_corpus = os.path.join(tempfile.gettempdir(), "knee_test")
    if not os.path.exists(src_corpus):
        print("building synthetic corpus first (11_test_preprocess build step)")
        sys.exit("run notebooks/11_test_preprocess.py first")

    shutil.copytree(src_corpus, comp, ignore=shutil.ignore_patterns("shards"))

    # decoy input: train.csv WITHOUT train_series/ (must be skipped by finder)
    decoy = os.path.join(kg, "input", "my-repo-dataset", "src")
    os.makedirs(decoy)
    with open(os.path.join(decoy, "train.csv"), "w") as f:
        f.write("decoy")

    # --- payload identity check (notebook vs repo) ---
    nb = json.load(open(os.path.join(HERE, "10-preprocess.ipynb"), encoding="utf-8"))
    payload = None
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            if src.startswith("# knee/preprocess.py source"):
                payload = src.split("_PP_PAYLOAD = ", 1)[1].split("\n\n", 1)[0]
                import ast
                payload = ast.literal_eval(payload)
                break
    assert payload is not None, "payload cell not found in notebook"
    repo_src = open(os.path.join(REPO, "src", "knee", "preprocess.py"),
                   encoding="utf-8").read()
    assert payload == repo_src, "notebook payload differs from repo src!"
    print("payload check: notebook == repo (byte-identical)")

    # --- execute cells ---
    W = os.path.join(kg, "working").replace("\\", "/")
    I = os.path.join(kg, "input")

    g = {"__name__": "nb_exec", "__file__": os.path.join(HERE, "10-preprocess.ipynb"),
         "Pool": SerialPool}  # serial shim; Kaggle (fork) uses real multiprocessing
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        src = src.replace("/kaggle/working", W)
        src = src.replace('find_data_dir("/kaggle/input")',
                          f'find_data_dir(r"{I}")')
        src = src.replace("N_WORKERS = 4", "N_WORKERS = 2")
        src = src.replace("SHARDS_PER_FILE = 2000", "SHARDS_PER_FILE = 3")
        # remove the real import; shim Pool is pre-injected in globals
        src = src.replace("from multiprocessing import Pool\n", "")
        try:
            exec(compile(src, f"cell{i}", "exec"), g)
            print(f"--- cell {i} OK")
        except Exception:
            print(f"--- cell {i} FAILED")
            traceback.print_exc()
            return 1

    # --- verify outputs ---
    import numpy as np
    import pandas as pd

    out = os.path.join(kg, "working", "shards")
    files = sorted(os.listdir(out))
    print("shard files:", files)
    assert "meta.csv" in files and "series_selection.csv" in files

    z = np.load(os.path.join(out, "part-0000.npz"))
    assert z["data"].shape[1:] == (3, 8, 224, 224)
    assert z["data"].dtype == np.uint8
    assert z["data"].max() > 0
    meta = pd.read_csv(os.path.join(out, "meta.csv"))
    shard0_meta = meta[meta["shard"] == 0]
    assert set(shard0_meta["StudyInstanceUID"]) == set(str(u) for u in z["uids"])
    total = len(meta)
    assert total == 6, f"expected 6 studies in meta, got {total}"
    print("meta rows:", total, "| shard shape:", z["data"].shape)

    # the package the notebook wrote must equal repo's preprocess
    written = open(os.path.join(kg, "working", "knee", "preprocess.py"),
                   encoding="utf-8").read()
    assert written == repo_src, "written package differs from repo"
    print("written knee/preprocess.py == repo (byte-identical)")

    print("\nNOTEBOOK END-TO-END: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
