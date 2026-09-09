"""Regenerate notebooks/40-infer.ipynb (self-contained Kaggle notebook).

Embeds the entire src/knee/ package VERBATIM so the notebook can run
with Internet OFF. No repo upload needed.

Usage: python notebooks/build_infer_ipynb.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
NB_PATH = os.path.join(HERE, "40-infer.ipynb")

# Read source files to embed
SRC_FILES = {
    "config.py": os.path.join(REPO, "src", "knee", "config.py"),
    "constants.py": os.path.join(REPO, "src", "knee", "constants.py"),
    "preprocess.py": os.path.join(REPO, "src", "knee", "preprocess.py"),
    "inference.py": os.path.join(REPO, "src", "knee", "inference.py"),
    "config.yaml": os.path.join(REPO, "config.yaml"),
}

# DINOv2 vendored files — use forward slashes for Kaggle (Linux) compatibility
DINOV2_FILES = {}
dinov2_dir = os.path.join(REPO, "src", "knee", "dinov2")
for root, dirs, files in os.walk(dinov2_dir):
    for f in files:
        if f.endswith(".py") or f.endswith(".pth"):
            full = os.path.join(root, f)
            rel = os.path.relpath(full, os.path.join(REPO, "src", "knee"))
            # Normalize to forward slashes for Kaggle paths
            rel = rel.replace(chr(92), "/")
            DINOV2_FILES[rel] = full

# model.py and dataset.py — embedded as stubs until A provides them
# The notebook will try to import from /kaggle/working/knee/model.py
# If A's model.py is attached as a dataset, it will be used instead.

CELL_PACKAGE = """import os, sys

os.makedirs("/kaggle/working/knee", exist_ok=True)
os.makedirs("/kaggle/working/knee/dinov2", exist_ok=True)
os.makedirs("/kaggle/working/knee/dinov2/layers", exist_ok=True)

# __init__.py
with open("/kaggle/working/knee/__init__.py", "w") as f:
    f.write('"\\\"\\\"\\\"knee package.\\\"\\\"\\\"\\n__version__ = \\"0.2.0\\"\\n')

# constants.py
with open("/kaggle/working/knee/constants.py", "w", encoding="utf-8") as f:
    f.write(
        'from typing import List\\n'
        'STUDY_ID = "StudyInstanceUID"\\n'
        'LABELS: List[str] = [\\n'
        '    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",\\n'
        '    "Medial OA", "Lateral OA", "PF OA",\\n'
        '    "Effusion", "Synovitis", "Baker\\'s", "Contusion", "Fracture",\\n'
        ']\\n'
        'NUM_LABELS = len(LABELS)\\n'
    )

print("package skeleton written")
"""

# Build cells for each embedded source file
def make_embed_cell(filename, content):
    return f'''# --- {filename} (embedded from repo) ---
_PAYLOAD_{filename.replace(".", "_").replace("/", "_")} = {content!r}

_target = "/kaggle/working/knee/{filename}"
os.makedirs(os.path.dirname(_target), exist_ok=True)
with open(_target, "w", encoding="utf-8") as f:
    f.write(_PAYLOAD_{filename.replace(".", "_").replace("/", "_")})
import py_compile
py_compile.compile(_target, doraise=True)
print("{filename} written:", len(_PAYLOAD_{filename.replace(".", "_", ).replace("/", "_")}), "chars")
'''

# Build cells for dinov2 files
def make_dinov2_cell(filename, content):
    # Sanitize variable name: replace all non-alphanumeric chars with _
    safe_var = "DINOV2_" + "".join(c if c.isalnum() else "_" for c in filename)
    # filename is like "dinov2/vision_transformer.py" — already includes dinov2/ prefix
    # target = /kaggle/working/knee/ + filename = /kaggle/working/knee/dinov2/vision_transformer.py
    target = "/kaggle/working/knee/" + filename
    return f'''# --- knee/{filename} ---
_{safe_var} = {content!r}

_target = "{target}"
os.makedirs(os.path.dirname(_target), exist_ok=True)
with open(_target, "w", encoding="utf-8") as f:
    f.write(_{safe_var})
print("{filename} written")
'''

# Read the inference payload
with open(os.path.join(HERE, "40-infer.py"), encoding="utf-8") as f:
    infer_payload = f.read()

CELL_INFER = f'''# --- Inference payload ---
_INFER_PAYLOAD = {infer_payload!r}

with open("/kaggle/working/40_infer_payload.py", "w", encoding="utf-8") as f:
    f.write(_INFER_PAYLOAD)

exec(open("/kaggle/working/40_infer_payload.py").read())
'''

# Assemble notebook
cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Phase 4 — Knee Inference & Submission (self-contained)\n",
            "\n",
            "Run on Kaggle (GPU notebook) with **Internet OFF**.\n",
            "Attach: competition dataset + `knee-weights` + `dinov2-weights`.\n",
            "\n",
            "Output: `/kaggle/working/submission.csv`\n",
        ],
    },
    {"cell_type": "code", "metadata": {}, "source": [CELL_PACKAGE], "outputs": [], "execution_count": None},
]

# Embed src/knee/ source files
for name, path in SRC_FILES.items():
    with open(path, encoding="utf-8") as f:
        content = f.read()
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [make_embed_cell(name, content)],
        "outputs": [],
        "execution_count": None,
    })

# Embed dinov2 files (skip .pth — will be loaded from dataset)
for name, path in DINOV2_FILES.items():
    if name.endswith(".pth"):
        continue
    with open(path, encoding="utf-8") as f:
        content = f.read()
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [make_dinov2_cell(name, content)],
        "outputs": [],
        "execution_count": None,
    })

# Inference cell
cells.append({
    "cell_type": "code",
    "metadata": {},
    "source": [CELL_INFER],
    "outputs": [],
    "execution_count": None,
})

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.12"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"wrote {NB_PATH} ({len(cells)} cells)")
