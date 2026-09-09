"""Debug helper: for gold-58, show which sentences fire each label rule.

Usage: python notebooks/02_debug_fp.py [label] [study_index]
No args -> print gold co-occurrence matrix for Contusion x Fracture (convention check).
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from knee.config import get as cfg  # noqa: E402
from knee.constants import LABELS  # noqa: E402
from knee.text_extractor import (  # noqa: E402
    ACL_ANCHOR, ACL_INJURY, BAKER, CONTUSION, CONTUSION_CONFIRM,
    CONTUSION_EXCL, EFFUSION, EFF_MIN, FRACTURE, MENISCUS_ANCHOR,
    MEN_INJURY, MCL_ANCHOR, MCL_INJURY, OA_SOFT, OA_TERMS, PCL_ANCHOR,
    SYNOVITIS, _LAT_RE, _MED_RE, _PF_RE, _SENT_SPLIT, _SECTION_STRIP,
    normalize, sentence_polarity,
)

DATA = cfg("paths", "local_data")


def get_gold():
    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    return train[train[LABELS].notna().any(axis=1)].reset_index(drop=True)


def sents_of(row):
    norm = normalize(str(row["Report"]))
    body = _SECTION_STRIP.sub(" ", norm)
    return [s for s in _SENT_SPLIT.split(body) if s.strip()]


def show(label: str, idx: int):
    gold = get_gold()
    row = gold.iloc[idx]
    print(f"study[{idx}] gold {label} = {row[label]}")
    which = {
        "ACL": lambda s: ACL_ANCHOR.search(s) and ACL_INJURY.search(s),
        "MCL": lambda s: MCL_ANCHOR.search(s) and MCL_INJURY.search(s),
        "Effusion": lambda s: EFFUSION.search(s),
        "Synovitis": lambda s: SYNOVITIS.search(s),
        "Baker's": lambda s: BAKER.search(s),
        "Contusion": lambda s: CONTUSION.search(s),
        "Fracture": lambda s: FRACTURE.search(s),
    }
    oa = lambda s: OA_TERMS.search(s)  # noqa: E731
    if label.endswith("Meniscus"):
        which[label] = lambda s: MENISCUS_ANCHOR.search(s) and MEN_INJURY.search(s)
    elif label.endswith("OA"):
        which[label] = oa
    fn = which[label]
    for i, s in enumerate(sents_of(row)):
        if fn(s):
            extra = ""
            if label.endswith("OA"):
                if _MED_RE.search(s):
                    extra += " [MED]"
                if _LAT_RE.search(s):
                    extra += " [LAT]"
                if _PF_RE.search(s):
                    extra += " [PF]"
                if OA_SOFT.search(s):
                    extra += " [SOFT]"
            if label == "Contusion":
                if CONTUSION_EXCL.search(s):
                    extra += " [EXCL]"
                if CONTUSION_CONFIRM.search(s):
                    extra += " [CONFIRM]"
            if label == "Effusion" and EFF_MIN.search(s):
                extra += " [MIN]"
            print(f"  p={sentence_polarity(s):.1f}{extra} | {s[:180]}")


def cooccurrence():
    gold = get_gold()
    y = gold[LABELS].to_numpy()
    i, j = LABELS.index("Contusion"), LABELS.index("Fracture")
    print(pd.crosstab(y[:, i], y[:, j], rownames=["Contusion"], colnames=["Fracture"]))
    print()
    i = LABELS.index("Effusion")
    print(pd.crosstab(y[:, i], y[:, j], rownames=["Effusion"], colnames=["Fracture"]))
    print()
    i, k = LABELS.index("Effusion"), LABELS.index("Synovitis")
    print(pd.crosstab(y[:, i], y[:, k], rownames=["Effusion"], colnames=["Synovitis"]))
    print()
    i, k = LABELS.index("Baker's"), LABELS.index("Effusion")
    print(pd.crosstab(y[:, i], y[:, k], rownames=["Baker's"], colnames=["Effusion"]))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        cooccurrence()
    else:
        show(sys.argv[1], int(sys.argv[2]))
