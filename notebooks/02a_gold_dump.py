"""Phase 2a: inspect gold-58 studies — labels vs full report text, per language.

Goal: learn how each of the 12 findings is phrased (and negated) in
English/Turkish/Spanish/German/Dutch/French/Cyrillic reports so we can design
the keyword extractor. Dumps full reports for the labeled studies.
"""

import os
import sys
import unicodedata

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from knee.config import get as cfg  # noqa: E402
from knee.constants import LABELS  # noqa: E402

DATA = cfg("paths", "local_data")
OUT = os.path.join(DATA, "reports", "02-gold-58-dump.txt")

LANG_HINTS = {
    "english": ["meniscus", "effusion", "tear", "findings", "no evidence", "mri knee"],
    "spanish": ["menisco", "derrame", "rodilla", "hallazgos", "rotura", "rmn"],
    "german": ["meniskus", "erguss", "kniegelenk", "kreuzband", "bandapparat", "binnenband"],
    "dutch": ["meniscus", "vocht", "beeldvorming", "bevindingen", "collateraal", "knie"],
    "french": ["menisque", "epanchement", "genou", "constatations", "ligament croise"],
    "turkish": ["menisk", "diz", "eklemi", "mrg", "bulgular", "sivi"],
    "cyrillic": ["мениск", "колян", "излив", "ставен"],
}


def strip_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def detect_lang(text: str) -> str:
    if any("\u0400" <= ch <= "\u04FF" for ch in text):
        return "cyrillic"
    t = strip_accents(text.lower())
    scores = {k: sum(t.count(p) for p in pats) for k, pats in LANG_HINTS.items()}
    scores = {k: v for k, v in scores.items() if v}
    return max(scores, key=scores.get) if scores else "other"


def main() -> int:
    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    gold = train[train[LABELS].notna().any(axis=1)].reset_index(drop=True)
    print(f"gold studies: {len(gold)}")

    langs = gold["Report"].map(detect_lang)
    print("language mix of gold-58:", langs.value_counts().to_dict())

    lines = []
    for i, row in gold.iterrows():
        labs = {c: int(row[c]) for c in LABELS}
        pos = [c for c, v in labs.items() if v == 1]
        neg = [c for c, v in labs.items() if v == 0]
        lines.append("=" * 100)
        lines.append(f"[{i}] lang={langs.iloc[i]}  POS: {pos}")
        lines.append(f"         NEG: {neg}")
        lines.append("-" * 100)
        lines.append(str(row["Report"]))
        lines.append("")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
