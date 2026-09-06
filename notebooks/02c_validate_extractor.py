"""Phase 2c: validate the keyword extractor against the 58 gold studies.

Outputs per-label: accuracy, precision, recall, F1, AUC (using soft scores),
plus dumps every error case (FP/FN) with report snippets for iteration.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from knee.constants import LABELS  # noqa: E402
from knee.text_extractor import extract_frame, normalize  # noqa: E402

DATA = r"E:\KneeAbnormal"
OUT = os.path.join(DATA, "reports", "02-extractor-errors.txt")


def main() -> int:
    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    gold = train[train[LABELS].notna().any(axis=1)].reset_index(drop=True)

    scores = extract_frame(gold)  # -1 / 0 / 0.5 / 1
    y = gold[LABELS].to_numpy(dtype=float)

    lines = []
    emit = lines.append

    def blank():
        lines.append("")

    emit("# Extractor vs gold-58")
    emit(f"{'label':<17}{'acc':>6}{'prec':>6}{'rec':>6}{'f1':>6}{'auc':>7}"
         f"{'FP':>4}{'FN':>4}{'miss':>5}")
    accs, f1s, aucs = [], [], []
    all_fp, all_fn = [], []
    for j, lab in enumerate(LABELS):
        s = scores[lab].to_numpy(dtype=float)
        yt = y[:, j]
        # hard prediction: positive if score >= 0.5 (hedged counts as pos),
        # negative if score == 0 (explicit negation), else default negative
        pred = (s >= 0.5).astype(int)
        tp = int(((pred == 1) & (yt == 1)).sum())
        fp = int(((pred == 1) & (yt == 0)).sum())
        fn = int(((pred == 0) & (yt == 1)).sum())
        tn = int(((pred == 0) & (yt == 0)).sum())
        acc = (tp + tn) / len(yt)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        # AUC over soft scores: map -1 (no mention) to 0.25 (weak negative prior)
        soft = np.where(s < 0, 0.25, s)
        try:
            auc = roc_auc_score(yt, soft)
        except ValueError:
            auc = float("nan")
        miss = int((s < 0).sum())
        emit(f"{lab:<17}{acc:>6.2f}{prec:>6.2f}{rec:>6.2f}{f1:>6.2f}{auc:>7.3f}"
             f"{fp:>4}{fn:>4}{miss:>5}")
        accs.append(acc)
        f1s.append(f1)
        aucs.append(auc if auc == auc else 0.5)
        for i in np.where((pred == 1) & (yt == 0))[0]:
            all_fp.append((lab, i))
        for i in np.where((pred == 0) & (yt == 1))[0]:
            all_fn.append((lab, i))

    blank()
    emit(f"macro: acc={np.mean(accs):.3f}  f1={np.mean(f1s):.3f}  auc={np.mean(aucs):.3f}")
    emit(f"total FP={len(all_fp)}  FN={len(all_fn)}")

    # error dump
    blank()
    emit("# Errors")
    seen = set()
    for lab, i in all_fp + all_fn:
        if (lab, i) in seen:
            continue
        seen.add((lab, i))
        yt = y[i, LABELS.index(lab)]
        kind = "FP" if yt == 0 else "FN"
        sc = scores.iloc[i][lab]
        emit(f"--- [{kind}] {lab}  gold={int(yt)}  extractor={sc}  study[{i}]")
        rpt = normalize(gold.iloc[i]["Report"])
        emit(f"    norm-report[:600]: {rpt[:600]}")
        blank()

    out = "\n".join(lines)
    print(out)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(out + "\n")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
