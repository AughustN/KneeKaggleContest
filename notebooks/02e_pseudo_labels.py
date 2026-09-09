"""Phase 2e: generate pseudo-labels for all unlabeled training studies.

Method: per-label selective ensemble (rule extractor + LOO-selected blend with
char-ngram LR where it validated better), then confidence thresholding:

  - scores >= 0.75 -> pseudo 1.0 (confident positive)
  - scores <= 0.25 -> pseudo 0.0 (confident negative, mainly explicit negation)
  - otherwise      -> -1.0 (masked; excluded from that label's training loss)

Outputs:
  data/pseudo_labels.csv  StudyInstanceUID + 12 label columns with -1/0/1
  reports/02-pseudo-stats.txt  distribution stats + spot checks
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from knee.config import get as cfg  # noqa: E402
from knee.constants import LABELS, STUDY_ID  # noqa: E402
from knee.text_extractor import extract_frame, normalize  # noqa: E402

DATA = cfg("paths", "local_data")
OUT_CSV = os.path.join(DATA, "data", "pseudo_labels.csv")
OUT_TXT = os.path.join(DATA, "reports", "02-pseudo-stats.txt")

# per-label blend weights (rule vs ml) from LOO selection
ENS_LABELS = set(cfg("pseudo_labels", "ensemble_labels", default=[
    "Medial Meniscus", "Medial OA", "Lateral OA", "Synovitis", "Fracture",
]))

POS_TH = cfg("pseudo_labels", "positive_threshold", default=0.75)
NEG_TH = cfg("pseudo_labels", "negative_threshold", default=0.25)


def make_clf():
    return make_pipeline(
        TfidfVectorizer(
            analyzer=cfg("ml", "tfidf_analyzer", default="char_wb"),
            ngram_range=tuple(cfg("ml", "tfidf_ngram_range", default=[2, 5])),
            min_df=cfg("ml", "tfidf_min_df", default=1),
            max_features=cfg("ml", "tfidf_max_features", default=200000),
            sublinear_tf=cfg("ml", "tfidf_sublinear_tf", default=True),
        ),
        LogisticRegression(
            C=cfg("ml", "lr_C", default=2.0),
            max_iter=cfg("ml", "lr_max_iter", default=3000),
            class_weight=cfg("ml", "lr_class_weight", default="balanced"),
        ),
    )


def main() -> int:
    os.makedirs(os.path.join(DATA, "data"), exist_ok=True)
    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    gold = train[train[LABELS].notna().any(axis=1)].reset_index(drop=True)
    unlabeled = train[train[LABELS].isna().all(axis=1)].reset_index(drop=True)

    lines = []
    emit = lines.append
    emit(f"unlabeled studies: {len(unlabeled)}")

    # rule scores for both sets
    rule_unl = extract_frame(unlabeled)
    rule_gold = extract_frame(gold)

    # ML models fit on gold (deployment fit, LOO was only for validation)
    texts_gold = [normalize(str(r)) for r in gold["Report"]]
    texts_unl = [normalize(str(r)) for r in unlabeled["Report"]]
    y_gold = gold[LABELS].to_numpy(dtype=float)

    ml_probs = {}
    for j, lab in enumerate(LABELS):
        clf = make_clf()
        clf.fit(texts_gold, y_gold[:, j])
        ml_probs[lab] = clf.predict_proba(texts_unl)[:, 1]

    # blend scores
    scores = pd.DataFrame(index=unlabeled.index)
    for lab in LABELS:
        r = rule_unl[lab].to_numpy(dtype=float)
        r_soft = np.where(r < 0, 0.25, r)  # no-mention -> weak negative prior
        if lab in ENS_LABELS:
            m = ml_probs[lab]
            # blend; ML is weak overall so weight it lightly
            scores[lab] = 0.75 * r_soft + 0.25 * m
        else:
            scores[lab] = r_soft

    # threshold into pseudo labels
    pseudo = pd.DataFrame({STUDY_ID: unlabeled[STUDY_ID].astype(str)})
    emit("")
    emit(f"{'label':<17}{'>=.75':>6}{'<=.25':>6}{'masked':>8}{'p_pos':>7}")
    for lab in LABELS:
        s = scores[lab].to_numpy(dtype=float)
        pl = np.full(len(s), -1.0)
        pl[s >= POS_TH] = 1.0
        pl[s <= NEG_TH] = 0.0
        pseudo[lab] = pl
        n_pos = int((pl == 1).sum())
        n_neg = int((pl == 0).sum())
        n_mask = int((pl < 0).sum())
        p_pos = n_pos / max(n_pos + n_neg, 1)
        emit(f"{lab:<17}{n_pos:>6}{n_neg:>6}{n_mask:>8}{p_pos:>7.2f}")

    # gold prevalence comparison (from recon)
    emit("")
    emit("gold-58 prevalence for comparison (labeled positives/58):")
    prev = {c: float(gold[c].sum()) / 58 for c in LABELS}
    emit("  " + ", ".join(f"{c}={v:.2f}" for c, v in prev.items()))

    pseudo.to_csv(OUT_CSV, index=False)
    emit("")
    emit(f"wrote {OUT_CSV}")

    out = "\n".join(lines)
    print(out)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(out + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
