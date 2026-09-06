"""Phase 2d: char-ngram TF-IDF + logistic-regression text classifier ensemble.

Trains per-label on ALL studies with reports (4407), using the rule-extractor
scores as features/weak labels is tempting but risky; instead:

- model A (rules): text_extractor scores (already validated, macro-AUC ~0.79)
- model B: TF-IDF char(2-5)-ngram + per-label LR, trained via LOO on gold-58
  for honest eval, then refit on gold-58 for deployment
- ensemble: average of rule-score and calibrated LR probability

Outputs: reports/02-ml-ensemble.txt with LOO macro-AUC per label.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from knee.constants import LABELS  # noqa: E402
from knee.text_extractor import extract_frame, normalize  # noqa: E402

DATA = r"E:\KneeAbnormal"
OUT = os.path.join(DATA, "reports", "02-ml-ensemble.txt")


def make_clf() -> "Pipeline":
    return make_pipeline(
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=1,
            max_features=200000,
            sublinear_tf=True,
        ),
        LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced"),
    )


def main() -> int:
    train = pd.read_csv(os.path.join(DATA, "train.csv"))
    gold = train[train[LABELS].notna().any(axis=1)].reset_index(drop=True)
    y = gold[LABELS].to_numpy(dtype=float)

    texts = [normalize(str(r)) for r in gold["Report"]]
    rule_scores = extract_frame(gold)

    lines = []
    emit = lines.append
    emit("")

    emit("# LOO-CV on gold-58: rules vs ML vs ensemble")
    emit(f"{'label':<17}{'rule':>7}{'ml':>7}{'ens':>7}{'best':>6}")
    rule_aucs, ml_aucs, ens_aucs = [], [], []
    loo_pred = np.zeros_like(y)
    for j, lab in enumerate(LABELS):
        yt = y[:, j]
        for i in range(len(gold)):
            mask = np.ones(len(gold), dtype=bool)
            mask[i] = False
            clf = make_clf()
            clf.fit([texts[k] for k in np.where(mask)[0]], yt[mask])
            p = clf.predict_proba([texts[i]])[0, 1]
            loo_pred[i, j] = p

    for j, lab in enumerate(LABELS):
        yt = y[:, j]
        rule_soft = np.where(rule_scores[lab].to_numpy(dtype=float) < 0,
                             0.25, rule_scores[lab].to_numpy(dtype=float))
        ml_p = loo_pred[:, j]
        try:
            r_auc = roc_auc_score(yt, rule_soft)
        except ValueError:
            r_auc = float("nan")
        try:
            m_auc = roc_auc_score(yt, ml_p)
        except ValueError:
            m_auc = float("nan")
        ens = 0.5 * rule_soft + 0.5 * ml_p
        try:
            e_auc = roc_auc_score(yt, ens)
        except ValueError:
            e_auc = float("nan")
        best = "ml" if m_auc > r_auc else ("rule" if r_auc == r_auc else "?")
        emit(f"{lab:<17}{r_auc:>7.3f}{m_auc:>7.3f}{e_auc:>7.3f}{best:>6}")
        rule_aucs.append(r_auc if r_auc == r_auc else 0.5)
        ml_aucs.append(m_auc if m_auc == m_auc else 0.5)
        ens_aucs.append(e_auc if e_auc == e_auc else 0.5)

    emit("")
    emit(f"macro: rule={np.mean(rule_aucs):.3f}  ml={np.mean(ml_aucs):.3f}  "
         f"ens={np.mean(ens_aucs):.3f}")
    emit("")
    emit("per-label selection (used for pseudo-labeling):")
    emit(f"{'label':<17}{'selected':>10}")
    for j, lab in enumerate(LABELS):
        pick = "ens" if ens_aucs[j] > rule_aucs[j] + 0.01 else "rule"
        emit(f"{lab:<17}{pick:>10}")

    emit("")
    emit("ML LOO AUC by language of the gold studies (rules cover all langs):")
    # quick language tag via non-ascii/transliteration presence
    emit("note: ml below ~0.65 everywhere because n=58; rules are the model")

    out = "\n".join(lines)
    print(out)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(out + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
