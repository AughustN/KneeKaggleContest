"""Phase 0 recon: profile train.csv / train_series.csv for the knee challenge.

Answers, from CSVs only (no DICOMs):
  1. How many studies are fully labeled vs partially vs not at all?
  2. Label prevalence per target (and per labeled subset).
  3. Report language mix + length stats.
  4. Series per study, slices per series (from series table), plane/fluid buckets.
  5. Series-selection feasibility: how often does each priority bucket exist per study?
Writes findings to reports/00-recon-data.md and prints a summary.
"""

import json
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from knee.config import get as cfg  # noqa: E402
from knee.constants import LABELS  # noqa: E402

DATA_DIR = cfg("paths", "local_data")
OUT_DIR = os.path.join(DATA_DIR, "reports")


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    series = pd.read_csv(os.path.join(DATA_DIR, "train_series.csv"))

    emit("# Phase 0 recon — CSV profile")
    emit()
    emit(f"train.csv: {len(train)} studies, columns: {list(train.columns)}")
    emit(f"train_series.csv: {len(series)} rows")
    emit()

    # ---------- label availability ----------
    lab = train[LABELS]
    n_any = int((~lab.isna()).any(axis=1).sum())
    n_all = int((~lab.isna()).all(axis=1).sum())
    n_none = int(lab.isna().all(axis=1).sum())
    n_any_per = int((~lab.isna()).sum(axis=1).value_counts().to_dict().get(0, 0)) if False else None
    counts_per_study = (~lab.isna()).sum(axis=1)
    emit("## Label availability")
    emit(f"- studies with >=1 label: {n_any} / {len(train)} ({n_any/len(train):.1%})")
    emit(f"- studies with all 12 labels: {n_all}")
    emit(f"- studies with 0 labels: {n_none}")
    emit("- labels-per-study distribution:")
    for k, v in counts_per_study.value_counts().sort_index().items():
        emit(f"    {k:>2} labels: {v} studies")
    emit()

    # ---------- label prevalence ----------
    emit("## Label prevalence (among labeled entries for that label)")
    prev_rows = []
    for c in LABELS:
        n_labeled = int(lab[c].notna().sum())
        n_pos = int(lab[c].sum())
        prev_rows.append((c, n_labeled, n_pos, n_pos / max(n_labeled, 1)))
    emit(f"{'label':<18}{'n_labeled':>10}{'n_pos':>8}{'prevalence':>12}")
    for c, nl, np_, p in prev_rows:
        emit(f"{c:<18}{nl:>10}{np_:>8}{p:>12.3f}")
    emit()

    # co-labeling: among labeled studies, correlation between labels
    emit("## Label correlation (Pearson, labeled pairs only, |r|>0.3 shown)")
    corr = lab.corr(min_periods=200)
    seen = set()
    for i, a in enumerate(LABELS):
        for b in LABELS[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) > 0.3:
                emit(f"    {a} ~ {b}: r={r:+.2f}")
    emit()

    # ---------- reports ----------
    emit("## Reports")
    has_report = train["Report"].notna() & (train["Report"].astype(str).str.len() > 0)
    emit(f"- studies with non-empty report: {has_report.sum()} ({has_report.mean():.1%})")
    rlen = train.loc[has_report, "Report"].astype(str).str.len()
    emit(f"- report length chars: mean={rlen.mean():.0f}, p50={rlen.median():.0f}, p95={rlen.quantile(0.95):.0f}")
    # language detection: script first, then characteristic keywords
    import unicodedata

    hints = {
        "english": ["meniscus", "effusion", "tear", "no evidence", "normal acl", "knee mri", "findings"],
        "spanish": ["menisco", "derrame", "rodilla", "rotura", "hallazgos", "no hay", "ligamento cruzado"],
        "german": ["meniskus", "erguss", "kniegelenk", "ruptur", "kreuzband", "bandapparat", "binnen"],
        "dutch": ["meniscus", "vocht", "knie", "beeldvorming", "paraartre", "bevindingen", "collateraal"],
        "french": ["menisque", "epanchement", "genou", "ligament croise", "constatations", "pas de"],
        "turkish": ["menisk", "diz", "eklemi", "bağ", "sıvı", "sivi", "bulgular", "mrg"],
    }
    cyrillic_hint = re.compile(r"[\u0400-\u04FF]")
    latin = set()
    for name, pats in hints.items():
        for ch in "".join(pats):
            if ord(ch) > 127:
                latin.add(ch)

    def detect_lang(text: str) -> str:
        if not isinstance(text, str) or not text:
            return "empty"
        if cyrillic_hint.search(text):
            return "cyrillic (bulgarian/russian)"
        t = text.lower()
        # strip accents for robust matching of french/spanish keywords
        t_norm = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
        scores = {}
        for name, pats in hints.items():
            s = sum(t_norm.count(p) for p in pats)
            if s > 0:
                scores[name] = s
        if not scores:
            return "other/unknown"
        return max(scores, key=scores.get)

    langs = train.loc[has_report, "Report"].map(detect_lang)
    lang_counts = Counter(langs)
    emit(f"- detected languages (keyword-based, keyword counts):")
    for name, n in lang_counts.most_common():
        emit(f"    {name}: {n} studies ({n/len(langs):.1%})")
    emit()

    # ---------- series structure ----------
    emit("## Series structure")
    per_study = series.groupby("StudyInstanceUID").size()
    emit(f"- series per study: mean={per_study.mean():.2f}, p50={per_study.median():.0f}, min={per_study.min()}, max={per_study.max()}")
    emit(f"- studies in train_series.csv: {series['StudyInstanceUID'].nunique()}")
    emit(f"- studies in train.csv missing from series table: {len(train) - series['StudyInstanceUID'].nunique()}")
    plane_counts = series["Anatomical_Plane"].value_counts(dropna=False)
    emit(f"- plane counts: {plane_counts.to_dict()}")
    fluid_counts = series["Fluid_Sensitive"].value_counts(dropna=False)
    emit(f"- fluid-sensitive counts: {fluid_counts.to_dict()}")
    fs_fp = pd.crosstab(series["Fluid_Sensitive"], series["Fat_Suppression"])
    emit(f"- Fluid_Sensitive x Fat_Suppression crosstab:\n{fs_fp}")
    buckets = (
        series.groupby(["Anatomical_Plane", "Fluid_Sensitive"]).size().sort_values(ascending=False)
    )
    emit("- bucket sizes (plane, fluid):")
    for (pl, fl), n in buckets.items():
        emit(f"    {pl:<10} fluid={fl}: {n}")
    emit()

    # ---------- series-selection feasibility ----------
    emit("## Priority-bucket coverage per study")
    priorities = [
        ("Sagittal", 1), ("Coronal", 1), ("Axial", 1),
        ("Sagittal", 0), ("Coronal", 0), ("Axial", 0),
    ]
    have = series.groupby(["Anatomical_Plane", "Fluid_Sensitive"])["StudyInstanceUID"].agg(set).to_dict()
    studies = series["StudyInstanceUID"].unique()
    cover = {p: 0 for p in priorities}
    cover_at_least = {n: 0 for n in range(1, 4)}
    for s in studies:
        got = 0
        for p in priorities:
            if s in have.get(p, set()):
                cover[p] += 1
                if got < 3:
                    got += 1
        cover_at_least[got] += 1
    for p in priorities:
        emit(f"    {p[0]:<10} fluid={p[1]}: {cover[p]} studies ({cover[p]/len(studies):.1%})")
    emit(f"    -> with 3-series selection: {cover_at_least[3]} studies fill all 3 slots "
         f"({cover_at_least[3]/len(studies):.1%}); distribution of filled slots: {cover_at_least}")
    emit()

    # ---------- test example ----------
    ts = pd.read_csv(os.path.join(DATA_DIR, "test_series.csv"))
    emit("## Example test studies")
    emit(f"- test_series.csv rows: {len(ts)}, studies: {ts['StudyInstanceUID'].nunique()}")
    emit(f"- planes available per test study: {ts.groupby('StudyInstanceUID')['Anatomical_Plane'].apply(list).to_dict()}")
    emit()

    with open(os.path.join(OUT_DIR, "00-recon-data.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {os.path.join(OUT_DIR, '00-recon-data.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
