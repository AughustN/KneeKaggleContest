"""Phase 0 recon — KAGGLE notebook version.

Run on Kaggle with the competition dataset attached. Complements the local
CSV recon (reports/00-recon-data.md) with DICOM-level checks:

  1. Transfer syntax distribution across a sample of series.
  2. Decoding works for all syntaxes with pydicom (+pylibjpeg/GDCM fallbacks).
  3. Slice counts per series, per bucket (validates 8-slice sampling).
  4. Pixel shapes/dtypes, spacing variation (validates resize-to-224).
  5. Allowlisted metadata sanity: ImagePositionPatient[2] present? InstanceNumber?
     (slice sorting keys).
  6. Timing: decode speed per series -> Phase 1 preprocessing ETA + Phase 4
     inference runtime estimate (Efficiency Prize).
  7. Report language quick estimate (fraction of non-ASCII reports etc.).
"""

import os
import random
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/input"  # auto-resolved below to the comp dir
LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
OUT = "/kaggle/working"


def find_data_dir():
    """Robust resolver: competition data may mount at /kaggle/input/<slug>/
    (standard) or /kaggle/input/competitions/<slug>/ (nested layout).

    Bounded-depth walk; prunes the giant DICOM trees (train_series/,
    test_series/) so it never scans the 569GB corpus.
    """
    base = "/kaggle/input"
    if not os.path.isdir(base):
        raise FileNotFoundError(f"{base} does not exist — attach the competition dataset")
    for root, dirs, files in os.walk(base):
        if "train.csv" in files:
            return root
        # bound walk depth to 3 levels below base (never scan DICOM trees)
        depth = root[len(base):].count(os.sep) + (1 if root != base else 0)
        if depth >= 3:
            dirs[:] = []
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
    raise FileNotFoundError(
        "train.csv not found under /kaggle/input — check that the competition "
        "dataset is attached and rules accepted"
    )


def main():
    random.seed(0)
    np.random.seed(0)
    data_dir = find_data_dir()
    print("data dir:", data_dir)
    train_series = pd.read_csv(os.path.join(data_dir, "train_series.csv"))
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    studies = sorted(train_series["StudyInstanceUID"].unique())
    print(f"{len(studies)} studies")

    # ---------------- install/verify decoders ----------------
    import pydicom
    print("pydicom", pydicom.__version__)
    try:
        import pylibjpeg  # noqa

        print("pylibjpeg OK")
    except ImportError:
        print("pylibjpeg MISSING — attach pylibjpeg wheels dataset (offline comp)")
    try:
        import gdcm  # noqa

        print("python-gdcm OK")
    except ImportError:
        print("python-gdcm missing (optional fallback)")

    # ---------------- sample series for DICOM stats ----------------
    rng = np.random.default_rng(0)
    sample_series = train_series.sample(n=min(300, len(train_series)), random_state=0)
    syntax_counter = Counter()
    slice_counts = defaultdict(list)  
    pixel_shapes = Counter()
    dtypes = Counter()
    have_ipp = Counter()
    have_instnum = Counter()
    decode_ms = []
    failures = Counter()

    for i, row in enumerate(sample_series.itertuples(index=False)):
        sdir = os.path.join(
            data_dir, "train_series", row.StudyInstanceUID, row.SeriesInstanceUID
        )
        if not os.path.isdir(sdir):
            failures["missing_dir"] += 1
            continue
        files = sorted(f for f in os.listdir(sdir) if f.lower().endswith(".dcm"))
        bucket = (row.Anatomical_Plane, int(row.Fluid_Sensitive))
        slice_counts[bucket].append(len(files))

        # read one file fully for metadata + pixel decode timing
        f0 = os.path.join(sdir, files[0])
        try:
            ds = pydicom.dcmread(f0)
            syntax_counter[str(ds.file_meta.TransferSyntaxUID)] += 1
            pixel_shapes[ds.pixel_array.shape] += 1
            dtypes[str(ds.pixel_array.dtype)] += 1
            have_ipp[hasattr(ds, "ImagePositionPatient")] += 1
            have_instnum[hasattr(ds, "InstanceNumber")] += 1
        except Exception as e:  # noqa
            failures[f"read:{type(e).__name__}"] += 1
            continue

        # decode whole small series for timing (cap at 45 files)
        t0 = time.perf_counter()
        for f in files[:45]:
            try:
                _ = pydicom.dcmread(os.path.join(sdir, f)).pixel_array
            except Exception as e:  # noqa
                failures[f"decode:{type(e).__name__}"] += 1
        dt = time.perf_counter() - t0
        decode_ms.append(dt / min(len(files), 45) * 1000)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(sample_series)} series scanned...")

    print("\n## Transfer syntaxes")
    for k, v in syntax_counter.most_common():
        print(f"    {k}: {v}")
    print("\n## Failures")
    print("   ", dict(failures) or "none")
    print("\n## Pixel shapes (top 10)")
    for k, v in pixel_shapes.most_common(10):
        print(f"    {k}: {v}")
    print("\n## dtypes", dict(dtypes))
    print("\n## ImagePositionPatient present:", dict(have_ipp))
    print("## InstanceNumber present:", dict(have_instnum))

    print("\n## Slice counts per bucket (n, mean, p50, p95, max)")
    for bucket, counts in sorted(slice_counts.items()):
        c = np.array(counts)
        print(
            f"    {bucket[0]:<9} fluid={bucket[1]}: n={len(c)}, "
            f"mean={c.mean():.1f}, p50={np.median(c):.0f}, "
            f"p95={np.percentile(c, 95):.0f}, max={c.max()}"
        )

    print("\n## Decode speed")
    dms = np.array(decode_ms)
    print(f"    per-slice decode: mean={dms.mean():.1f} ms, p95={np.percentile(dms,95):.1f} ms")
    # Phase 4 runtime estimate: ~1300 test studies x 3 series x ~30 slices
    est_s = dms.mean() / 1000 * 1300 * 3 * 30
    print(f"    -> naive full-decode inference estimate: {est_s/60:.1f} min (must be optimized)")
    est_p1 = dms.mean() / 1000 * 4407 * 3 * 30
    print(f"    -> Phase 1 preprocessing (3 series x all slices): ~{est_p1/3600:.1f} h one-pass")

    # ---------------- report language quick stats ----------------
    reports = train["Report"].fillna("").astype(str)
    nonascii = reports.apply(lambda s: any(ord(ch) > 127 for ch in s))
    print("\n## Reports")
    print(f"    non-ASCII fraction (non-English hint): {nonascii.mean():.1%}")

    # ---------------- save outputs ----------------
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "recon-dicom.txt"), "w", encoding="utf-8") as f:
        f.write("see notebook output; summary stats above\n")
    print("\nDone.")


if __name__ == "__main__":
    main()
