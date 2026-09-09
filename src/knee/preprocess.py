"""Shared preprocessing logic for the knee challenge (single source of truth).

Used by:
- notebooks/10-preprocess.py   (Kaggle one-pass: train corpus -> npz shards)
- notebooks/40-infer.py        (test-time preprocessing, efficiency-tuned)
- local synthetic tests

Constants (from Phase 0 recon):
- 3 series per study by priority: sag-fluid > cor-fluid > ax-fluid,
  fallbacks sag-nonfluid > cor-nonfluid > ax-nonfluid  (100% coverage verified)
- 8 slices per series, uniformly sampled, sorted by ImagePositionPatient[2]
  (100% present in recon; InstanceNumber fallback)
- 224x224 uint8 (percentile-normalized per volume: p1/p99.5 clip -> scale)
- per-volume normalization handles the mixed uint16/int16, 256-1024 matrices
"""

import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pydicom
except ImportError as e:  # pragma: no cover
    raise ImportError("pydicom required") from e

# --- load from config.yaml (with hardcoded fallbacks for Kaggle/embedded use) ---
try:
    from knee.config import get as cfg
    IMG_SIZE = cfg("preprocessing", "img_size", default=224)
    NUM_SLICES = cfg("preprocessing", "num_slices", default=8)
    MAX_SERIES = cfg("preprocessing", "max_series", default=3)
    P_LO = cfg("preprocessing", "percentile_low", default=1.0)
    P_HI = cfg("preprocessing", "percentile_high", default=99.5)
    _SERIES_PRIORITY = [
        tuple(p) for p in cfg("preprocessing", "series_priority", default=[
            ["Sagittal", 1], ["Coronal", 1], ["Axial", 1],
            ["Sagittal", 0], ["Coronal", 0], ["Axial", 0],
        ])
    ]
except Exception:
    IMG_SIZE = 224
    NUM_SLICES = 8
    MAX_SERIES = 3
    P_LO = 1.0
    P_HI = 99.5
    _SERIES_PRIORITY = [
        ("Sagittal", 1), ("Coronal", 1), ("Axial", 1),
        ("Sagittal", 0), ("Coronal", 0), ("Axial", 0),
    ]


def find_data_dir(base: str = None) -> str:
    """Locate the competition dir; handles nested /kaggle/input/competitions/<slug>/.

    Bounded-depth walk; prunes train_series/test_series DICOM trees.
    """
    if base is None:
        try:
            from knee.config import get as cfg
            base = cfg("paths", "kaggle_input", default="/kaggle/input")
        except Exception:
            base = "/kaggle/input"
    if not os.path.isdir(base):
        raise FileNotFoundError(f"{base} does not exist — attach the competition dataset")
    for root, dirs, files in os.walk(base):
        if "train.csv" in files or "test.csv" in files:
            return root
        depth = root[len(base):].count(os.sep) + (1 if root != base else 0)
        if depth >= 3:
            dirs[:] = []
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
    raise FileNotFoundError("competition data not found under " + base)


def select_series(study_series_rows, max_series: int = MAX_SERIES) -> List[str]:
    """Pick up to max_series SeriesInstanceUIDs by plane/fluid priority.

    `study_series_rows`: iterable of dicts/Series-like with keys
    Anatomical_Plane, Fluid_Sensitive, SeriesInstanceUID.
    Deterministic (sorted UIDs) so train/test selections match.
    """
    by_bucket: Dict[Tuple[str, int], List[str]] = {}
    for row in study_series_rows:
        plane = str(row["Anatomical_Plane"])
        fluid = int(row["Fluid_Sensitive"])
        by_bucket.setdefault((plane, fluid), []).append(str(row["SeriesInstanceUID"]))

    chosen: List[str] = []
    for bucket in _SERIES_PRIORITY:
        cands = by_bucket.get(bucket, [])
        if cands:
            chosen.append(sorted(cands)[0])
        if len(chosen) >= max_series:
            break
    # fill any remaining slots deterministically
    for uid in sorted(u for cands in by_bucket.values() for u in cands):
        if uid not in chosen and len(chosen) < max_series:
            chosen.append(uid)
    return chosen


def _read_slice_position(path: str) -> float:
    """Header-only read for physical Z position; ~1ms. Falls back to 0.0."""
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            return float(ipp[2])
        num = getattr(ds, "InstanceNumber", None)
        if num is not None:
            return float(num)
    except Exception:
        pass
    return 0.0


def _slice_sort_key(path: str) -> Tuple[float, str]:
    """Header-only read for position; ~1ms. Falls back to filename."""
    return (_read_slice_position(path), path)


def _select_indices_by_position(
    n: int, num_slices: int, positions: List[float]
) -> List[int]:
    """Select slice indices with uniform physical spacing.

    Samples evenly across the physical Z-axis span instead of file-count,
    giving better anatomical coverage when slices are non-uniformly spaced.

    Falls back to index-based uniform sampling when positions are unavailable
    (all zeros) or when the series has fewer slices than requested.
    """
    if n <= num_slices:
        return list(range(n))

    # check if physical positions are meaningful
    span = positions[-1] - positions[0]
    if abs(span) < 1e-6:
        # no physical spacing info — fall back to index-based
        return sorted(set(int(i) for i in np.linspace(0, n - 1, num_slices)))

    # sample uniformly across physical span
    start, stop = positions[0], positions[-1]
    targets = np.linspace(start, stop, num_slices)

    idx = []
    j = 0
    for t in targets:
        # advance pointer to nearest unchosen slice
        best = j
        best_dist = abs(positions[j] - t)
        while j < n - 1:
            d = abs(positions[j + 1] - t)
            if d < best_dist:
                j += 1
                best = j
                best_dist = d
            else:
                break
        if best not in idx:
            idx.append(best)
        else:
            # if duplicate, try neighbours
            for offset in range(1, n):
                for cand in (best - offset, best + offset):
                    if 0 <= cand < n and cand not in idx:
                        idx.append(cand)
                        break
                if len(idx) > len(targets):
                    break

    # pad if dedup shrank below num_slices
    j = 0
    while len(idx) < num_slices and j < n:
        if j not in idx:
            idx.append(j)
        j += 1

    return sorted(idx[:num_slices])


_DCM_RE = re.compile(r"\.dcm$", re.IGNORECASE)


def read_series(
    series_dir: str,
    num_slices: int = NUM_SLICES,
    img_size: int = IMG_SIZE,
    header_only_first: bool = False,
) -> Optional[np.ndarray]:
    """Read one DICOM series -> (num_slices, img_size, img_size) uint8.

    Efficiency: header-scans ALL files ONCE (cheap) to sort + collect physical
    positions, then pixel-decodes ONLY the sampled slices.

    Returns None on unreadable/empty series (caller zero-fills).
    """
    try:
        files = [os.path.join(series_dir, f) for f in os.listdir(series_dir)
                 if _DCM_RE.search(f)]
    except OSError:
        return None
    if not files:
        return None

    # --- single-pass header scan: read position + sort ---
    entries = []  # [(position, path)]
    for p in files:
        entries.append((_read_slice_position(p), p))
    entries.sort(key=lambda e: e[0])

    n = len(entries)
    positions = [e[0] for e in entries]
    sorted_files = [e[1] for e in entries]
    idx = _select_indices_by_position(n, num_slices, positions)

    # --- pixel-decode only the selected slices ---
    slices = []
    for i in idx:
        try:
            ds = pydicom.dcmread(sorted_files[i], force=True)
            img = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
            inter = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
            if slope != 1.0 or inter != 0.0:
                img = img * slope + inter
            slices.append(img)
        except Exception:
            continue
    if not slices:
        return None

    # pad to num_slices by repeating last slice
    while len(slices) < num_slices:
        slices.append(slices[-1])

    vol = np.stack(slices)  # (S, H, W) float32

    # per-volume percentile normalization -> uint8
    finite = vol[np.isfinite(vol)]
    if finite.size == 0:
        return None
    p_lo, p_hi = np.percentile(finite, [P_LO, P_HI])
    if p_hi <= p_lo:
        p_hi = p_lo + 1e-6
    vol = np.clip(vol, p_lo, p_hi)
    vol = (vol - p_lo) / (p_hi - p_lo)

    # resize
    try:
        import cv2
        out = np.empty((vol.shape[0], img_size, img_size), dtype=np.uint8)
        for k in range(vol.shape[0]):
            r = cv2.resize(vol[k], (img_size, img_size), interpolation=cv2.INTER_AREA)
            out[k] = np.clip(np.rint(r * 255.0), 0, 255).astype(np.uint8)
        return out
    except ImportError:
        # PIL fallback (Kaggle always has cv2, local may not)
        from PIL import Image
        out = np.empty((vol.shape[0], img_size, img_size), dtype=np.uint8)
        for k in range(vol.shape[0]):
            im = Image.fromarray((vol[k] * 255).astype(np.uint8))
            out[k] = np.array(im.resize((img_size, img_size), Image.BILINEAR))
        return out


def study_tensor(
    study_dir: str,
    series_uids: List[str],
    num_slices: int = NUM_SLICES,
    img_size: int = IMG_SIZE,
) -> np.ndarray:
    """Stack selected series -> (MAX_SERIES, NUM_SLICES, H, W) uint8.

    Missing/unreadable series are zero-filled so the shape is fixed.
    """
    out = np.zeros((MAX_SERIES, num_slices, img_size, img_size), dtype=np.uint8)
    for k, suid in enumerate(series_uids[:MAX_SERIES]):
        sdir = os.path.join(study_dir, suid)
        t = read_series(sdir, num_slices, img_size)
        if t is not None:
            out[k] = t
    return out


def process_study(args):
    """Multiprocessing worker: (uid, series_uids, series_root) -> result tuple.

    Kept at module scope (picklable). Returns
    (uid, ok, tensor) — tensor is zero-filled (MAX_SERIES, NUM_SLICES, H, W)
    on failure so downstream stacking never breaks.
    """
    uid, series_uids, series_root = args
    try:
        t = study_tensor(os.path.join(series_root, uid), series_uids)
        return uid, True, t
    except Exception:
        return uid, False, np.zeros(
            (MAX_SERIES, NUM_SLICES, IMG_SIZE, IMG_SIZE), np.uint8
        )
