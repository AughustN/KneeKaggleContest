"""Local end-to-end test of the inference pipeline (B4).

Creates synthetic DICOMs + fake checkpoint, runs inference, verifies:
- probs shape (N, 12)
- columns match sample_submission (= LABELS)
- values in [0, 1], no NaN
- studies with missing series still get a row in submission
"""

import os
import sys
import shutil
import tempfile

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import torch

# Setup paths
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC_DIR)

DATA_ROOT = os.path.join(tempfile.gettempdir(), "knee_infer_test")
N_STUDIES = 6

BUCKETS = [
    ("Sagittal", 1, 32), ("Coronal", 1, 30), ("Axial", 1, 40),
    ("Sagittal", 0, 24), ("Coronal", 0, 28),
]


def make_dcm(path, pixel, pos, dtype, slope=1.0, intercept=0.0):
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    ds.SOPInstanceUID = generate_uid()
    ds.ImagePositionPatient = [0.0, 0.0, float(pos)]
    ds.InstanceNumber = int(pos)
    ds.RescaleSlope = slope
    ds.RescaleIntercept = intercept
    ds.Rows, ds.Columns = pixel.shape
    arr = pixel.astype(dtype)
    ds.PixelData = arr.tobytes()
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1 if dtype == np.int16 else 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    pydicom.dcmwrite(path, ds, enforce_file_format=True)


def build_corpus():
    if os.path.exists(DATA_ROOT):
        shutil.rmtree(DATA_ROOT)
    os.makedirs(DATA_ROOT, exist_ok=True)

    rng = np.random.default_rng(42)
    studies, series_rows = [], []

    for s in range(N_STUDIES):
        suid = f"1.2.826.0.1.test.study.{s:04d}"
        buckets = [b for i, b in enumerate(BUCKETS) if i != s % len(BUCKETS)]
        for b_i, (plane, fluid, n_sl) in enumerate(buckets):
            seruid = f"1.2.826.0.1.test.series.{s:04d}.{b_i}"
            sdir = os.path.join(DATA_ROOT, "test_series", suid, seruid)
            os.makedirs(sdir, exist_ok=True)
            size = int(rng.choice([256, 320, 384]))
            dtype = np.int16 if (s + b_i) % 2 else np.uint16
            slope = 2.0 if (s + b_i) % 3 == 0 else 1.0
            intercept = -1024.0 if (s + b_i) % 4 == 0 else 0.0
            base = rng.random((size, size)) * 200
            positions = list(rng.permutation(n_sl))
            for sl in range(n_sl):
                img = base + positions[sl] * 1.5
                fname = f"im_{rng.integers(0, 10**6):06d}.dcm"
                make_dcm(os.path.join(sdir, fname), img, positions[sl], dtype,
                         slope, intercept)
            series_rows.append({
                "StudyInstanceUID": suid, "SeriesInstanceUID": seruid,
                "Fluid_Sensitive": fluid, "Anatomical_Plane": plane,
            })
        studies.append({"StudyInstanceUID": suid})

    # Study with NO series at all (edge case)
    suid_missing = "1.2.826.0.1.test.study.9999"
    studies.append({"StudyInstanceUID": suid_missing})

    pd.DataFrame(studies).to_csv(os.path.join(DATA_ROOT, "test.csv"), index=False)
    pd.DataFrame(series_rows).to_csv(
        os.path.join(DATA_ROOT, "test_series.csv"), index=False)

    return studies


def create_fake_checkpoint():
    """Create a fake best_model.pt matching A's contract format.

    Builds a minimal model inline (no import from knee.model) so the test
    can run before A provides the real model.py.
    """
    from knee.dinov2 import vit_small
    import torch.nn as nn

    # Minimal model matching contract architecture:
    # input (B, 24, 3, 224, 224) -> DINOv2 -> CLS 384-d -> mean slices -> Linear -> (B, 12)
    backbone = vit_small(patch_size=14, img_size=224)
    embed_dim = backbone.embed_dim  # 384
    head = nn.Sequential(nn.Dropout(0.3), nn.Linear(embed_dim * 3, 12))

    # Combine into a simple container for state_dict
    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = head
        def forward(self, x):
            B = x.shape[0]
            x = x.view(B * 3 * 8, 3, 224, 224)
            with torch.no_grad():
                feats = self.backbone(x)
            feats = feats.view(B, 3, 8, -1).mean(dim=2).view(B, -1)
            return self.head(feats)

    model = FakeModel()
    ckpt = {
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "config": {"img_size": 224, "num_slices": 8, "max_series": 3},
        "label_order": [
            "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
            "Medial OA", "Lateral OA", "PF OA",
            "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
        ],
        "val_auc": 0.5,
        "epoch": 0,
    }
    path = os.path.join(DATA_ROOT, "best_model.pt")
    torch.save(ckpt, path)
    print(f"fake checkpoint: {path}")
    return ckpt, model


def main():
    print("=== B4: Inference local test ===\n")

    # 1. Build synthetic corpus
    studies = build_corpus()
    print(f"corpus: {DATA_ROOT} ({len(studies)} studies)")

    # 2. Create fake checkpoint + model (no import from knee.model)
    ckpt, model = create_fake_checkpoint()
    print(f"fake checkpoint + model ready")

    # 3. Patch find_data_dir to use our test directory
    import knee.preprocess as pp
    _orig_find = pp.find_data_dir
    pp.find_data_dir = lambda *a, **kw: DATA_ROOT

    # 4. Run inference pipeline
    from knee.inference import predict, write_submission
    from knee.preprocess import IMG_SIZE, MAX_SERIES, NUM_SLICES, select_series
    from multiprocessing import Pool

    # Inline prep_tensor (matches knee.dataset.prep_tensor spec from contract)
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3, 1, 1)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3, 1, 1)

    def prep_tensor(vol_uint8):
        """Convert (N, 3, 8, 224, 224) uint8 -> (N, 24, 3, 224, 224) float32."""
        N, S, C, H, W = vol_uint8.shape
        x = vol_uint8.reshape(N, S * C, H, W).astype(np.float32) / 255.0
        x = np.stack([x, x, x], axis=2)  # replicate grayscale to 3 channels
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x.astype(np.float32)

    test_df = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))
    series_df = pd.read_csv(os.path.join(DATA_ROOT, "test_series.csv"))
    study_series = series_df.groupby("StudyInstanceUID")

    # Preprocess
    tasks = []
    for uid in test_df["StudyInstanceUID"]:
        if uid in study_series.groups:
            rows = study_series.get_group(uid).to_dict("records")
            uids = select_series(rows)
        else:
            uids = []
        tasks.append((uid, uids, os.path.join(DATA_ROOT, "test_series")))

    from knee.preprocess import process_study
    raw_tensors = {}
    with Pool(2) as pool:
        for uid, ok, tensor in pool.imap(process_study, tasks, chunksize=2):
            raw_tensors[uid] = tensor

    uids_ordered = list(test_df["StudyInstanceUID"])
    tensors = np.stack([raw_tensors[u] for u in uids_ordered])
    tensors = prep_tensor(tensors)
    print(f"tensors: {tensors.shape}, {tensors.dtype}")

    # Predict
    probs = predict(model, tensors, batch_size=16)
    print(f"probs: {probs.shape}, range [{probs.min():.4f}, {probs.max():.4f}]")

    # Write submission
    out_path = os.path.join(DATA_ROOT, "submission.csv")
    write_submission(probs, uids_ordered, out_path)
    print(f"submission: {out_path}")

    # 5. Assertions
    from knee.constants import LABELS

    df = pd.read_csv(out_path)
    print(f"\nsubmission shape: {df.shape}")
    print(f"columns: {list(df.columns)}")

    # Check 1: correct columns
    expected_cols = ["StudyInstanceUID"] + LABELS
    assert list(df.columns) == expected_cols, f"column mismatch: {list(df.columns)}"
    print("PASS: columns match")

    # Check 2: correct number of rows (all studies including missing series)
    assert len(df) == len(studies), f"row count: {len(df)} != {len(studies)}"
    print(f"PASS: {len(df)} rows (all {len(studies)} studies present)")

    # Check 3: all UIDs from test.csv are present
    test_uids = set(test_df["StudyInstanceUID"])
    sub_uids = set(df["StudyInstanceUID"])
    assert test_uids == sub_uids, f"UID mismatch: missing={test_uids-sub_uids}"
    print("PASS: all study UIDs present")

    # Check 4: probabilities in [0, 1]
    prob_cols = df[LABELS].values
    assert prob_cols.min() >= 0.0, f"min prob < 0: {prob_cols.min()}"
    assert prob_cols.max() <= 1.0, f"max prob > 1: {prob_cols.max()}"
    print("PASS: all probs in [0, 1]")

    # Check 5: no NaN
    assert not df.isna().any().any(), "NaN found in submission"
    print("PASS: no NaN")

    # Check 6: study with missing series has a row (zero-filled)
    missing_row = df[df["StudyInstanceUID"] == "1.2.826.0.1.test.study.9999"]
    assert len(missing_row) == 1, f"missing-series study: {len(missing_row)} rows"
    print("PASS: missing-series study has a row")

    # Cleanup
    pp.find_data_dir = _orig_find

    print("\n=== ALL B4 CHECKS PASSED ===")


if __name__ == "__main__":
    main()
