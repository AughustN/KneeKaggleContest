# Phase 0 recon — DICOM profile (Kaggle, 300 sampled series)

Source: notebooks/00-recon-kaggle.py, run 2026-09-05. Data dir:
`/kaggle/input/competitions/rsna-knee-abnormality-detection` (nested mount layout).

## Findings

### Transfer syntax
- ALL 300 sampled series: `1.2.840.10008.1.2.1` (Explicit VR Little Endian, uncompressed)
- Zero decode failures with pydicom 3.0.2 alone (pylibjpeg NOT installed)
- BUT: competition docs promise JPEG Lossless / JPEG 2000 / Implicit VR also exist.
  300/24,371 series ≈ 1.2% sample — compressed syntaxes may be rare but present.
  -> Phase 1 step 0: full-corpus header-only syntax scan (cheap, ~ms/file, no pixel decode)
  -> attach pylibjpeg + openjpeg wheels as a Kaggle dataset regardless (insurance;
     one unhandled syntax = crashed preprocessing job)

### Pixel data
- Square matrices, highly variable: 256…1024 (top: 512, 384, 640, 1024, 320)
- dtypes: uint16 (75%), int16 (25%)
- -> resize-to-224 confirmed; percentile normalization handles signed/unsigned;
     per-volume stats (not global) required

### Slice sorting keys (critical)
- ImagePositionPatient: 300/300 present
- InstanceNumber: 300/300 present
- -> sort by IPP[2] (fall back to InstanceNumber) is reliable corpus-wide

### Slice counts per bucket (n, mean, p50, p95, max)
| bucket | n | mean | p50 | p95 | max |
|---|---|---|---|---|---|
| Axial fluid=0 | 15 | 43.3 | 30 | 92 | 200 |
| Axial fluid=1 | 65 | 38.0 | 31 | 112 | 160 |
| Coronal fluid=0 | 58 | 28.3 | 29 | 36 | 39 |
| Coronal fluid=1 | 50 | 29.9 | 31 | 39 | 42 |
| Sagittal fluid=0 | 64 | 31.3 | 30 | 47 | 70 |
| Sagittal fluid=1 | 48 | 36.7 | 28 | 43 | 320 |
- -> 8-slice uniform sampling validated (p50 ≈ 30 slices); long tails handled by sampling

### Decode speed
- 10.1 ms/slice mean, 18.7 ms p95 (single-threaded, uncompressed)
- Phase 1 one-pass estimate (3 series x all slices, 4407 studies): ~1.1 h -> fine in 12h session
- Naive inference (1300 studies x 3 series x 30 slices): ~19.6 min -> too slow for Efficiency Prize

## EFFICIENCY PRIZE LEVER (important)
DICOM here is one-file-per-slice (not multi-frame tiled), so:
1. Read headers only (stop_before_pixels=True, ~1 ms/file) for ALL slices -> sort positions
2. Pixel-decode ONLY the 8 sampled slices (skip the other ~22)
   => ~3.75x fewer decodes: 117k -> 31k slices for inference ≈ 5.3 min single-thread
3. Multiprocessing (4 workers) -> ~1.5-2 min total decode
Header scan of full corpus also yields the transfer-syntax census for free.

## Reports
- non-ASCII fraction 59.3% (consistent with 60.7% non-English languages found in CSV recon)

## Phase 0 conclusions — plan constants now locked
- Series selection: 3 priority slots, Fluid_Sensitive flag (keep Fat_Suppression separate) — valid
- Slice sampling: 8 slices/series, sorted by IPP[2] — valid
- Image size: 224x224 (divisible by DINOv2 patch 14) — valid
- Preprocessing budget: ~1.1 h one-pass on Kaggle CPU — acceptable
- Phase 4 inference: header-scan + selective decode + multiprocessing — target ~2 min decode
- Phase 2 (text) needs NO images — can start locally immediately with train.csv
