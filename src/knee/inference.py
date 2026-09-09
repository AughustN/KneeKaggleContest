"""Inference pipeline for the knee challenge.

Usage:
    from knee.inference import load_checkpoint, predict, write_submission

    model, cfg = load_checkpoint("best_model.pt")
    probs = predict(model, tensors, batch_size=32, device="cuda")
    write_submission(probs, study_uids, "submission.csv")
"""

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from knee.constants import LABELS


def load_checkpoint(path: str, device: str = None) -> Tuple[nn.Module, dict]:
    """Load a checkpoint saved by Person A and return (model, config).

    Checkpoint format (saved by A):
        torch.save({
            "model_state": model.module.state_dict(),
            "config": {"img_size": 224, "num_slices": 8, "max_series": 3},
            "label_order": constants.LABELS,
            ...
        })
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    # Build model arch — import here so device is known
    from knee.model import KneeModel
    model = KneeModel(cfg=cfg)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, cfg


@torch.no_grad()
def predict(
    model: nn.Module,
    tensors: np.ndarray,
    batch_size: int = 32,
    device: str = None,
) -> np.ndarray:
    """Run inference on preprocessed tensors.

    Args:
        model: loaded KneeModel
        tensors: (N, 24, 3, 224, 224) float32 (after prep_tensor)
        batch_size: inference batch size
        device: "cuda" or "cpu"

    Returns:
        probs: (N, 12) float32 in [0, 1]
    """
    if device is None:
        device = next(model.parameters()).device

    N = tensors.shape[0]
    all_probs = np.zeros((N, len(LABELS)), dtype=np.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = torch.from_numpy(tensors[start:end]).to(device)
        logits = model(batch)  # (B, 12)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs[start:end] = probs

    return all_probs


def write_submission(
    probs: np.ndarray,
    study_uids: List[str],
    out_path: str,
    label_order: List[str] = None,
) -> str:
    """Write submission.csv in the exact format expected by Kaggle.

    Args:
        probs: (N, 12) probabilities
        study_uids: list of StudyInstanceUID, length N
        out_path: output CSV path
        label_order: column order (defaults to knee.constants.LABELS)

    Returns:
        path to the written file
    """
    if label_order is None:
        label_order = LABELS

    assert len(probs) == len(study_uids), \
        f"probs ({len(probs)}) and uids ({len(study_uids)}) must match"
    assert probs.shape[1] == len(label_order), \
        f"probs has {probs.shape[1]} columns, expected {len(label_order)}"

    # Clip to [0, 1] and replace NaN
    probs = np.clip(probs, 0.0, 1.0)
    probs = np.nan_to_num(probs, nan=0.5)

    df = pd.DataFrame(probs, columns=label_order)
    df.insert(0, "StudyInstanceUID", study_uids)
    df.to_csv(out_path, index=False)
    return out_path
