"""Centralized config loader for the knee package.

Reads config.yaml from repo root (or /kaggle/working on Kaggle).
All modules should import config instead of hardcoding values.

Usage:
    from knee.config import CFG
    data_dir = CFG["paths"]["local_data"]
"""

import os
from typing import Any, Dict

import yaml

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CFG: Dict[str, Any] = {}


def _find_config() -> str:
    """Locate config.yaml: repo root first, then /kaggle/working."""
    candidates = [
        os.path.join(_REPO_ROOT, "config.yaml"),
        "/kaggle/working/config.yaml",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"config.yaml not found in {_REPO_ROOT} or /kaggle/working"
    )


def load_config(path: str = None) -> Dict[str, Any]:
    """Load and cache the YAML config. Repeated calls return the same dict."""
    global _CFG
    if _CFG:
        return _CFG
    if path is None:
        path = _find_config()
    with open(path, encoding="utf-8") as f:
        _CFG = yaml.safe_load(f)
    return _CFG


def get(*keys: str, default: Any = None) -> Any:
    """Nested key access: get('paths', 'local_data')."""
    cfg = load_config()
    for k in keys:
        if isinstance(cfg, dict):
            cfg = cfg.get(k, default)
        else:
            return default
    return cfg


# convenience: load on import
CFG = load_config
