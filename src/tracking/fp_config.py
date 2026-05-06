"""Load ``TrackManager`` / ``GeometricEgoMotion`` tuning from YAML (single config file)."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Optional, Type

import yaml

from src.tracking.fp_suppressor import FalsePositiveSuppressor
from src.tracking.geometric_ego_motion import GeometricEgoMotion
from src.tracking.track_manager import TrackManager


def tracking_fp_yaml_default_path(repo_root: Path) -> Path:
    return (repo_root / "config" / "tracking_fp.yaml").resolve()


def resolve_tracking_fp_yaml(
    explicit: Optional[Path],
    *,
    repo_root: Path,
    no_config: bool,
) -> Optional[Path]:
    """Pick YAML path for FP / tracking tuning.

    Returns ``None`` when no file should be loaded (use empty mapping → constructor defaults).
    """
    if no_config:
        return None
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if p.is_file():
            return p
        print(f"[tracking_fp] config file not found (using code defaults): {p}", flush=True)
        return None
    cand = tracking_fp_yaml_default_path(repo_root)
    return cand if cand.is_file() else None


def load_tracking_fp_yaml_dict(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        print(f"[tracking_fp] expected mapping at root, got {type(raw).__name__}", flush=True)
        return {}
    return dict(raw)


def _kwargs_for_init(cls: Type[Any], raw: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    out: Dict[str, Any] = {}
    for k, v in raw.items():
        if k not in params or k == "self":
            continue
        p = params[k]
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        out[k] = v
    return out


def _nested_mapping(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    v = cfg.get(key)
    return dict(v) if isinstance(v, dict) else {}


def build_false_positive_suppressor_from_mapping(
    cfg: Dict[str, Any],
    *,
    geo_only: bool,
) -> FalsePositiveSuppressor:
    """Build suppressor from a config dict (e.g. YAML root). Empty ``cfg`` → code defaults."""
    tm_kw = _kwargs_for_init(TrackManager, _nested_mapping(cfg, "track_manager"))
    geo_kw = _kwargs_for_init(GeometricEgoMotion, _nested_mapping(cfg, "geometric_ego_motion"))
    fp_sec = _nested_mapping(cfg, "fp_suppressor")
    emit_unc = bool(fp_sec.get("emit_unconfirmed_tracks", False))
    geo = GeometricEgoMotion(**geo_kw)
    if geo_only:
        return FalsePositiveSuppressor(geo_estimator=geo, geo_only=True)
    tm = TrackManager(**tm_kw)
    return FalsePositiveSuppressor(
        track_manager=tm,
        geo_estimator=geo,
        emit_unconfirmed_tracks=emit_unc,
    )


__all__ = [
    "build_false_positive_suppressor_from_mapping",
    "load_tracking_fp_yaml_dict",
    "resolve_tracking_fp_yaml",
    "tracking_fp_yaml_default_path",
]
