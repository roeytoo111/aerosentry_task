#!/usr/bin/env python3
"""Split an image dataset into Train/Val/Test without temporal leakage across sequences.

Images are grouped by a derived *sequence ID* (e.g. ``videoID`` from ``videoID_frameN``).
Every file in a group lands in exactly one split—never scattered across Train/Val/Test.

Assignment uses a **deterministic, group-level** rule (hash of sequence ID into ratio
bins), not a per-file random draw, so neighbor frames from the same capture stay together
and leakage across time is avoided.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Tuple


IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
)


def extract_sequence_id(filename: str) -> str:
    """Infer sequence / video key from a filename stem.

    Expected patterns include ``videoID_frameNumber`` or Roboflow-style
    ``..._jpg.rf.<hash>`` exports; the last ``_``-separated token is treated as the
    frame or suffix token, everything before it as the sequence ID.

    Args:
        filename: Base name or path (e.g. ``clip3_MP4-0076_jpg.rf.abc.jpg``).

    Returns:
        Stable string key grouping all frames from the same logical sequence.
    """
    stem = Path(filename).stem
    if ".rf." in stem:
        stem = stem.split(".rf.", maxsplit=1)[0]
    parts = stem.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:-1])
    return stem


def iter_image_files(root: Path) -> List[Path]:
    """Collect image paths under ``root`` (non-recursive: one directory level)."""
    paths: List[Path] = []
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(p)
    return paths


def group_by_sequence(paths: Iterable[Path]) -> Dict[str, List[Path]]:
    """Map sequence ID to sorted file paths."""
    buckets: DefaultDict[str, List[Path]] = defaultdict(list)
    for p in paths:
        sid = extract_sequence_id(p.name)
        buckets[sid].append(p)
    for files in buckets.values():
        files.sort(key=lambda x: x.name)
    return dict(buckets)


def _stable_hash_bucket(sequence_id: str, bins: int = 1_000_000) -> int:
    """Map a string to ``[0, bins)`` deterministically."""
    h = hashlib.sha256(sequence_id.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % bins


def assign_split_for_group(
    sequence_id: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> str:
    """Assign *one* split label to a whole sequence using hashed binning.

    Args:
        sequence_id: Group key shared by all frames of a sequence.
        train_ratio: Fraction of the *probability mass* for train (group-level).
        val_ratio: Fraction for validation.
        test_ratio: Fraction for test. Must sum to ~1.0.

    Returns:
        ``"train"``, ``"val"``, or ``"test"``.
    """
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Sum of split ratios must be positive")
    t, v, s = train_ratio / total, val_ratio / total, test_ratio / total
    u = _stable_hash_bucket(sequence_id, bins=1_000_000) / 1_000_000.0
    if u < t:
        return "train"
    if u < t + v:
        return "val"
    return "test"


def copy_groups_to_splits(
    grouped: Dict[str, List[Path]],
    output_root: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    symlink: bool = False,
) -> Tuple[int, int, int]:
    """Materialize train/val/test folders; returns counts per split (files, not groups)."""
    splits = ("train", "val", "test")
    for d in splits:
        (output_root / d).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}
    for seq_id in sorted(grouped.keys()):
        label = assign_split_for_group(seq_id, train_ratio, val_ratio, test_ratio)
        dest_dir = output_root / label
        for src in grouped[seq_id]:
            dest = dest_dir / src.name
            if dest.exists():
                raise FileExistsError(f"Target exists (name collision): {dest}")
            if symlink:
                dest.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dest)
            counts[label] += 1
    return counts["train"], counts["val"], counts["test"]


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Group-wise Train/Val/Test split by video/sequence ID to prevent temporal "
            "leakage (no random per-image assignment)."
        )
    )
    p.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Directory containing raw images (non-recursive).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Root directory where train/, val/, test/ will be created.",
    )
    p.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Target fraction of *groups* for train (hash-based assignment).",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Target fraction of groups for validation.",
    )
    p.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Target fraction of groups for test.",
    )
    p.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink instead of copy (saves disk; paths must remain valid).",
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.source.is_dir():
        print(f"Source is not a directory: {args.source}", file=sys.stderr)
        return 1
    imgs = iter_image_files(args.source)
    if not imgs:
        print(f"No images found in {args.source}", file=sys.stderr)
        return 1

    grouped = group_by_sequence(imgs)
    n_groups = len(grouped)
    n_files = sum(len(v) for v in grouped.values())

    tr, va, te = copy_groups_to_splits(
        grouped,
        args.output,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        symlink=args.symlink,
    )
    print(f"Sequences (groups): {n_groups}, image files: {n_files}")
    print(f"Written — train: {tr}, val: {va}, test: {te} (files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
