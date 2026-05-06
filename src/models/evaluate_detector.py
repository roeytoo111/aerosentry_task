"""Offline evaluation with threshold-swept metrics (beyond mAP-only summaries)."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO


@dataclass
class ImageEvalArrays:
    """Per-image tensors for matching (XYXY pixel coordinates, 0-based class ids)."""

    gt_xyxy: np.ndarray
    gt_cls: np.ndarray
    pred_xyxy: np.ndarray
    pred_cls: np.ndarray
    pred_conf: np.ndarray


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_path(base_yaml: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative).expanduser()
    if p.is_file():
        return p
    cand = (base_yaml.parent / maybe_relative).resolve()
    return cand


def _dataset_split_image_dir(data_yaml: Path, cfg: Dict[str, Any], split: str) -> Path:
    """Resolve ``train``/``val``/``test`` image folder like Ultralytics (``path`` + split)."""
    split_path = cfg.get(split)
    if not split_path:
        raise KeyError(f"Split '{split}' missing in {data_yaml}")
    rel = Path(split_path)
    if rel.is_absolute():
        return rel.resolve()
    root_key = cfg.get("path")
    if root_key is not None:
        root = _resolve_path(data_yaml, str(root_key))
    else:
        root = data_yaml.parent
    return (root / rel).resolve()


def _xywhn_to_xyxy(xywhn: np.ndarray, w: int, h: int) -> np.ndarray:
    """Convert normalized YOLO ``xywh`` to pixel ``xyxy``.

    Args:
        xywhn: Array ``[N, 4]`` in relative coordinates.
        w: Image width in pixels.
        h: Image height in pixels.

    Returns:
        Array ``[N, 4]`` in absolute xyxy.
    """
    if xywhn.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    x, y, bw, bh = xywhn.T
    x1 = (x - bw / 2.0) * w
    y1 = (y - bh / 2.0) * h
    x2 = (x + bw / 2.0) * w
    y2 = (y + bh / 2.0) * h
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


def _box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between sets ``a`` (N,4) and ``b`` (M,4) in xyxy format."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0.0, a_max=None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter + 1e-7
    return inter / union


def load_yolo_gt(label_path: Path, im_w: int, im_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """Read YOLO label file into xyxy (pixels) and class indices.

    Supports per-line **detection** format (``cls x y w h``, five numbers) and **segmentation**
    polygons (``cls x1 y1 x2 y2 ...``); polygons are converted to an axis-aligned bounding box.
    """
    if not label_path.is_file():
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
    boxes: List[np.ndarray] = []
    classes: List[int] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        try:
            cls_id = int(float(parts[0]))
        except (ValueError, IndexError):
            continue
        n = len(parts)
        if n == 5:
            xywhn = np.array(
                [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])],
                dtype=np.float32,
            ).reshape(1, -1)
            xyxy = _xywhn_to_xyxy(xywhn, im_w, im_h)[0]
            boxes.append(xyxy)
            classes.append(cls_id)
        elif n > 5 and (n - 1) % 2 == 0:
            coords = [float(parts[i]) for i in range(1, n)]
            xs = np.array(coords[0::2], dtype=np.float32)
            ys = np.array(coords[1::2], dtype=np.float32)
            x1, x2 = float(xs.min() * im_w), float(xs.max() * im_w)
            y1, y2 = float(ys.min() * im_h), float(ys.max() * im_h)
            boxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))
            classes.append(cls_id)
    if not boxes:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
    return np.stack(boxes, axis=0), np.array(classes, dtype=np.int64)


def match_image(
    gt_xyxy: np.ndarray,
    gt_cls: np.ndarray,
    pred_xyxy: np.ndarray,
    pred_cls: np.ndarray,
    pred_conf: np.ndarray,
    conf_thresh: float,
    iou_thresh: float,
) -> Tuple[int, int, int]:
    """Greedy IoU matching at a fixed confidence cutoff (per image).

    Returns:
        ``(tp, fp, fn)`` counts for this image.
    """
    keep = pred_conf >= conf_thresh
    p_box = pred_xyxy[keep]
    p_cls = pred_cls[keep]
    p_conf = pred_conf[keep]
    order = np.argsort(-p_conf)
    p_box, p_cls = p_box[order], p_cls[order]

    gt_used = np.zeros(len(gt_xyxy), dtype=bool)
    tp = fp = 0
    for i in range(len(p_box)):
        best_j = -1
        best_iou = 0.0
        for j in range(len(gt_xyxy)):
            if gt_used[j]:
                continue
            if int(p_cls[i]) != int(gt_cls[j]):
                continue
            iou = _box_iou_matrix(p_box[i : i + 1], gt_xyxy[j : j + 1])[0, 0]
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_thresh:
            tp += 1
            gt_used[best_j] = True
        else:
            fp += 1
    fn = int((~gt_used).sum())
    return tp, fp, fn


def aggregate_metrics(
    per_image: Sequence[ImageEvalArrays],
    conf_thresholds: Sequence[float],
    iou_thresh: float,
) -> Dict[str, Dict[str, float]]:
    """Compute precision, recall, F1, and FDR at each confidence threshold."""
    out: Dict[str, Dict[str, float]] = {}
    for t in conf_thresholds:
        tp = fp = fn = 0
        for sample in per_image:
            a, b, c = match_image(
                sample.gt_xyxy,
                sample.gt_cls,
                sample.pred_xyxy,
                sample.pred_cls,
                sample.pred_conf,
                conf_thresh=float(t),
                iou_thresh=iou_thresh,
            )
            tp, fp, fn = tp + a, fp + b, fn + c
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        fdr = 1.0 - precision
        key = f"{t:.2f}"
        out[key] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "fdr": float(fdr),
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
        }
    return out


def _label_path_for_image(im_path: Path) -> Path:
    """Resolve ``labels/*.txt`` next to a standard ``.../images/`` folder when possible."""
    if im_path.parent.name.lower() == "images":
        return im_path.parent.parent / "labels" / f"{im_path.stem}.txt"
    return im_path.with_suffix(".txt")


def gather_predictions(
    weights: Path,
    data_yaml: Path,
    split: str,
    imgsz: int,
    device: str,
) -> List[ImageEvalArrays]:
    """Run the model once at low conf, then threshold in software."""
    cfg = _load_yaml(data_yaml)
    img_dir = _dataset_split_image_dir(data_yaml, cfg, split)

    model = YOLO(str(weights))
    results: List[ImageEvalArrays] = []
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.webp")
    files: List[Path] = []
    for pat in exts:
        files.extend(sorted(img_dir.glob(pat)))
    if not files:
        raise FileNotFoundError(f"No images under {img_dir}")

    for im_path in files:
        im = cv2.imread(str(im_path))
        if im is None:
            continue
        h, w = im.shape[:2]
        lbl = _label_path_for_image(im_path)
        gt_xyxy, gt_cls = load_yolo_gt(lbl, w, h)

        # Ultralytics handles NMS internally; use low conf to retain headroom for threshold sweep.
        pred = model.predict(
            source=im,
            imgsz=imgsz,
            conf=0.001,
            iou=0.7,
            device=device,
            verbose=False,
        )[0]
        if pred.boxes is None or len(pred.boxes) == 0:
            results.append(
                ImageEvalArrays(
                    gt_xyxy=gt_xyxy,
                    gt_cls=gt_cls,
                    pred_xyxy=np.zeros((0, 4), np.float32),
                    pred_cls=np.zeros((0,), np.int64),
                    pred_conf=np.zeros((0,), np.float32),
                )
            )
            continue
        boxes = pred.boxes
        pred_xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        pred_cls = boxes.cls.cpu().numpy().astype(np.int64)
        pred_conf = boxes.conf.cpu().numpy().astype(np.float32)
        results.append(
            ImageEvalArrays(
                gt_xyxy=gt_xyxy,
                gt_cls=gt_cls,
                pred_xyxy=pred_xyxy,
                pred_cls=pred_cls,
                pred_conf=pred_conf,
            )
        )
    return results


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Custom YOLO eval metrics on image splits.")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True, help="Ultralytics data.yaml")
    p.add_argument("--split", type=str, default="val", choices=["val", "test", "train"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument(
        "--conf-thresholds",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75],
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional file to save the same metric lines (UTF-8). Still prints to stdout.",
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ["WANDB_DISABLED"] = "true"

    per_image = gather_predictions(
        args.weights,
        args.data,
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
    )
    metrics = aggregate_metrics(per_image, args.conf_thresholds, args.iou_threshold)

    lines: List[str] = []
    for ck, row in sorted(metrics.items(), key=lambda kv: float(kv[0])):
        line = (
            f"conf={ck}  P={row['precision']:.4f}  R={row['recall']:.4f}  "
            f"F1={row['f1']:.4f}  FDR={row['fdr']:.4f}  TP/FP/FN="
            f"{int(row['tp'])}/{int(row['fp'])}/{int(row['fn'])}"
        )
        print(line)
        lines.append(line)
    if args.out is not None:
        args.out = Path(args.out).expanduser().resolve()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
