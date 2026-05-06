# Robust UAV Vision: Hybrid Detection & Geometric FP Suppression

This repository implements an advanced vision pipeline for UAV-based target detection. It balances **real-time efficiency** with **high-fidelity robustness** using a cascaded fallback mechanism and a geometry-aware filtering stage.

---

## Project Overview

In UAV scenarios, standard detectors often struggle with motion blur, small targets, and complex background clutter. This project addresses these challenges through:

* **Cascaded Hybrid Detector**: "Uncertainty-driven" logic that triggers a heavier model only when needed.
* **Geometric FP Suppression**: Filtering "ghost" detections using egomotion-consistent geometry (F & H matrices).

---

## Setup & Installation

**Requirements:** Python **3.10+**, CUDA-enabled GPU.

```bash
# Clone and enter the repository
git clone [https://github.com/roeytoo111/aerosentry_task.git](https://github.com/roeytoo111/aerosentry_task.git)
cd aerosentry_task

# Environment setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
```

> **Note:** Ensure `config/dataset_aerosentry.yaml` points to your local dataset paths.

---

## Core Architecture

### 1. Cascaded Hybrid Detector (`HybridDetector`)
The system employs a **Dual-Path** inference strategy to optimize the compute-vs-accuracy trade-off:

1.  **Primary Path (YOLO11)**: Runs on every frame for high-speed tracking.
2.  **Fallback Trigger (RT-DETR)**: Re-processes the *same* frame with a heavier model if:
    * **Semantic Uncertainty**: YOLO detects objects but with scores below `uncertainty_thresh`.
    * **Track Loss**: A "miss streak" is detected by the tracking loop.
3.  **Cooldown Mechanism**: Prevents rapid oscillation between models to maintain stable throughput.

### 2. False-Positive (FP) Suppressor
Located in `src/tracking/fp_suppressor.py`, this stage cleans the raw detector output:

* **Temporal Filtering**: Greedy IoU association with **M-of-N voting** to confirm tracks and **One Euro Smoothing** for jitter reduction.
* **Global Geometry Gate**: Uses ORB features and RANSAC to estimate the **Fundamental (F)** and **Homography (H)** matrices of the scene.
* **Consistency Check**: Detections behaving like static background (based on inlier ratios under F/H) are discarded as FPs. Only objects moving independently from the camera's ego-motion are kept.

---

## Usage Examples

### Training & Evaluation
```bash
# Train baseline model
python3 run.py train --experiment A

# Evaluate on test split
python3 run.py eval --weights runs/detect/.../best.pt --split test
```

### Inference with FP Suppression
Run the hybrid demo which exercises both the fallback logic and the geometric filter:
```bash
python3 examples/hybrid_video_demo.py \
  --video path/to/input.mp4 \
  --yolo path/to/yolo_weights.pt \
  --rtdetr rtdetr-l.pt \
  --fp-suppressor \
  --out outputs/hybrid_result.mp4
```

### Ablation Comparison
Generate a comparative report between raw detection and geometric-only filtering:
```bash
python3 run.py compare-fp-video \
  --weights runs/detect/model_A.pt runs/detect/model_B.pt \
  --video path/to/test_clip.mp4 \
  --fp-suppressor \
  --out-md outputs/comparison_report.md
```

---

## Configuration
Centralized tuning lives in `config/tracking_fp.yaml`:
* `track_manager`: M-of-N and One Euro parameters.
* `geometric_ego_motion`: RANSAC thresholds and ORB feature limits.
* `hybrid_demo_detector`: Fallback uncertainty thresholds.

---

## 📖 Relevant articles
https://arxiv.org/abs/2602.13324

https://arxiv.org/abs/2402.08550

https://arxiv.org/abs/2412.04147
