# AeroSentry — small-UAV detection (take-home)

End-to-end project: **train a YOLO detector** on a YOLO-format image dataset, **evaluate** on held-out frames, and **run video inference** with an optional **false-positive (FP) reduction** stack on top of raw detections.

This repo is **code only**. Dataset paths, videos, and `*.pt` checkpoints live on your machine (see `.gitignore`). Point `config/dataset_aerosentry.yaml` at your data before training or evaluation.

---

## Architecture (short)

| Layer | Role |
|--------|------|
| **Detector** | **YOLO11n** via Ultralytics — one forward pass per frame; outputs boxes, scores, class. |
| **Video path** | OpenCV reads frames → detector → optional post-process. |
| **FP reduction (optional)** | **`FalsePositiveSuppressor`**: lightweight **tracking** (e.g. M-of-N consistency, smoothing) plus a **geometric gate** (**`GeometricEgoMotion`**: sparse features, essential/homography models) to drop detections that disagree with estimated camera motion. Use **`--fp-suppressor`** for the full gate or **`--fp-geo-only`** to ablate tracking and keep geometry-only behavior. |

Unified entry point: **`python run.py`** with subcommands `train`, `eval`, `infer`, `compare-fp-video`, etc.

Deeper diagrams and module map: [`docs/README.md`](docs/README.md). Algorithmic detail (Hebrew): [`docs/PROJECT_GUIDE_HE.md`](docs/PROJECT_GUIDE_HE.md).

---

## Setup (reviewers)

**Requirements:** Python **3.10+**, CUDA optional (use `--device cpu` if needed).

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
export PYTHONPATH=.
```

Edit **`config/dataset_aerosentry.yaml`** so `path`, `train`, `val`, and `test` point to a valid YOLO layout on disk.

---

## Produce weights (checkpoints)

Weights are **not** committed. After training you get `best.pt` under `runs/detect/aerosentry/<run_name>/weights/`.

**Step-by-step (no Docker):** [`docs/SUBMISSION.md`](docs/SUBMISSION.md)

Minimal flow:

```bash
python3 run.py train --experiment A    # baseline — use B for domain-style augmentations
find runs -name best.pt
```

Resume: `python3 run.py train --experiment A --resume runs/detect/aerosentry/<run>/weights/last.pt`

---

## Run the model

**Metrics on image splits** (P/R/F1 sweep over confidence thresholds):

```bash
python3 run.py eval \
  --weights /path/to/best.pt \
  --data config/dataset_aerosentry.yaml \
  --split test \
  --device 0
```

**Video with boxes** (writes an MP4):

```bash
mkdir -p outputs
python3 run.py infer \
  --weights /path/to/best.pt \
  --source /path/to/video.mp4 \
  --device 0 \
  --conf 0.25 \
  --out outputs/annotated.mp4
```

**With FP reduction:** add `--fp-suppressor` (or `--fp-geo-only` for geometry-only ablation).

**Reproducible benchmark table** (raw vs FP modes on one clip, multiple checkpoints):

```bash
python3 run.py compare-fp-video --help
# e.g. --discover-runs runs/detect/aerosentry --video /path/to/clip.mp4 --device 0
```

A compact results snapshot (tables + links to example outputs): [`docs/REPORT.md`](docs/REPORT.md).

---

## Written report (PDF)

The assignment asks for a **short report (3–6 pages)** covering:

- Architecture diagram and pipeline walk-through  
- Model choice, training setup, and training-curve diagnostics  
- FP-reduction approach and alternatives considered  
- **Jetson Orin Nano** deployment plan: precision, expected latency, memory, risks  
- Evaluation setup, metrics, results on **`test/`** and **`test_footage/`**, and limitations  
- **“Another week”** improvements  

**Submitted write-up:** add your PDF beside the repo and link it here, for example:

[Written report (PDF)](docs/REPORT_SUBMISSION.pdf)

*(Replace the path with your actual filename if different.)*

---

## Demo (see the system run)

Pick one or more:

- **Short annotated videos:** run `infer` with `--out` (and optionally `--fp-suppressor`) on provided test footage.  
- **Quantitative demo:** `compare-fp-video` writes Markdown/CSV/JSON under `outputs/` (see `--out-md`).  
- **On-disk examples:** if present, see `outputs/` for poster-derived clips referenced from [`docs/REPORT.md`](docs/REPORT.md).

```bash
python3 run.py --help
```

---

## License / privacy

Do not commit sensitive dataset paths or customer footage in public forks; keep local paths in `config/dataset_aerosentry.yaml` only.
