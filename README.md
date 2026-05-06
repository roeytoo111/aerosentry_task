## CV engineer assignment 

End-to-end **YOLO11** training on a YOLO-format image dataset, **offline evaluation** on image splits, and **video inference** with a **single detector checkpoint** plus an optional **false-positive reduction** layer.


This repository is **code only**: datasets, videos, and `*.pt` checkpoints stay local (see `.gitignore`). Set paths in **`config/dataset_aerosentry.yaml`**.

---

## Setup

**Requirements:** Python **3.10+**.

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

## Produce weights

Weights are **not** committed. After training you get `best.pt` under `runs/detect/aerosentry/<run_name>/weights/`.

Configure runs in **`config/experiments.yaml`**, then:

```bash
python3 run.py train --experiment A    # baseline — use B for domain-style augmentations
find runs -name best.pt
```

Resume: `python3 run.py train --experiment A --resume runs/detect/aerosentry/<run>/weights/last.pt`

---

## Run the model

**Metrics on image splits** :

```bash
python3 run.py eval \
  --weights /path/to/best.pt \
  --data config/dataset_aerosentry.yaml \
  --split test \
  --device 0
```

**Video with boxes** (writes an MP4): one forward pass per frame through your **YOLO** weights; detections are optionaly passed through **`FalsePositiveSuppressor`** (`TrackManager` + **`GeometricEgoMotion`**, or geometry-only with `--fp-geo-only`).

```bash
mkdir -p outputs
python3 run.py infer \
  --weights /path/to/best.pt \
  --source /path/to/video.mp4 \
  --device 0 \
  --conf 0.25 \
  --out outputs/annotated.mp4
```

**With FP reduction:** add `--fp-suppressor` (or `--fp-geo-only` for geometry-only ablation). Tuning: **`config/tracking_fp.yaml`** when those flags are set (unless `--fp-no-config`).

**Video benchmark table** — `python3 run.py compare-fp-video` runs three full passes per checkpoint (Raw, Full FP, Geo-only). Example below uses `conf=0.25` and `imgsz=640`; two clips, separate `--out-md` so runs do not overwrite each other.

```bash
mkdir -p outputs
python3 run.py compare-fp-video \
  --weights runs/detect/aerosentry/yolo11_baseline-4/weights/best.pt \
          runs/detect/aerosentry/yolo11_domain_aug-2/weights/best.pt \
  --names baseline domain_aug \
  --video "Video Analytics/Test Footage/Arsuf F1 09_04_2025 - Made with Clipchamp.mp4" \
  --device 0 --conf 0.25 --imgsz 640 \
  --out-md outputs/fp_video_compare_arsuf.md

python3 run.py compare-fp-video \
  --weights runs/detect/aerosentry/yolo11_baseline-4/weights/best.pt \
          runs/detect/aerosentry/yolo11_domain_aug-2/weights/best.pt \
  --names baseline domain_aug \
  --video outputs/poster.mp4 \
  --device 0 --conf 0.25 --imgsz 640 \
  --out-md outputs/fp_video_compare_poster.md
```

Adjust `--video` and checkpoint paths for your machine.

---

## Written Report & Pipeline Architecture:

<p align="center">
  <img
    src="pipeline_architecture.png"
    alt="Pipeline architecture: run.py CLI entry, execution modes (infer, train, eval, utils), per-frame video path (YOLO, TrackManager, GeometricEgoMotion to FrameData), offline image evaluation, and training to best.pt"
    width="920"
  >
</p>

**Report:** [Computer_Vision_Engineer_Task.pdf](Computer_Vision_Engineer_Task.pdf)

---

## Demo 

Pick one or more:

- **Short annotated videos:** run `infer` with `--out` (and optionally `--fp-suppressor`) on provided test footage.  
- **Quantitative demo:** `compare-fp-video` writes Markdown/CSV/JSON under `outputs/` (see `--out-md`).  
- **On-disk examples:** optional annotated clips under `outputs/`.

```bash
python3 run.py --help
```

---

