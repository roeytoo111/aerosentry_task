# How to produce trained weights (no Docker)

The repository does **not** commit `*.pt` or `runs/` (see `.gitignore`). Checkpoints are created on your machine after training.

## 1. Environment

- **Python 3.10+**
- From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
export PYTHONPATH=.
```

(`PYTHONPATH` must include the repo root so `src/` imports work.)

## 2. Dataset

Edit **`config/dataset_aerosentry.yaml`**: `path`, `train`, `val`, `test` must point to a valid **YOLO** layout (images + labels). Training will not run without this.

## 3. Backbone (first run)

Training starts from **`yolo11n.pt`** (see `config/experiments.yaml` → `global.model`). Ultralytics **downloads** it automatically the first time if the file is missing.

## 4. Train (reproducibility)

- **Seed:** `global.seed: 42` in `config/experiments.yaml`.
- **`deterministic: True`** is passed into Ultralytics from `run_training()` in `src/models/train_detector.py`.  
  On GPU, runs are **best-effort** reproducible (CUDA nondeterminism may still cause small differences).

**Experiment A (baseline):**

```bash
python3 run.py train --experiment A
```

**Experiment B (domain blur / exposure-style albumentations):**

```bash
python3 run.py train --experiment B
```

**Shorter run** (sanity check, not full quality):

```bash
python3 run.py train --experiment A --epochs 5
```

## 5. Where weights are written

After training, checkpoints are under:

```text
runs/detect/aerosentry/<run_name>/weights/best.pt
runs/detect/aerosentry/<run_name>/weights/last.pt
```

`<run_name>` comes from `train.name` in YAML (e.g. `yolo11_baseline` for A, `yolo11_domain_aug` for B unless overridden).

Find them:

```bash
find runs -name best.pt
```

Use `best.pt` for `run.py eval`, `run.py infer`, etc.

## 6. Resume / fine-tune

```bash
python3 run.py train --experiment A --resume runs/detect/aerosentry/<run_name>/weights/last.pt
```

See help text on `--resume` in `train_detector.py` for full vs fine-tune behavior.
