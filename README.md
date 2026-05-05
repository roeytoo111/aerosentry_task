# AeroSentry — מדריך הרצה: מאימון ועד תוצאה בווידאו

הפרויקט מאמן **YOLO** על **תמונות** (YOLO layout), מודד **Precision / Recall / F1** על val/test, ומריץ **אינפרנס וידאו** עם אופציה לשכבת מעקב וסינון FP גאומטרי.

**המאגר הוא קוד בלבד** — הדאטאסט והווידאו נשארים אצלך בדיסק; לא לקמפל פרטי נתונים רגישים.

**דרישה:** **Python 3.10+** (בכל הפקודות להלן נעשה שימוש ב־`python3`).

---

## זרימה מלאה (עשרת אלפים רגל)

| שלב | מה עושים | פקודה / קובץ |
|-----|-----------|----------------|
| 1 | סביבה + תלויות | סעיף [התקנה](#1-התקנה) |
| 2 | לחבר נתיב דאטאסט | `config/dataset_aerosentry.yaml` |
| 3 | לבחור ניסוי אימון | `config/experiments.yaml` (`A` / `B` / `C`) |
| 4 | לאמן | `python3 run.py train --experiment A` |
| 5 | לאתר `best.pt` | `find runs -name best.pt` |
| 6 | להעריך על תמונות | `python3 run.py eval --weights … --split val` |
| 7 | לבדוק בווידאו | `python3 run.py infer --weights … --source … --out …` |
| 7b | טבלת השוואה — כמה מודלים על אותו וידאו (עם/בלי FP) | `python3 run.py compare-fp-video …` |
| 8 | (אופציונלי) דוח טקטי / ייצוא / פיצול | `report`, `export`, `split` — ראו [סקירת פקודות](#סקירת-פקודות-runpy) |

---

## 1) התקנה

מתיקיית השורש של המאגר (`aerosentry_task`):

```bash
cd /path/to/aerosentry_task
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
export PYTHONPATH=.
```

**בכל טרמינל חדש** (אותה תיקייה):

```bash
cd /path/to/aerosentry_task
source .venv/bin/activate
export PYTHONPATH=.
```

אימות גרסה:

```bash
python3 --version   # צפוי 3.10 או מעלה
```

---

## 2) חיבור הדאטאסט

ערוך את **`config/dataset_aerosentry.yaml`**:

- **`path`** — התיקייה שמכילה את `train`, `valid`, `test` (לרוב שורש ייצוא YOLO מ־Roboflow). ניתן נתיב יחסי מ־`config/`, למשל `../Video Analytics/.../yolov11`.
- **`train`**, **`val`**, **`test`** — בדרך כלל `train/images`, `valid/images`, `test/images` (ולייבלים במקביל).

אם הנתיב שגוי, אימון והערכה ייכשלו או ידווחו על אפס תמונות.

---

## 3) ניסויי אימון

הגדרות ניסויים (ארכיטקטורה, אופטימיזר, epoch וכו׳) נמצאות ב־**`config/experiments.yaml`**.

- **`python3 run.py train --experiment A`** — ניסוי baseline (החלף ב־`B` או `C` לפי הקובץ).
- **`--resume PATH/last.pt`** — להמשך אימון.
- **`--epochs N`** — לעקוף את מספר הEpochs ב־YAML להרצה זו.

---

## 4) אימון

```bash
python3 run.py train --experiment A
```

לאחר סיום תקבל תיקייה בסגנון  
`runs/detect/aerosentry/<run_name>/` עם **`weights/best.pt`** ו־**`weights/last.pt`**.

לאתר צ׳קפוינט:

```bash
find runs -name best.pt
```

העתק את הנתיב המלא ל־`--weights` בשלבים הבאים (אל תשתמש במחרוזת מילולית `...`).

אם **`weights/` ריק** — האימון נקטע מוקדם; הרץ שוב או בחר ריצה אחרת עם `best.pt`.

---

## 5) הערכה על תמונות (val / test)

מדפיס **Precision / Recall / F1** במספר ספי **confidence** — על **תמונות בלבד**, לא על וידאו.

```bash
python3 run.py eval \
  --weights runs/detect/aerosentry/yolo11_baseline-4/weights/best.pt \
  --data config/dataset_aerosentry.yaml \
  --split val \
  --device 0
```

- החלף **`--weights`** לנתיב ה־`best.pt` שלך.
- **ללא GPU:** `--device cpu`
- **מחלקת test:** `--split test`

פלט טיפוסי: שורות `conf=0.25  P=...  R=...  F1=...  TP/FP/FN=...`  
סף `conf` גבוה יותר בדרך כלל מעלה precision ומוריד recall — תקין.

---

## 6) אינפרנס וידאו (התוצאה “הסופית” ויזואלית)

יוצר קובץ וידאו עם תיבות (BGR, OpenCV `VideoWriter`).

```bash
mkdir -p outputs
python3 run.py infer \
  --weights runs/detect/aerosentry/yolo11_baseline-4/weights/best.pt \
  --source "/full/path/to/video.mp4" \
  --device 0 \
  --conf 0.25 \
  --out outputs/annotated.mp4
```

- **`--conf`** — סף confidence ל־Ultralytics (ברירת מחדל `0.25`).
- **`--imgsz 640`** — ניתן לשנות בהתאם לאימון (ברירת מחדל ב־CLI היא 640).
- **`--max-frames 300`** — לבדיקה מהירה על קטע קצר.
- **`opencv-python-headless`** — אין חלון; השתמש ב־**`--out`** (לא רק `--show`).

### אינפרנס + שער FP (מעקב M-of-N, החלקה, גאומטריה F/H)

```bash
python3 run.py infer \
  --weights runs/detect/aerosentry/yolo11_baseline-4/weights/best.pt \
  --source "/full/path/to/video.mp4" \
  --device 0 \
  --conf 0.25 \
  --fp-suppressor \
  --out outputs/with_fp_gate.mp4
```

בסיום אמור להופיע שורה כמו **`Wrote .../annotated.mp4`**.

### השוואת כמה מודלים על אותו וידאו (עם ובלי FP Reducer)

לכל `best.pt`: מריצים את הווידאו **פעמיים** — ספירת זיהויי YOLO בלבד, ואז אחרי `FalsePositiveSuppressor`. הפלט: **`outputs/fp_video_compare.md`** (טבלה), **`fp_video_compare.csv`**, **`fp_video_compare.json`**.

```bash
mkdir -p outputs

# כל התיקיות runs/detect/aerosentry/*/weights/best.pt
python3 run.py compare-fp-video \
  --discover-runs runs/detect/aerosentry \
  --video "Video Analytics/Test Footage/homography_checking.mov" \
  --device 0 \
  --conf 0.25

# או רשימה ידנית + תוויות לשורות בטבלה
python3 run.py compare-fp-video \
  --weights runs/detect/aerosentry/yolo11_baseline-4/weights/best.pt \
        runs/detect/aerosentry/other_run/weights/best.pt \
  --names "exp-A-baseline4" "exp-B-custom" \
  --video "Video Analytics/Test Footage/homography_checking.mov" \
  --device 0 \
  --conf 0.25 \
  --out-md outputs/my_compare.md
```

לניסוי קצר: הוסף `--max-frames 300`.

---

## סקירת פקודות (`run.py`)

```bash
python3 run.py --help
python3 run.py train --help
python3 run.py infer --help
```

| פקודה | Plain-English |
|--------|----------------|
| `train` | Learn weights from YOLO image folders. |
| `eval` | Print metrics on val/test **images**. |
| `infer` | Run trained weights on a **video** file; save or show boxes. |
| `report` | Build a tactical markdown report (weights + video path, optional distractors / JSON). |
| `export` | Export ONNX / TensorRT (advanced environment). |
| `split` | Optional: rebuild train/val/test from raw images **without** mixing the same clip—skip if Roboflow already split. |
| `compare-fp-video` | Several `best.pt` on **one** video: raw counts vs FP suppressor; MD + CSV + JSON. |

---

## דוגמאות ומתי להשתמש בכל פקודה

להלן **דוגמת `python3`** לכל פקודה ו**למה** בדרך כלל משתמשים בה. הנתיבים ל־`best.pt` / וידאו — להחליף אצלך. הנחה: `cd` לשורש המאגר, `export PYTHONPATH=.` (או venv פעיל).

### `train` — לאמן מודל מתיקיות תמונה+לייבל

**מתי:** מתחילים מחדש, משנים ניסוי ב־`experiments.yaml`, או ממשיכים אימון.

```bash
python3 run.py train --experiment A
python3 run.py train --experiment B --epochs 100
python3 run.py train --experiment A --resume runs/detect/aerosentry/some_run/weights/last.pt
```

---

### `eval` — מדדים על val/test (תמונות בלבד)

**מתי:** להשוות ריצות, לבחור סף `conf`, לדווח P/R/F1 לפני/אחרי שינוי דאטאסט — **לא** מחליף בדיקת וידאו.

```bash
python3 run.py eval \
  --weights runs/detect/aerosentry/yolo11_baseline-4/weights/best.pt \
  --data config/dataset_aerosentry.yaml \
  --split val \
  --device 0
```

---

### `infer` — וידאו עם תיבות (המודל האמיתי)

**מתי:** הדגמה ויזואלית, בדיקת קליפ שטח, איסוף וידאו מתויג ידנית — זה **הכלי ל־`best.pt`**.

```bash
python3 run.py infer \
  --weights runs/detect/aerosentry/yolo11_baseline-4/weights/best.pt \
  --source "Video Analytics/Test Footage/homography_checking.mov" \
  --device 0 \
  --conf 0.25 \
  --out outputs/run.mp4
```

עם שער FP (מעקב + גאומטריה): הוסף `--fp-suppressor`. לבדיקה מהירה: `--max-frames 300`.

---

### `report` — דוח טקטי (טבלת Markdown + JSON)

**מתי:** לארוז מדדי **latency / FPS** (וברירת מחדל bench חי FP32) עם מדדי אופציונליים מדיסטרקטורים; למלא שורות FP16/INT8 לרוב דרך `--metrics-json` אם ייצאת מודלים נפרדים.

**דורש** לפחות `--weights` ו־`--video` **רק אם** רוצים `run_live_bench` (נוסף למה שכבר ב־JSON).

```bash
python3 run.py report \
  --weights runs/detect/aerosentry/yolo11_baseline-4/weights/best.pt \
  --video "Video Analytics/Test Footage/homography_checking.mov" \
  --distractor-dir path/to/hard_negative_images \
  --out docs/TACTICAL_REPORT.md
```

רק למזג מדדים שכבר בקובץ (בלי וידאו חי):

```bash
python3 run.py report --metrics-json path/to/metrics.json --out TACTICAL_REPORT.md
```

---

### `export` — ONNX / מנוע TensorRT (ייצור / Jetson)

**מתי:** פריסה מחוץ ל־PyTorch, האצת Jetson, שרשרת FP16/INT8 — דורש סביבה עם חבילות ייצוא תואמות.

```bash
python3 run.py export \
  --weights runs/detect/aerosentry/yolo11_baseline-4/weights/best.pt \
  --out exports/my_engine \
  --imgsz 640 \
  --fp16
```

INT8 לרוב דורש `--int8` ו־`--calibration-data` עם YAML מתאים לפריימי כיול (ראו `--help` של הכלי).

**בלי CUDA** (`torch.cuda.is_available()` = False): ייצוא **ONNX** עדיין יכול להצליח; בניית **TensorRT `.engine` תידלג** אוטומטית (הרץ שוב על מכונת GPU / Jetson אם צריך מנוע).

---

### `split` — פיצול train/val/test **לפי רצף** (בלי דליפה טמפורלית)

**מתי:** כל התמונות בשטח אחד, **אין** פיצול Roboflow; רוצים שכל הפריים מאותו קליפ ייפלו לאותו split.

```bash
python3 run.py split \
  --source path/to/all/raw/images \
  --output path/to/yolo_dataset_root \
  --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
```

**מתי לא:** כבר יש `train/` / `valid/` / `test/` מתואמים ל־`dataset_aerosentry.yaml`.

---

### `compare-fp-video` — טבלת השוואה על וידאו (עם / בלי FP)

**מתי:** כמה משקולות על **אותו** קליפ — לסיכום מספרי (לא וידאו מתויג). ראו גם [סעיף 6](#6-אינפרנס-וידאו-התוצאה-הסופית-ויזואלית).

```bash
python3 run.py compare-fp-video --discover-runs runs/detect/aerosentry \
  --video "Video Analytics/Test Footage/homography_checking.mov" \
  --device 0 --conf 0.25
```

---

### `demo` — דמו עם דטקטור **מזויף** (לא YOLO מאומן)

**מתי:** לוודא שהצינור (וידאו → post-process) עובד בלי משקולות; **לא** לאיכות ביעד אמיתי.

```bash
python3 run.py demo --video "Video Analytics/Test Footage/homography_checking.mov"
python3 run.py demo --video "path/to/your_clip.mp4" --model-name yolov11
```

למודל שאומן על הדאטאסט שלך — תמיד **`infer` + `best.pt`**.

---

## הבדלים חשובים

- **אימון / eval** — עובדים על **קבצי תמונה** לפי ה־YAML.
- **infer** — **ווידאו** (או מה ש־OpenCV `VideoCapture` יודע לפתוח).
- **eval** משתמש בסweep פנימי של ספי `conf`; **infer** משתמש ב־**`--conf`** של Ultralytics ישירות.

---

## תקלות נפוצות

- **`Weights file not found`** — הנתיב ל־`best.pt` שגוי; הרץ `find runs -name best.pt`.
- **`Invalid --weights path (literal '...')`** — הועתק placeholder מהמדריך; החלף בנתיב אמיתי.
- **אין חלון וידאו** — התקן build עם GUI או השתמש רק ב־`--out`.
- **CUDA / מכשיר** — `--device cpu` אם אין GPU.
