# הוכחת תפעול: `GeometricEgoMotion` בריצת וידאו אמיתית

מסמך זה מתעד **ריצה אחת** של `infer` עם `--fp-suppressor --geo-debug`, כדי לתמוך בדוח / במטלה: **כל שלבי המודול** `src/tracking/geometric_ego_motion.py` בוצעו בפועל והופק פלט ניתן לבדיקה.

---

## פרמטרי הריצה (להעתקה לשחזור)

```bash
cd ~/aerosentry_task
export PYTHONPATH=.

python3 run.py infer \
  --weights runs/detect/aerosentry/yolo11_baseline-4/weights/best.pt \
  --source "Video Analytics/Test Footage/homography_checking.mov" \
  --device 0 \
  --conf 0.25 \
  --fp-suppressor \
  --geo-debug \
  --max-frames 120 \
  --out outputs/debug_geo.mp4
```

**מטא־דאטה מהריצה המתועדת:**

| שדה | ערך |
|-----|-----|
| וידאו | `homography_checking.mov` (דוגמה מהמאגר: 1080×1920 @ ~30 FPS) |
| פריימים שעובדו | 120 (מתוך ~386 בקובץ) |
| זמן כולל (דיווח סקריפט) | ~11.75 s → ~10.22 FPS effective |
| פלט ויזואלי | `outputs/debug_geo.mp4` |
| נתיב ORB | **CPU** — `path=CPU (no cv2.cuda)` (OpenCV ללא `cv2.cuda` בסביבה זו) |

---

## איך הלוג מקשר לקוד (`geometric_ego_motion.py`)

| דפוס בלוג | פונקציה / אזור בקוד | מה זה מוכיח |
|------------|---------------------|---------------|
| `[FalsePositiveSuppressor] geo skipped … no_confirmed_tracks` / `no_prev_bgr` | `fp_suppressor.py` — תנאי לפני קריאה לגאומטריה | אין הרצת ORB כשאין מאושרים או אין פריים קודם — עקבי עם התכן. |
| `[GeometricEgoMotion] extract_match … path=CPU … good_matches=N` | `_extract_and_match_cuda` → `_extract_and_match_cpu` | **נקודות + Lowe + מספר התאמות** — ענף החילוץ עובד. |
| `[GeometricEgoMotion] global_models … pts=N F_inliers=… H_inliers=…` | `_compute_global_models` → `findFundamentalMat` + `findHomography` | **RANSAC ל־F ול־H** — ענף המודל הגלובלי עובד. |
| `DROP grounded-FP ratio_F=… > 0.65` | `analyze_bbox_motion` — סף `fp_inlier_ratio_F` | **סינון “נדבק לרקע”** לפי inliers בתוך ה־ROI. |
| `DROP planar-FP ratio_H=… > 0.85` | `analyze_bbox_motion` — סף `fp_inlier_ratio_H` | **סינון מישור/מדיה שטוחה** לפי H. |
| `KEEP airborne ratio_F=… ratio_H=…` | `analyze_bbox_motion` — עמידה משני הספים | **שמירת מטרה כ“אווירית”** כשהיחס נמוך מספי ה־FP. |
| `track_id=0`, אחר כך `2`, `3` | מעקב מאושר מ־`TrackManager` | אותו פריים יכול להפעיל **מספר קריאות** `analyze_bbox_motion` (מסלול לכל `track_id`). |

קובץ המקור בקוד:

- `src/tracking/geometric_ego_motion.py` — כל שלושת השלבים למעלה ממומשים שם.

---

## דגימה מהירה מהלוג (עיקרי התנהגות)

**א. דילוג גאומטריה לפני אישור / ללא `prev` (צפוי):**

```text
[FalsePositiveSuppressor] geo skipped frame_id=0: no_confirmed_tracks, no_prev_bgr
[FalsePositiveSuppressor] geo skipped frame_id=1: no_confirmed_tracks
…
```

**ב. מחזור מלא ORB → F/H → החלטת ROI (דוגמה):**

```text
[GeometricEgoMotion] extract_match: frame_id=6 path=CPU (no cv2.cuda)
[GeometricEgoMotion] extract_match: frame_id=6 path=CPU good_matches=206
[GeometricEgoMotion] global_models: frame_id=6 pts=206 F_inliers=138 H_inliers=159 (RANSAC F thresh=1.0px H thresh=3.0px)
[GeometricEgoMotion] analyze_bbox: [frame_id=6 track_id=0] DROP grounded-FP ratio_F=0.727 > 0.65 (inliers 117/161)
```

**ג. סינון לפי הומוגרפיה (מישור):**

```text
[GeometricEgoMotion] analyze_bbox: [frame_id=15 track_id=0] DROP planar-FP ratio_H=0.873 > 0.85 (inliers 89/102)
```

**ד. שמירה (לא עובר ספי FP):**

```text
[GeometricEgoMotion] analyze_bbox: [frame_id=30 track_id=0] KEEP airborne ratio_F=0.612 ratio_H=0.746 (inliers F 41/67 H 50/67)
```

**ה. יותר ממסלול מאושר בפריים אחד:**

```text
[GeometricEgoMotion] analyze_bbox: [frame_id=96 track_id=2] KEEP airborne …
[GeometricEgoMotion] analyze_bbox: [frame_id=96 track_id=3] KEEP airborne …
```

---

## מסקנה לדוח

1. **קובץ הווידאו** `outputs/debug_geo.mp4` קיים אחרי הריצה — הוכחה ויזואלית לצינור מלא (YOLO + FP + ציור).
2. **הלוג עם `--geo-debug`** מראה במפורש:
   - התאמות ORB (כולל נפילה ל־CPU כשאין CUDA ב־OpenCV),
   - הערכת **F** ו־**H** עם מספר inliers,
   - החלטות **KEEP / DROP** עם יחסים בתוך תיבת המסלול.
3. לכן ניתן לטעון במסמך הטכני: **`GeometricEgoMotion` הופעל end-to-end על קליפ אמיתי**, לא רק כיחידה אנכית מבודדת.

---

## הערות מתודולוגיות (לשקיפות בראיון)

- **יחס FP reduction** בטבלאות `compare-fp-video` הוא ספירה תפעולית; **הלוג כאן** מפרט *למה* פריים/mסלול סווג כ־DROP (F מול H).
- על **Jetson** מומלץ חבילת OpenCV עם CUDA כדי לראות `path=CUDA` בלוג; על x86/laptop עם `opencv-python` רגיל התנהגות **CPU** היא צפויה ותקינה.

---

*ניתן לצרף לדוח את קובץ הלוג המלא מהטרמינל כנספח נפרד (`homography_infer_geo_debug.log`) אם נדרש עמידה פורמלית ב“ארכיון ריצה”.*
