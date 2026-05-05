# REPORT — video model benchmark

**Weights:** `runs/detect/aerosentry/yolo11_baseline-4/weights/best.pt` (baseline) · `…/yolo11_domain_aug-2/weights/best.pt` (domain_aug2). **Detector:** conf `0.25`, imgsz `640`. Tool: `python3 run.py compare-fp-video` — each row = three full passes (Raw, Full FP, Geo-only).

**Arsuf (source):** [`Video Analytics/.../Arsuf F1 ... Clipchamp.mp4`](../Video%20Analytics/Test%20Footage/Arsuf%20F1%2009_04_2025%20-%20Made%20with%20Clipchamp.mp4)

**Poster — video links (repo-relative):**  
- Input: [`outputs/poster.mp4`](../outputs/poster.mp4)  
- Annotated (examples on disk): [`outputs/poster_fp_reducer.mp4`](../outputs/poster_fp_reducer.mp4) · [`outputs/poster_geo_fp_reducer.mp4`](../outputs/poster_geo_fp_reducer.mp4)

**Legend**  
**Σ** = total boxes over all frames. **Δ%** = `100×(1 − after/Raw Σ)` (count-based, not labeled P/R). **t Raw / t FP / t Geo** = wall time in seconds for each pass.

## Arsuf — 5960 frames

| Model | Raw Σ | Full FP Σ | Δ% | Geo Σ | Δ% | t Raw | t FP | t Geo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 965 | 304 | 68.5 | 620 | 35.8 | 90.3 | 140.0 | 186.2 |
| domain_aug2 | 1535 | 244 | 84.1 | 468 | 69.5 | 89.1 | 191.2 | 242.0 |

## Poster — 345 frames

| Model | Raw Σ | Full FP Σ | Δ% | Geo Σ | Δ% | t Raw | t FP | t Geo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 121 | 0 | 100.0 | 1 | 99.2 | 8.9 | 11.9 | 18.4 |
| domain_aug2 | 413 | 3 | 99.3 | 3 | 99.3 | 6.3 | 45.0 | 48.2 |
