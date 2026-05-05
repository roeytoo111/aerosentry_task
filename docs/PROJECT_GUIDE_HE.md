# CV-Engineer-Assignment: AeroSentry Pipeline Architecture

תרשים זה מציג את זרימת המידע והארכיטקטורה של הפרויקט בצורה קומפקטית, משלב ה-CLI (נקודת הכניסה), דרך הכלים, האורקסטרציה, מודלי הגילוי, ועד למערכת המעקב וסינון התראות השווא (FP Gate) שמעדכנת את חוזי הנתונים.

```mermaid
flowchart LR
    RUN([run.py])

    subgraph Tools [CLI & Scripts]
        direction TB
        INF[infer_video.py]
        TR[train_detector.py]
        EV[evaluate_detector.py]
        UTILS[export, report, split, benchmark]
    end

    subgraph Orchestration [Pipeline & Eval]
        PM[pipeline_manager.py]
        TE[tactical_evaluator.py]
    end

    subgraph Detection [Models]
        UF[UltralyticsYolo]
        BF[base_detector.py]
        DF[detector_factory.py]
    end

    subgraph Tracking [Tracking & FP Gate]
        FP[fp_suppressor.py]
        TM[track_manager.py]
        GE[geometric_ego_motion.py]
        FI[filters.py]
    end

    DC[(data_contracts.py)]

    %% Routing (Compact Syntax)
    RUN --> INF & TR & EV & UTILS & PM
    INF & TE --> UF & FP
    PM --> BF & FP & DC
    FP --> TM & GE & DC
    TM --> FI
    UF --> DC