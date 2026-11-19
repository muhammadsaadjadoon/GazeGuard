# GazeGuard custom-training guide — Windows, Python 3.14

## What this actually trains

A small **binary landmark classifier** (logistic regression implemented with NumPy), based on **8 numerical features** from both irises, eye direction and head orientation. Label `lens` is the positive class. `screen`, `left`, `right`, `up`, `down`, `away` are negative classes. It is **not** a large end-to-end image model and cannot guarantee tiny lens/screen deviations from a commodity webcam.

## 1. Install and run the existing app

From the extracted root folder (where `run.py` and `pyproject.toml` sit):

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe run.py
```

Dark browser dashboard, automatic camera startup, hand skeleton and advanced manual calibration remain in place. Close the running app before collecting data: most webcams cannot be used by two processes simultaneously.

## 2. Collect REAL data with the webcam

Inform every participant: we capture numeric iris/head position **landmarks**, not raw video/images, but landmarks are still sensitive personal information. Obtain informed consent and use anonymous participant IDs.

```powershell
.\.venv\Scripts\python.exe -m training.collect_webcam --participant p01 --consent
```

Keys while the collection window is active:

- `1` = look directly at physical camera lens
- `2` = look at screen center (not at lens)
- `3` = look left
- `4` = look right
- `5` = look up
- `6` = look down
- `Q` = stop and save the CSV

After pressing a target key, hold your gaze for ~9 seconds while the app collects samples. Vary lighting, glasses, seated distance, camera heights, and screen positions. Repeat **all targets** in at least **two independent sessions** per person, preferably across **multiple consenting participants**. More diversity is needed for reliable deployment.

CSV output: `datasets/private/landmarks.csv`. The app never collects raw camera images in this workflow.

## 3. Train and evaluate on held-out groups

```powershell
.\.venv\Scripts\python.exe -m training.train_landmarks --csv datasets/private/landmarks.csv --output models/trained_gaze.json
```

Training uses NumPy feature standardization and a weighted nonlinear-feature logistic classifier. It splits by participant when possible, otherwise by session. It **refuses** training when validation cannot represent both classes or recall is too low. The report shows `TP`, `FP`, `TN`, `FN`, recall and false-ACTIVE rate for the held-out group; a single test session is not a production benchmark.

## 4. Optional integration with GazeGuard

The trained model is deliberately **OFF BY DEFAULT** so a weak model cannot automatically change live decisions. Copy a reviewed model into the local app-data directory printed by `doctor`, which normally is `%USERPROFILE%\.gazeguard_v1`.

```powershell
Copy-Item models\trained_gaze.json "$env:USERPROFILE\.gazeguard_v1\trained_gaze.json"
$env:GAZEGUARD_ENABLE_TRAINED_GAZE="1"
.\.venv\Scripts\python.exe run.py
```

The model can only **veto** an automatic ACTIVE frame as LOOKING AWAY. It can never turn an already-rejected baseline frame into ACTIVE. This conservative integration is intentional. It does not replace gaze calibration or a validated hardware eye tracker.

## 5. Before a public product launch

Evaluate on separate people and cameras not used to fit or select thresholds; record lens-vs-screen confusion matrices and reaction time in realistic lighting. Obtain dataset rights and privacy/consent arrangements. Never automatically accuse someone of cheating based solely on gaze.

See `datasets/DATASET_SOURCES.md` for external, third-party research datasets and their licensing caveats. The project ZIP contains **no externally sourced image datasets**.
