# GazeGuard 12 — Camera monitoring + opt-in training workspace

A Python 3.14 Windows project based on **GazeGuard 11** with its dark browser dashboard, automatic webcam startup, face/iris/head tracking, 21-point hand skeleton, two visible live attention states, report export and optional advanced manual calibration.

**New in this folder:** a real, runnable numeric-landmark data collector and a NumPy-based trainable lens-versus-nonlens classifier with nonlinear features, with held-out session/participant evaluation and an **optional, off-by-default** live integration.

## Launch on Windows PowerShell

Extract this ZIP directly into a **new folder**. At the top level you must see `pyproject.toml` and `run.py`. In that folder:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe run.py
```

Or double-click `start_windows.bat` for initial setup. Later launch using `start_only_windows.bat`. The local dashboard auto-opens; its server stays on `127.0.0.1`. Camera/models are loaded automatically (official MediaPipe Task models download on first run). Close the terminal or press `Ctrl+C` to stop the server.

## Full contents

- `src/gazeguard/` — existing app, real-time camera, hand drawing, gaze classification and web UI
- `training/collect_webcam.py` — **actual data collection** of labelled numerical eye/face landmarks (no image/video files)
- `training/train_landmarks.py` — binary logistic training plus held-out validation metrics
- `src/gazeguard/trained_gaze.py` — inference/model JSON format used by the optional live integration
- `TRAINING_GUIDE.md` — step-by-step collection, train, test and opt-in deployment commands
- `datasets/DATASET_SOURCES.md` — official access paths + licensing notes for external datasets
- `datasets/private/` — your own opt-in observations (initially empty)
- `datasets/public_external/` — external research datasets **not shipped**
- `models/` — initially empty; train to produce `trained_gaze.json`
- `tests/` — automated logic and training tests
- `scripts/collect_windows.bat`, `scripts/train_windows.bat`, `scripts/run_trained_windows.bat` — optional training workflow helpers

## Train your own lens/screen classifier

**Close the dashboard first**, then capture samples with an explicit participant opt-in:

```powershell
.\.venv\Scripts\python.exe -m training.collect_webcam --participant p01 --consent
```

Press `1` while looking directly at lens, `2` for screen center, `3` left, `4` right, `5` up, `6` down, `Q` to quit. Repeat across **at least two sessions** (preferably many people/cameras); otherwise group-split training refuses to claim meaningful validation.

```powershell
.\.venv\Scripts\python.exe -m training.train_landmarks --csv datasets/private/landmarks.csv --output models/trained_gaze.json
```

The new model is **OFF BY DEFAULT**, cannot override a baseline LOOKING AWAY to ACTIVE, and must be evaluated on independent volunteers before practical use. For opting in and installing the trained JSON, read `TRAINING_GUIDE.md`.

## Datasets: what's actually supplied

This project contains a **working data collection and training pipeline**, NOT pre-downloaded third-party datasets. MPIIGaze, MPIIFaceGaze, ETH-XGaze and GazeCapture have research/non-commercial licensing restrictions. WEyeDS distribution terms must also be checked. Download any external dataset from its official site; you **cannot** train directly on their heterogeneous raw ZIP archives using the numeric CSV trainer without format-specific preprocessing and labels. The training CSV created by our own collector **is directly compatible**.

## Security and limitations

- Data collection requires explicit consent. The collector stores numerical landmark data only; never publish participants' data without explicit permissions.
- Calibration-free eye direction **does not prove** looking at the exact physical camera lens. Near-lens screen gaze can be indistinguishable for RGB webcams.
- Automated tests are not real-world accuracy measurements. Evaluation across separate users, cameras and lighting conditions is mandatory before commercial deployment.
- No remote upload, no web accounts, no real biometric data shipped, no licensed third-party model weights bundled.
- The app is not a basis for automatic cheating accusations or disciplinary decisions.
