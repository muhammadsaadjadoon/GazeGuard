# Official gaze dataset sources (not bundled)

These **third-party datasets are NOT present in this ZIP**. Their authors control the distribution and licences. Each needs to be obtained directly from the official site, under its actual usage conditions. **Do not commit raw data to GitHub.**

| Source | Official access | Relevance | Important usage note |
|---|---|---|---|
| WEyeDS | https://robbinslab.github.io/weyeds/ | Desktop/laptop webcam gaze study | Check and follow the licence/consent conditions on the distribution page before downloading or reusing |
| MPIIGaze | https://www.collaborative-ai.org/research/datasets/MPIIGaze/ | Real-world laptop gaze | Non-commercial scientific use; **not pre-approved for a commercial launch** |
| MPIIFaceGaze | https://www.collaborative-ai.org/research/datasets/MPIIFaceGaze/ | Full face + gaze annotation | Non-commercial scientific use |
| ETH-XGaze | https://xgaze.ait.ethz.ch/ | Broad head pose and gaze diversity | Registration/data use agreement; commercial use prohibited |
| GazeCapture | https://gazecapture.csail.mit.edu/download.php | Mobile gaze capture and screen target points | Research-only licence, including restrictions on derived trained models |

**How to use with THIS release:** the training code expects a standardized GazeGuard CSV of **eight MediaPipe landmark features and explicit `lens`, `screen`, `left`, etc. labels**. The dataset formats and labels above differ, so simply dropping their ZIPs in the folder will NOT train GazeGuard. No unverified automatic converters for those datasets are included. Public gaze-direction datasets alone do not establish exact camera-lens fixation labels for your webcam.

The fully working training workflow included here is the **consent-based custom webcam data collection** (`training/collect_webcam.py`) and fitting/validation of its resulting CSV (`training/train_landmarks.py`). Raw face images are **not** collected. The feature CSV is still potentially sensitive biometric-related information: obtain consent, restrict access, and delete it when no longer needed.
