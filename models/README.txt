Models created by training/train_landmarks.py are plain JSON, not pickle.
Default output: models/trained_gaze.json
To deploy, copy this file to the app data directory printed by `python -m gazeguard.cli doctor` as `trained_gaze.json`.
Then set GAZEGUARD_ENABLE_TRAINED_GAZE=1 before starting run.py.
This is optional and must be tested on users/devices not in training.
