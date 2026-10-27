# TDNN inference package

This directory groups the inference entry, TDNN network definition, fixed
checkpoints, template databases and integration documentation. It is an
inference snapshot; existing training and evaluation directories remain intact.

## Contents

```text
ml/tdnn/
  identify.py                 Unified inference entry
  small_model.py              TDNN network definition
  ML_SOFTWARE_INTERFACE.md    English interface specification
  requirements.txt           Direct dependency versions
  manifest.json              Model, database and code hashes
  checkpoints/
    <run>/best.pt             Six fixed model versions
  templates/
    <model>_<strategy>.json   Ten matching three-member databases
```

Database model paths are relative to this directory. Moving the complete
folder preserves these paths. Do not mix checkpoints and template databases.

## Run from the repository root

Use the existing Windows inference-capable environment:

```powershell
.\.venv-train\Scripts\python.exe .\ml\tdnn\identify.py `
  --wav .\data\processed\member02\session02\member02_session02_001.wav `
  --model tdnn `
  --db .\ml\tdnn\templates\linear10_multiwindow.json
```

This example uses the original fixed TDNN baseline. Other checkpoints are
available through their matching database; no deployment model is selected
automatically. The database supplies the model path and window strategy.

Input must be WAV, 16 kHz, mono, PCM16 and at least 3 seconds. Output is JSON.
Without `--threshold`, results are enrolled-member rankings, not calibrated
unknown-speaker rejection. See `ML_SOFTWARE_INTERFACE.md` for the complete
schema, error handling and persistent Python API.

`identify.py` supports both model families in the full project, but this folder
packages TDNN only. CAM++ also requires `voiceid.py`, its ONNX model and its
runtime dependencies; use the repository-root entry for CAM++.

## Dependencies

`requirements.txt` records the direct package versions used by the existing
Windows TDNN runtime, without pinning its CUDA wheel suffix. It is not a
complete platform lock or a validated KV260 installation recipe. For the
currently tested machine, use the existing `.venv-train` environment. ARM64
deployment still requires separate dependency and performance validation.

## GitLab files

Upload the source files, this README, the interface specification,
`requirements.txt`, `.gitignore` and `manifest.json`.

`checkpoints/` and `templates/` are deliberately ignored by Git, matching the
project's existing separation of code from model and enrolment artifacts.
They are present locally so the package can run. A source-only checkout will
not run until the matching weights and templates are supplied separately.
Do not force-add enrolment templates as part of an ordinary source upload.
No recordings or virtual environments are included in this directory.

## Verification

Checkpoints are copied without modification. Template member vectors and
model hashes are preserved; only `model_path` is changed to a relative path.
`manifest.json` records the source and packaged hashes.

Both center-window and multiwindow input handling use the same implementation
as the repository-root entry. Packaging checks verify all relative model
paths, compare source and packaged vectors, and reproduce a known independent
session02 query. No training occurs during packaging or inference.
