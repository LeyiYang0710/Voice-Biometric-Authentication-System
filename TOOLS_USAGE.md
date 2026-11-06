**Software tools usage guide**

This guide covers the repository-root entry `identify.py`, the audio converter `prepare_audio.py`, and the enrollment tool `enrol.py`. Run all commands from the repository root.

Use one Python environment containing both model families' dependencies. Windows examples use the existing `.venv-train` environment; its name does not start training.

| Tool | Input | Output |
|---|---|---|
| `prepare_audio.py` | A recording supported by soundfile | A mono, 16 kHz, PCM16 WAV and conversion report |
| `enrol.py` | Enrollment CSV, member list and model selection | A new speaker-template database |
| `identify.py` | Query WAV and model selection | Identity, similarity scores and runtime metadata as JSON |

Typical flow: recording → conversion if needed → enrollment manifest → templates → development calibration and profile freeze → identification.

For ordinary identification with an existing fixed profile, start at step 2. To test newly generated templates, use step 6.

**1. Check the environment and working directory.**

Open PowerShell in the project directory and confirm these files are present:

```powershell
Get-Item .\identify.py, .\prepare_audio.py, .\enrol.py
```

Copy commands only, without the `PS ...>` or `>>` terminal prompts. Complete each step before continuing.

On the current Windows machine, use the configured interpreter:

```powershell
.\.venv-train\Scripts\python.exe --version
```

If the existing TDNN environment lacks only the CAM++ backend, install these two packages. This command assumes NumPy, SoundFile, SciPy, PyTorch and torchaudio are already installed:

```powershell
.\.venv-train\Scripts\python.exe -m pip install --no-deps sherpa_onnx==2.0.0 sherpa-onnx-core==2.0.0
```

Fresh Windows setup only: create an environment and install both existing dependency lists. Skip this block if the environment already runs both models.

```powershell
py -3.12 -m venv .venv-train
.\.venv-train\Scripts\python.exe -m pip install -r .\ml\tdnn\requirements.txt -r .\requirements-inference.lock.txt
```

The CAM++ ONNX file is stored with Git LFS. If a fresh checkout has not retrieved the model contents, install Git LFS, then run:

```powershell
git lfs install
git lfs pull
```

The small LFS pointer file cannot be used as the model itself.

The example audio paths refer to the existing dataset. Audio and local manifests under `data/` are not included in a normal source checkout. Supply your own recordings or obtain the prepared dataset and manifest.

**2. Switch the default identification model with `--model`.**

Use this existing development recording for a smoke check. To identify another recording, replace the path with a real WAV file.

```powershell
$testWav = ".\data\processed\member02\session02\member02_session02_001.wav"

.\.venv-train\Scripts\python.exe .\identify.py --model campplus --wav $testWav
.\.venv-train\Scripts\python.exe .\identify.py --model tdnn --wav $testWav
```

For ordinary fixed-profile use, supply only the model and recording. The entry loads the matching weights, templates, threshold and conditions ID automatically. Do not supply `--db` or `--threshold` for this mode.

| Model | Weights | Templates | Development threshold |
|---|---|---|---:|
| `campplus` | `models/onnx/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx` | `ml/campplus/templates/campplus_multiwindow.json`, 192 dimensions | `0.7719998955726624` |
| `tdnn` | `ml/tdnn/checkpoints/libri251_m03_20h/best.pt` | `ml/tdnn/templates/libri251_fp32_multiwindow.json`, 64 dimensions | `0.5757178068161011` |

Thresholds are read from `ml/campplus/frozen_threshold.json` and `ml/tdnn/frozen_libri251_fp32.json` respectively.

Both models are on the `ml_TDNN` branch. Switching these CPU inference paths does not require switching Git branches or rebuilding the FPGA bitstream.

Example CAM++ result excerpt for this recording; other recordings will produce different scores and may produce different identities:

```json
{
  "recording_id": "member02_session02_001",
  "status": "ok",
  "identity": "member02",
  "best_member": "member02",
  "model_id": "campplus",
  "decision_mode": "thresholded",
  "threshold": 0.7719998955726624,
  "scores": {
    "member02": 0.823859691619873,
    "member03": 0.4738521873950958,
    "member04": 0.3689129650592804,
    "member05": 0.23832155764102936
  },
  "window_count": 5,
  "scope": "public_audio_desktop_development_only",
  "deployment_approved": false
}
```

For the same recording, TDNN also returns member02, with a member02 score of `0.6997263431549072` and its own threshold of `0.5757178068161011`.

Similarity scores are not probabilities and are not directly comparable between models.

**3. Convert input audio when necessary.**

The model input contract is mono, 16 kHz, PCM16 WAV, at least three seconds and not entirely zero.

Use an existing enrollment recording to try the converter. The output uses a new filename and preserves the source:

```powershell
.\.venv-train\Scripts\python.exe .\prepare_audio.py --input .\data\processed\member02\session01\member02_session01_001.wav --output .\data\converted\format_demo.wav
```

This recording already meets the contract, so it is copied exactly. Other supported inputs are converted as needed, without automatic loudness normalization.

| Option | Meaning |
|---|---|
| `--input` | Source recording |
| `--output` | New `.wav` destination |
| `--channel 0` | Select the first channel; indexing starts at zero |
| `--mixdown` | Average channels into mono |
| `--allow-clipping` | Explicitly allow out-of-range converted samples to be clipped into the PCM16 range |

Multichannel audio requires an explicit channel selection or mixdown. These options are mutually exclusive. Decoder support depends on soundfile; arbitrary M4A input is not guaranteed.

A successful conversion generates:

```text
data/converted/format_demo.wav
data/converted/format_demo.conversion.json
```

Example conversion report structure for the recording above. Paths and hashes are placeholders here; the actual report contains full values. Do not use example JSON to replace generated files.

```json
{
  "source": "<absolute source WAV path>",
  "output": "<absolute output WAV path>",
  "source_sha256": "<source SHA-256>",
  "output_sha256": "<output SHA-256>",
  "source_rate": 16000,
  "source_channels": 1,
  "selected_channel": null,
  "mixdown": false,
  "output_rate": 16000,
  "output_channels": 1,
  "subtype": "PCM_16",
  "output_samples": 125931,
  "seconds": 7.8706875,
  "converted_clipping_samples": 0,
  "amplitude_normalization": false,
  "source_unchanged": true
}
```

The source and output hashes match for this exact-copy example. When conversion changes the audio representation, the output hash will usually differ.

**4. Prepare the enrollment manifest.**

`enrol.py` reads a UTF-8 CSV describing which recording belongs to which member. It requires these four columns. Relative audio paths are resolved from the CSV directory.

| Column | Content |
|---|---|
| `path` | Path to compliant enrollment audio |
| `recording_id` | A recording identifier unique within the manifest |
| `speaker_id` | The member identity, such as member02 |
| `split` | Must be `enrollment` |

Format example for a CSV stored under `data/manifests/`. These two rows illustrate the structure; they are insufficient for the normal enrollment minimum.

```csv
path,recording_id,speaker_id,split
../processed/member02/session01/member02_session01_001.wav,member02_session01_001,member02,enrollment
../processed/member02/session01/member02_session01_002.wav,member02_session01_002,member02,enrollment
```

The local delivery bundle includes `data/manifests/mvp_enrollment_four.csv`: 10 recordings each for member02/03/04/05, 40 recordings in total. This local manifest is not part of a normal source checkout. Obtain it with the dataset, or prepare a manifest for your own recordings.

If conversion was required, reference the converted files in the manifest. The existing 40 recordings are already compliant, so the supplied manifest references the original `processed/` files directly.

Enrollment rejects `session02`, `session03` and `heldout` paths, repeated recording identifiers, and duplicate PCM content. Renaming one recording does not make it an independent recording.

**5. Enroll using the selected model.**

Use the current four-member manifest to validate the tools. These commands process the same recordings but generate separate databases for CAM++ and TDNN:

```powershell
.\.venv-train\Scripts\python.exe .\enrol.py --model campplus --manifest .\data\manifests\mvp_enrollment_four.csv --output .\data\templates\campplus_enrollment_demo.json --conditions-id desktop_members_mic_unverified_v1 --members member02 member03 member04 member05

.\.venv-train\Scripts\python.exe .\enrol.py --model tdnn --manifest .\data\manifests\mvp_enrollment_four.csv --output .\data\templates\tdnn_enrollment_demo.json --conditions-id desktop_members_mic_unverified_v1 --members member02 member03 member04 member05
```

| Option | Meaning |
|---|---|
| `--model` | `campplus` or `tdnn`; uses the existing fixed model weights |
| `--manifest` | Enrollment CSV |
| `--output` | New template JSON; existing files are refused |
| `--conditions-id` | Recording-condition label for this enrollment batch |
| `--members` | Members to enroll; defaults to member01 through member05 |
| `--min-recordings` | Minimum recordings per member; default 10 |

The conditions ID records the intended capture setup. It does not automatically detect microphones or environments.

Example CAM++ enrollment result. Path formatting and hashes depend on the actual generated file:

```json
{
  "output": "data/templates/campplus_enrollment_demo.json",
  "model_type": "campplus",
  "members": ["member02", "member03", "member04", "member05"],
  "recordings": 40,
  "template_db_sha256": "<generated template SHA-256>",
  "completeness": "partial_diagnostic_templates",
  "threshold_requires_recalibration": true
}
```

`partial_diagnostic_templates` describes a four-member diagnostic database, not a failure. TDNN uses the same result structure with a different model name and output file.

The generated database contains:

| Field | Meaning |
|---|---|
| `model_type` | Selected model |
| `model_sha256` | Model-file fingerprint |
| `model_path` | Model path relative to the repository root |
| `strategy` | `multiwindow` processing |
| `members` | Unit speaker-template vectors keyed by member identity |
| `enrollment_conditions_id` | Enrollment recording conditions |
| `enrollment_manifest_sha256` | Manifest fingerprint |
| `enrollment_recordings` | Recording IDs, member IDs, audio-file hashes and PCM hashes |
| `completeness` | Complete or partial enrollment |
| `threshold_requires_recalibration` | Indicates that a matching calibration is required |
| `inference_branch` | TDNN branch metadata, set to `fp32` for this tool |

The `members` object has this structure:

```text
CAM++:
  member02 -> 192 floats
  member03 -> 192 floats
  member04 -> 192 floats
  member05 -> 192 floats

TDNN:
  each member -> 64 floats
```

These vectors must be generated by the tool. Inspect member identities and vector dimensions with:

```powershell
$newDatabase = Get-Content .\data\templates\campplus_enrollment_demo.json -Raw | ConvertFrom-Json
$newDatabase.members.PSObject.Properties.Name
$newDatabase.members.member02.Count
```

The final line should print `192` for CAM++ or `64` for a TDNN database.

Enrollment writes a new database. It does not retrain model weights, replace the current default templates, or activate a new threshold profile.

**6. Diagnose newly generated templates with an explicit `--db`.**

Use an independent session02 recording to query each new database:

```powershell
$testWav = ".\data\processed\member02\session02\member02_session02_001.wav"

.\.venv-train\Scripts\python.exe .\identify.py --model campplus --db .\data\templates\campplus_enrollment_demo.json --wav $testWav
.\.venv-train\Scripts\python.exe .\identify.py --model tdnn --db .\data\templates\tdnn_enrollment_demo.json --wav $testWav
```

For the existing local dataset, both examples should return member02. Result excerpt:

```json
{
  "status": "ok",
  "identity": "member02",
  "best_member": "member02",
  "decision_mode": "ranking_only",
  "threshold": null
}
```

An explicit database without a threshold runs ranking only. It selects the highest-scoring enrolled member and does not provide calibrated Unknown rejection.

| Invocation | Behavior |
|---|---|
| `--model` and `--wav` only | Uses the current fixed templates and calibrated threshold; returns a member or Unknown |
| Explicit `--db`, without `--threshold` | Ranks identities using the specified database |
| Explicit `--db` and `--threshold` | Uses a caller-supplied threshold; calibration is the caller's responsibility |

Promoting a new database requires separate development recordings, threshold selection or validation, updated matching bindings, and verification. Copying an old threshold or editing a fingerprint alone does not complete calibration. These tools do not calibrate or publish a profile automatically.

**7. Complete five-member enrollment after collecting the missing data.**

Prepare `data/manifests/mvp_enrollment_five.csv` with compliant enrollment recordings for member01 through member05, at least 10 recordings each. The supplied local bundle contains only the four-member manifest.

Do not reuse development or held-out recordings to fill missing enrollment slots.

Run the following only after the five-member manifest and recordings are available. Omitting `--members` enables the default five-member requirement:

```powershell
.\.venv-train\Scripts\python.exe .\enrol.py --model campplus --manifest .\data\manifests\mvp_enrollment_five.csv --output .\data\templates\campplus_five_candidate.json --conditions-id kv260_mic_v1
```

Use `kv260_mic_v1` only if it truthfully describes the recording conditions. Missing data causes enrollment to fail without generating the database. Successful five-member enrollment reports `five_member_templates` and still requires subsequent calibration before default-profile activation.

**8. Use the same model options on Linux.**

Run the following from the repository root in a Python 3.12 environment with both backends installed. The example recordings and manifest must also exist on that machine.

The Windows virtual environment cannot be copied to ARM64 Linux. ARM64 package installation and board behavior still require validation; the Windows lock files do not establish a verified ARM deployment recipe.

```bash
python3 identify.py --model campplus --wav data/processed/member02/session02/member02_session02_001.wav
python3 identify.py --model tdnn --wav data/processed/member02/session02/member02_session02_001.wav

python3 prepare_audio.py --input data/processed/member02/session01/member02_session01_001.wav --output data/converted/linux_format_demo.wav

python3 enrol.py --model campplus --manifest data/manifests/mvp_enrollment_four.csv --output data/templates/linux_campplus_enrollment_demo.json --conditions-id desktop_members_mic_unverified_v1 --members member02 member03 member04 member05
python3 enrol.py --model tdnn --manifest data/manifests/mvp_enrollment_four.csv --output data/templates/linux_tdnn_enrollment_demo.json --conditions-id desktop_members_mic_unverified_v1 --members member02 member03 member04 member05
```

**9. Read results and handle common failures.**

`identify.py` reports runtime results as follows:

| Status | Meaning | Exit code |
|---|---|---:|
| `ok`, member identity | Member accepted | 0 |
| `ok`, `Unknown` | The recording's highest score is below the configured threshold | 0 |
| `invalid`, `identity: null` | Invalid audio format, duration or content | 2 |
| `error`, `identity: null` | File, profile, model or runtime failure | 1 |

Argument syntax errors use argparse's stderr output, normally with exit code 2. Conversion and enrollment commands print JSON on success but report input failures as exceptions and exit. They do not silently skip invalid files.

| Symptom | Action |
|---|---|
| `No module named numpy` | Select the configured `.venv-train` interpreter |
| Missing torch or sherpa-onnx | Install the selected backend's dependencies in the same environment |
| Missing WAV or CSV | Supply real files and verify paths; audio is not in the source checkout |
| `Multichannel audio requires...` | Choose `--channel 0` or `--mixdown` |
| `Output ... new files` or `Output already exists` | Choose a new output name; existing files are preserved |
| `Missing enrollment recordings` | Supply missing data, or explicitly select four members for the diagnostic workflow |
| `Development/held-out recordings...` | Use enrollment data only; do not relabel evaluation data |
| `Duplicate PCM` | Remove duplicate entries and use independent recordings |
| `Frozen ... changed` or fingerprint mismatch | Restore matching assets, check LFS model retrieval, or perform reviewed recalibration for a new database |
| `ranking_only` for a new database | Expected when `--db` is supplied without a threshold |

Check `status` first, then read `identity` after success. `best_member` is only the highest-scoring enrolled member and may still be present when the final identity is Unknown.

`processing_seconds` excludes model initialization. `timings.startup_seconds` records initialization separately. A new command also incurs Python startup, imports and profile checks.

Both fixed profiles currently contain four members and use development thresholds. `deployment_approved: false` records that scope; it is not a command failure. The CLI does not perform DMA capture, PL quality-statistics readout or hardware-accelerated convolution.

**10. Commit the three source and documentation files.**

```powershell
git add -- prepare_audio.py enrol.py TOOLS_USAGE.md
git diff --cached --name-only
```

Confirm that the staged files contain only the intended changes, then run:

```powershell
git commit -m "Add audio conversion and enrollment tools with usage guide"
git push origin ml_TDNN
```

Local manifests, converted audio and diagnostic outputs remain under ignored `data/`. The currently released model assets are tracked separately; generated candidates do not activate themselves.
