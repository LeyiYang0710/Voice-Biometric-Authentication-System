# ML Software Interface Specification

**Version:** 1.0  
**Date:** 27 October 2026  
**Status:** Implemented and verified on Windows

## 1. Purpose and scope

This document defines the implemented `identify.py` command-line interface and Python API for integrating CAM++ and fixed TDNN speaker-identification models into the application.

The caller supplies a complete WAV file. The module validates the audio, extracts an embedding using the selected model, compares it with enrolled member templates, and returns a structured result.

The default mode ranks enrolled members. Unknown-speaker rejection requires independent threshold calibration. KV260 deployment has not yet been validated.

### Responsibilities

| Component | Responsibility |
|---|---|
| Calling application | Recording, resampling, completing file writes, scheduling requests, and displaying results. |
| ML module | WAV validation, window selection, embedding extraction, template matching, and result formatting. |

Hardware registers, DMA, buffer ownership, and hardware quality statistics are outside this interface.

```text
Complete WAV file
    |
    v
Audio validation
    |
    v
Window selection from the template database
    |
    v
CAM++ or TDNN embedding extraction
    |
    v
Comparison with enrolled member templates
    |
    v
Structured identification result
```

## 2. Audio input

| Item | Requirement |
|---|---|
| Container | Standard RIFF/WAVE. The usual extension is `.wav`. |
| Encoding | Uncompressed signed 16-bit PCM, little-endian. |
| Sample rate | **16,000 Hz**. |
| Channels | **Mono**. |
| Minimum duration | **3 seconds**, or 48,000 samples. |
| Validity | Complete, readable, non-empty, and not entirely zero-valued. |
| Filename | Any valid filename, such as `query_001.wav`. It does not need to contain the speaker's identity. |
| Request identifier | Optional `recording_id`. If omitted, the filename without its extension is used. |

The caller must finish and close the file before submitting it.

- The entry does not resample audio or mix channels. The caller must satisfy the input format.
- M4A, MP3, floating-point WAV, and raw PCM renamed to `.wav` are not accepted.
- The implementation does not impose a maximum duration. The calling application should limit recording length according to its memory and response-time budget.
- This entry does not implement calibrated noise, clipping, or voice activity detection gates. Passing validation does not establish that the recording contains speech.

## 3. Models and template databases

The integration examples use these fixed combinations:

| Model | Template database | Model file |
|---|---|---|
| CAM++ | `data/templates/campplus_multiwindow.json` | The matching ONNX model specified by the database. |
| Original fixed TDNN baseline | `data/templates/linear10_multiwindow.json` | `results/small_tdnn_run01/best.pt` |

These databases currently contain **member02, member03, and member04**. Each call must explicitly select a database. Other TDNN versions require their own matching weights and database; the entry does not automatically choose a deployment model.

### Database requirements

The database JSON must contain:

```text
model_sha256
model_path
strategy
members
```

At initialization, the implementation checks the model SHA-256 against the database and validates template dimensions, finite values, and unit-vector norms. Templates must not be mixed across different model hashes or window strategies.

The older full-recording `data/templates_demo.json` lacks the required window metadata and is not accepted by this unified entry.

### Window strategies

| Strategy | Processing |
|---|---|
| `center3s` | Extract the middle 3 seconds and normalize its embedding. |
| `multiwindow` | Use 3-second windows with a 1.5-second hop and include the final window. Normalize each window embedding, average the embeddings, and normalize the recording embedding before template matching. |

The strategy is read from the database. The caller should submit the complete recording rather than selecting windows itself.

### Runtime environments

On the currently verified Windows setup:

- `.venv` runs CAM++.
- `.venv-train` runs TDNN inference on CPU. Its name does not start training.
- The entry does not train models or modify templates.
- KV260 requires compatible ARM64 dependencies and separate deployment validation.

## 4. Command-line interface

Run the following commands from the project root. Replace the example WAV path with the recording to identify.

### CAM++

```powershell
.\.venv\Scripts\python.exe .\identify.py `
  --wav .\data\processed\member02\session02\member02_session02_001.wav `
  --model campplus `
  --db .\data\templates\campplus_multiwindow.json
```

### Original fixed TDNN baseline

```powershell
.\.venv-train\Scripts\python.exe .\identify.py `
  --wav .\data\processed\member02\session02\member02_session02_001.wav `
  --model tdnn `
  --checkpoint .\results\small_tdnn_run01\best.pt `
  --db .\data\templates\linear10_multiwindow.json
```

### Arguments

| Argument | Required | Meaning |
|---|---|---|
| `--wav` | Yes | Path to the complete input WAV. |
| `--model` | Yes | `campplus` or `tdnn`. |
| `--db` | Yes | Template JSON matching the model and preprocessing strategy. |
| `--model-path` / `--checkpoint` | No | Two names for the same argument. Overrides the model location stored in the database; the hash must still match. |
| `--recording-id` | No | Caller-provided identifier for the recording. |
| `--threshold` | No | Finite cosine threshold in `[-1, 1]`. Supply only after independent calibration; omit for ranking only. |

Relative WAV and database paths are resolved from the current working directory. A relative model path is resolved from the directory containing `identify.py`.

Existing databases contain Windows absolute model paths. When moving the package, supply the new location with `--model-path` or `--checkpoint`. The SHA-256 check still applies.

## 5. Persistent Python API

Initialize the identifier once, then process recordings sequentially in the same inference worker. This avoids reloading the model for every recording.

```python
from identify import SpeakerIdentifier, InvalidAudio

identifier = SpeakerIdentifier(
    model_type="campplus",
    database_path="data/templates/campplus_multiwindow.json",
    # model_path=None, threshold=None
)

result = identifier.identify(
    "query_001.wav",
    recording_id="query_001",
)
```

Signatures:

```python
SpeakerIdentifier(model_type, database_path, model_path=None, threshold=None)
identifier.identify(wav_path, recording_id=None)
```

- The Python API returns a `dict`.
- Invalid recordings raise `InvalidAudio`. Configuration and runtime errors propagate to the caller.
- The CLI converts expected runtime failures into JSON.
- Use one identifier instance sequentially from one inference worker.
- This module does not implement an HTTP/socket service, task queue, or `busy` status.

## 6. Runtime status and exit codes

| Exit code | Status | Meaning |
|---|---|---|
| `0` | `ok` | Identification completed. `Unknown` in thresholded mode is also a successful operation. |
| `2` | `invalid` | Incompatible audio format, fewer than 3 seconds, empty audio, entirely zero-valued audio, or an undecodable recording. |
| `1` | `error` | Missing file, incompatible model/database, missing dependency, or model runtime failure. |

Consumers must check `status` before reading fields that exist only in successful results.

An `invalid` or `error` result contains the following fields:

```json
{
  "recording_id": "query_001",
  "status": "invalid",
  "identity": null,
  "scores": {},
  "reason": "Recording must be at least 3 seconds",
  "model_id": "campplus"
}
```

The CLI writes result JSON to stdout. Command-line syntax errors, such as missing required arguments, use argparse's usage message on stderr rather than this JSON schema.

## 7. Successful result schema

The following is a complete successful result example. Similarities come from a verified Windows CAM++ query. Timing values are illustrative and must not be treated as KV260 measurements.

```json
{
  "recording_id": "query_001",
  "status": "ok",
  "identity": "member02",
  "best_member": "member02",
  "scores": {
    "member02": 0.82385969,
    "member03": 0.47385219,
    "member04": 0.36891297
  },
  "reason": null,
  "model_id": "campplus",
  "model_sha256": "aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2",
  "template_db_id": "campplus_multiwindow.json",
  "template_db_sha256": "6e198239821bee6eb6cabaaebc4f316ddb3d7acf855ddfb6c73e10c9fad0311d",
  "strategy": "multiwindow",
  "decision_mode": "ranking_only",
  "threshold": null,
  "audio_seconds": 8.3826875,
  "window_count": 5,
  "processing_seconds": 0.15,
  "timings": {
    "startup_seconds": 1.87,
    "audio_preparation_seconds": 0.002,
    "embedding_seconds": 0.148,
    "matching_seconds": 0.00002
  }
}
```

| Field | Meaning |
|---|---|
| `recording_id` | Identifier associating the input recording with its result. |
| `status`, `reason` | Runtime status and failure reason; `reason=null` on success. |
| `identity` | Decision: a member ID, or `Unknown` in thresholded mode. |
| `best_member` | Highest-scoring enrolled member, retained even when the decision is `Unknown`. |
| `scores` | Cosine similarities for enrolled members. **These are not probabilities.** |
| `model_id`, `model_sha256` | Model family and exact weights fingerprint. |
| `template_db_id`, `template_db_sha256` | Database filename and content fingerprint. |
| `strategy`, `window_count`, `audio_seconds` | Window strategy, number of windows, and input duration. |
| `decision_mode`, `threshold` | Decision mode and optional rejection threshold. |
| `processing_seconds`, `timings` | Per-recording processing time, initialization time, and stage timings. |

### Decision modes

By default, `decision_mode="ranking_only"` and `threshold=null`. The module returns the closest enrolled member. An unknown speaker will still receive an enrolled-member ranking; this is not a calibrated authentication decision.

Supplying `--threshold` enables `thresholded` mode:

- A best score **strictly below** the threshold returns `identity="Unknown"`.
- A best score equal to or above the threshold returns the highest-scoring member.

Calibrate the threshold independently for the exact model, templates, window strategy, and input conditions. The implementation validates the numeric range, not whether calibration has been performed.

### Timing definitions

- `processing_seconds` includes audio reading/window preparation, embedding extraction, and template matching. It excludes initialization.
- `timings.startup_seconds` records model and database initialization separately.
- None of these fields includes hardware recording time or establishes board performance.

## 8. Delivery files

| File | Purpose |
|---|---|
| `identify.py` | Unified CLI and Python API. |
| `voiceid.py` | CAM++ extractor initialization used by the unified entry. |
| `small_model.py` | TDNN network definition. |
| Selected model | CAM++ ONNX model or fixed TDNN `.pt` checkpoint. |
| Matching template JSON | Member templates for the exact model and strategy. |
| Dependency instructions | Runtime environment and installation requirements. |
| This specification | Input, invocation, output, error semantics, and validation scope. |
| Authorized test WAV and reference result | Integration checks for the calling application. |

The existing Windows dependency lock files are not KV260 installation recipes. In particular, the TDNN environment includes a CUDA build. ARM64 dependencies require separate validation; copying a Windows virtual environment is not a deployment method.

## 9. Verification and limitations

### Completed checks

- CAM++ and the original TDNN baseline each reproduced **40 saved session02 query results** across the center-window and multiwindow strategies.
- The maximum difference in existing member02/member03 scores was approximately **1.79e-7**.
- **Seven invalid-audio cases** were checked in each runtime.
- Model/database hash mismatch, threshold branches, CLI JSON, and exit codes were checked.
- Existing models, templates, and original source files were checked for unintended changes.
- Member04 enrolment recordings were used only for an inference smoke check, not independent accuracy evaluation.
- Session03 recordings were not used in these interface checks.

### Remaining work

- Independent unknown-speaker threshold calibration.
- Validation with recordings from the actual capture device.
- KV260 ARM64 deployment and performance measurements.

Threshold branch tests do not establish a calibrated rejection threshold. These interface checks do not establish three-member recognition accuracy or board performance.

## 10. Implementation provenance

`identify.py` integrates the existing model implementations. CAM++ extraction reuses `voiceid.py` and sherpa-onnx. TDNN extraction reuses the existing project `SmallTDNN` implementation and saved weights. This interface does not reimplement or train CAM++.

This specification follows the implemented `identify.py`, the project `INFERENCE_INTERFACE.md`, and the recorded interface verification results.
