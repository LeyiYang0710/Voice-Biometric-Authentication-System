# ML Software Interface Specification

Version 1.0 | 27 October 2026

This specification describes the implemented `identify.py` CLI and Python API.
It supersedes the earlier proposal that a unified entry still needed to be
created. The scope is software integration only: the caller supplies a complete
WAV file and receives a speaker-identification result.

`identify.py` integrates the existing model implementations. CAM++ extraction uses
the existing `voiceid.py` and sherpa-onnx library. TDNN extraction uses the
existing project `SmallTDNN` network and saved weights. This entry never trains
models or updates templates.

## Input and model selection

Input must be a readable WAV file with 16 kHz, mono, PCM16 audio of at least
3 seconds. Empty, completely silent, undecodable and incompatible recordings
are rejected. There is no calibrated noise, clipping or speech-presence gate
in this entry. Acquisition and resampling remain the caller's responsibility.

Use a database under `data/templates/` containing `model_sha256`, `model_path`,
`strategy` and `members`. The old full-recording `data/templates_demo.json`
does not contain this strategy metadata and is not accepted by this entry.
The model hash must match the database. Old member vectors remain unchanged.

The database selects the preprocessing strategy:

- `center3s`: use the center 48,000 samples.
- `multiwindow`: use 48,000-sample windows with a 24,000-sample hop and include
  the final window. Normalize each window embedding, average, then normalize
  the recording embedding. Match this vector against the stored unit templates.

## Commands on the current Windows machine

Run from the project directory. The two existing environments contain different
libraries: `.venv` supports CAM++; `.venv-train` supports TDNN inference on CPU.
Using the environment named `train` does not start training. On KV260, install
compatible ARM64 dependencies; board deployment has not been validated.

CAM++ with the three-member multiwindow database:

```powershell
.\.venv\Scripts\python.exe .\identify.py --wav .\data\processed\member02\session02\member02_session02_001.wav --model campplus --db .\data\templates\campplus_multiwindow.json
```

Original fixed TDNN baseline with the corresponding database:

```powershell
.\.venv-train\Scripts\python.exe .\identify.py --wav .\data\processed\member02\session02\member02_session02_001.wav --model tdnn --checkpoint .\results\small_tdnn_run01\best.pt --db .\data\templates\linear10_multiwindow.json
```

Other fixed TDNN versions can use their own matching database. For example,
`small_tdnn_amsoftmax_m02_200ep_isolated_run01_multiwindow.json` selects that
checkpoint from its `model_path`. This does not select or promote a deployment
model automatically. If files move, `--model-path` (alias `--checkpoint`)
overrides the stored path; the SHA-256 check still applies.

| Argument | Required | Meaning |
|---|---|---|
| `--wav` | Yes | Path to a complete input WAV |
| `--model` | Yes | `campplus` or `tdnn` |
| `--db` | Yes | Compatible template database JSON |
| `--model-path` / `--checkpoint` | No | Model location override with unchanged hash |
| `--recording-id` | No | Caller identifier; defaults to WAV filename stem |
| `--threshold` | No | Finite cosine threshold in [-1, 1]; omit for ranking |

WAV and database relative paths are resolved from the current working
directory. A relative model path is resolved from the directory containing
`identify.py`. Existing database model paths are Windows absolute paths; use
`--model-path` when moving the package to another location.

## JSON output and exit codes

`status=ok` includes `recording_id`, `identity`, `best_member`, `scores`,
`model_id`, model/database SHA-256 values, `strategy`, `decision_mode`,
`threshold`, audio length, window count and timing fields.

Without `--threshold`, `decision_mode=ranking_only`, `threshold=null`, and
`identity` reports the highest-scoring enrolled member. This is closed-set
ranking, not unknown-speaker rejection or a calibrated authentication decision.

Only supply `--threshold` after independent calibration for the exact model,
templates, preprocessing and input conditions. The code validates the numeric
range, not the calibration. In thresholded mode a best score strictly below
the threshold returns `Unknown`. Similarities are not probabilities.

Runtime input failures return JSON with `identity=null`, an empty `scores`
object and a readable `reason`. Standard command-line syntax errors use
argparse's normal usage message on stderr.

The success result contains these fields:

| Fields | Meaning |
|---|---|
| `recording_id`, `status`, `reason` | Request association and runtime status |
| `identity`, `best_member`, `scores` | Decision, highest-scoring member and cosine similarities |
| `model_id`, `model_sha256` | Model family and exact weights fingerprint |
| `template_db_id`, `template_db_sha256` | Database filename and content fingerprint |
| `strategy`, `window_count`, `audio_seconds` | Preprocessing strategy and input dimensions |
| `decision_mode`, `threshold` | Ranking-only or explicit threshold mode |
| `processing_seconds`, `timings` | Per-call time plus initialization and stage timings |

The runtime error/invalid result is smaller: `recording_id`, `status`,
`identity`, `scores`, `reason`, `model_id`. Consumers must check `status`
before reading success-only fields. The CLI prints the result as JSON to
stdout. No `busy` status or networking service is implemented here.

| Exit code | Status | Meaning |
|---|---|---|
| 0 | ok | Identification completed, including Unknown in thresholded mode |
| 2 | invalid | Audio violates the input contract |
| 1 | error | File, configuration, model or runtime dependency failure |

`processing_seconds` measures one call's audio preparation, embedding and
matching time. `timings.startup_seconds` separately records model/database
initialization and is excluded from per-recording processing time. These values
do not include hardware capture time or claim KV260 performance.

## Persistent use by the inference worker

```python
from identify import SpeakerIdentifier

identifier = SpeakerIdentifier(
    model_type="campplus",
    database_path="data/templates/campplus_multiwindow.json",
)
# Load once, then call sequentially for recordings delivered by the queue.
result = identifier.identify("query_001.wav", recording_id="query_001")
```

Use one inference worker per identifier. The Python API raises `InvalidAudio`
for invalid recordings and propagates configuration/runtime errors; the CLI
converts the expected failures to JSON. Board quality statistics, buffer
ownership, queue management and a network service are outside this module.

## Verification

The task workspace contains `verify_identify.py` and verification reports.
It compares member02/member03 session02 scores against preserved comparison
results for both strategies, checks WAV boundaries, hash mismatch rejection,
threshold branches, JSON output and CLI exit codes, and verifies old model,
template and source hashes. Adding member04 can change the winning member
without changing the original two similarity scores. Member04 session01 is
used only as an enrolment smoke check, not an independent accuracy evaluation.
The integration tests do not use session03.

CAM++ and the original TDNN baseline each reproduced 40 saved independent
session02 query results across the two strategies. The maximum old-member
score difference was 1.79e-7. Seven invalid-audio cases per runtime, template
hash mismatch, threshold branches and CLI JSON/exit codes were checked.
These checks establish interface consistency on Windows, not accuracy on all
three members or performance on KV260. Threshold branch tests do not calibrate
an authentication threshold.

## Delivery files

Share `identify.py`, `voiceid.py` for CAM++, `small_model.py` for TDNN,
the selected fixed model, its matching three-member database, this document,
dependency installation information, and an authorized test WAV/reference
result. Enrolled members currently include member02, member03 and member04.
Do not mix templates from different model hashes or window strategies.
The existing dependency lock files describe the Windows environments; the
TDNN lock includes a CUDA build and is not a KV260 installation recipe.

The caller owns recording, resampling, file completion, request scheduling
and any hardware-quality statistics. This module owns WAV validation,
windowing, embedding extraction, template matching and result formatting.
