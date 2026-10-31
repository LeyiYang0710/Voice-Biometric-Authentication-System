"""Identify speakers with fixed CAM++ or TDNN models and matching templates.

Project integration code. The model implementations remain in their original
libraries and small_model.py. No training or template updates happen here.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 48000
HOP_SAMPLES = 24000


class InvalidAudio(ValueError):
    """The supplied recording does not satisfy the inference input contract."""


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalize(values):
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norm < 1e-8):
        raise ValueError("Invalid or degenerate embedding")
    return values / norm


def load_windows(path, strategy):
    # Input validation belongs at the WAV boundary, before model inference.
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    try:
        info = sf.info(path)
        if (info.format, info.subtype, info.samplerate, info.channels) != (
            "WAV", "PCM_16", SAMPLE_RATE, 1
        ):
            raise InvalidAudio("Expected WAV, PCM16, mono, 16000 Hz")
        if info.frames < WINDOW_SAMPLES:
            raise InvalidAudio("Recording must be at least 3 seconds")
        audio, _ = sf.read(path, dtype="float32")
    except sf.LibsndfileError as error:
        raise InvalidAudio(f"Cannot decode WAV: {error}") from error
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) == 0:
        raise InvalidAudio("Recording is nonfinite or completely silent")

    if strategy == "center3s":
        starts = [(len(audio) - WINDOW_SAMPLES) // 2]
    else:
        starts = list(range(0, len(audio) - WINDOW_SAMPLES + 1, HOP_SAMPLES))
        if starts[-1] != len(audio) - WINDOW_SAMPLES:
            starts.append(len(audio) - WINDOW_SAMPLES)
    return np.stack([audio[s:s + WINDOW_SAMPLES] for s in starts]), len(audio)


class SpeakerIdentifier:
    """Load once at startup, then identify recordings from one inference worker."""

    def __init__(self, model_type, database_path, model_path=None, threshold=None):
        start = time.perf_counter()
        if model_type not in {"campplus", "tdnn"}:
            raise ValueError("Model must be campplus or tdnn")
        if threshold is not None and (
            not np.isfinite(threshold) or not -1 <= threshold <= 1
        ):
            raise ValueError("Threshold must be finite and between -1 and 1")
        self.model_type = model_type
        self.threshold = threshold
        self.database_path = Path(database_path)
        database = json.loads(self.database_path.read_text(encoding="utf-8"))
        required = {"model_sha256", "model_path", "strategy", "members"}
        if not isinstance(database, dict) or not required.issubset(database):
            raise ValueError("Use a database with model_sha256, model_path, strategy and members")
        self.strategy = database["strategy"]
        if self.strategy not in {"center3s", "multiwindow"}:
            raise ValueError("Database strategy must be center3s or multiwindow")
        self.model_path = Path(model_path or database["model_path"])
        if not self.model_path.is_absolute():
            self.model_path = ROOT / self.model_path
        self.model_hash = checksum(self.model_path)
        if self.model_hash != database["model_sha256"]:
            raise ValueError("Model/template SHA-256 mismatch")
        self.database_hash = checksum(self.database_path)
        members = database["members"]
        if not isinstance(members, dict) or not members:
            raise ValueError("Database must contain enrolled members")
        self.member_names = list(members)
        self.templates = np.asarray(list(members.values()), dtype=np.float32)
        dimension = 192 if model_type == "campplus" else 64
        if (
            self.templates.shape != (len(members), dimension)
            or not np.isfinite(self.templates).all()
            or not np.allclose(np.linalg.norm(self.templates, axis=1), 1, atol=1e-5)
        ):
            raise ValueError(f"Expected finite unit templates with dimension {dimension}")

        if model_type == "campplus":
            # Keep torch optional for the CAM++ runtime.
            from voiceid import create_extractor

            self.extractor = create_extractor(self.model_path)
        else:
            import torch
            from small_model import SmallTDNN

            torch.set_num_threads(2)
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, dict) or not {"labels", "config", "model"}.issubset(checkpoint):
                raise ValueError("Expected a project TDNN checkpoint with labels, config and model")
            self.model = SmallTDNN(
                len(checkpoint["labels"]),
                cosine_classifier=checkpoint["config"].get("cosine_classifier", False),
            )
            self.model.load_state_dict(checkpoint["model"], strict=True)
            self.model.eval()
        self.startup_seconds = time.perf_counter() - start

    def extract_windows(self, windows):
        if self.model_type == "campplus":
            vectors = []
            for window in windows:
                stream = self.extractor.create_stream()
                stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=window)
                stream.input_finished()
                if not self.extractor.is_ready(stream):
                    raise InvalidAudio("CAM++ cannot process this audio window")
                vectors.append(np.asarray(self.extractor.compute(stream), dtype=np.float32))
            return np.stack(vectors)

        import torch

        with torch.inference_mode():
            return self.model.embed(torch.from_numpy(windows)).numpy()

    def identify(self, wav_path, recording_id=None):
        start = time.perf_counter()
        wav_path = Path(wav_path)
        windows, samples = load_windows(wav_path, self.strategy)
        prepared = time.perf_counter()
        # Normalize windows, average within a recording, then normalize again.
        embedding = normalize(normalize(self.extract_windows(windows)).mean(axis=0))
        extracted = time.perf_counter()
        scores = {
            member: float(template @ embedding)
            for member, template in zip(self.member_names, self.templates)
        }
        best_member = max(scores, key=scores.get)
        identity = best_member
        if self.threshold is not None and scores[best_member] < self.threshold:
            identity = "Unknown"
        end = time.perf_counter()
        return {
            "recording_id": recording_id or wav_path.stem,
            "status": "ok",
            "identity": identity,
            "best_member": best_member,
            "scores": scores,
            "reason": None,
            "model_id": self.model_type,
            "model_sha256": self.model_hash,
            "template_db_id": self.database_path.name,
            "template_db_sha256": self.database_hash,
            "strategy": self.strategy,
            "decision_mode": "ranking_only" if self.threshold is None else "thresholded",
            "threshold": self.threshold,
            "audio_seconds": samples / SAMPLE_RATE,
            "window_count": len(windows),
            "processing_seconds": end - start,
            "timings": {
                "startup_seconds": self.startup_seconds,
                "audio_preparation_seconds": prepared - start,
                "embedding_seconds": extracted - prepared,
                "matching_seconds": end - extracted,
            },
        }


def _identify_default(model_type, wav_path, recording_id=None):
    """Use the checked-in model, templates and development threshold together."""
    if "session03" in {part.lower() for part in wav_path.resolve().parts}:
        raise ValueError("Team session03 remains reserved for final evaluation")
    if model_type == "tdnn":
        from ml.tdnn.identify_frozen import FrozenTDNN, InvalidAudio as TDNNInvalidAudio

        release_path = ROOT / "ml/tdnn/frozen_libri251_fp32.json"
        release = json.loads(release_path.read_text(encoding="utf-8"))
        try:
            return FrozenTDNN(release["conditions_id"]).identify(wav_path, recording_id)
        except TDNNInvalidAudio as error:
            # The imported backend can load identify under a separate module name.
            raise InvalidAudio(str(error)) from error

    config_path = ROOT / "ml/campplus/frozen_threshold.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("schema_version") != 1
        or config.get("status") != "frozen_development_threshold"
        or config.get("model_type") != "campplus"
        or config.get("accept_rule") != "best_score >= threshold"
    ):
        raise ValueError("Expected the frozen CAM++ development configuration")
    threshold = config["threshold"]
    if type(threshold) not in (int, float) or not np.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("Frozen threshold must be finite and between -1 and 1")
    database_path = config_path.parent / config["template_path"]
    binding = config["binding"]
    if checksum(database_path) != binding["template_db_sha256"]:
        raise ValueError("Frozen CAM++ template database changed")
    database = json.loads(database_path.read_text(encoding="utf-8"))
    if (
        database["model_sha256"] != binding["model_sha256"]
        or database["strategy"] != binding["strategy"]
        or list(database["members"]) != binding["members"]
    ):
        raise ValueError("Model, templates or window strategy differ from the CAM++ calibration")
    identifier = SpeakerIdentifier("campplus", database_path, threshold=threshold)
    result = identifier.identify(wav_path, recording_id)
    result.update(
        conditions_id=binding["conditions_id"], scope=config["scope"],
        deployment_approved=config["deployment_approved"], calibration_sha256=checksum(config_path),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--model", choices=["campplus", "tdnn"], required=True)
    parser.add_argument("--db", type=Path,
                        help="Optional explicit database; omit to use the model's fixed profile")
    parser.add_argument("--model-path", "--checkpoint", dest="model_path", type=Path,
                        help="Override the database model path; its hash must still match")
    parser.add_argument("--threshold", type=float,
                        help="Explicit --db only; omit for ranking-only use of that database")
    parser.add_argument("--recording-id")
    args = parser.parse_args()
    if args.db is None and (args.model_path is not None or args.threshold is not None):
        parser.error("--model-path/--threshold overrides require an explicit --db")
    try:
        if args.db is None:
            result = _identify_default(args.model, args.wav, args.recording_id)
        else:
            identifier = SpeakerIdentifier(args.model, args.db, args.model_path, args.threshold)
            result = identifier.identify(args.wav, args.recording_id)
        code = 0
    except (InvalidAudio, OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        result = {
            "recording_id": args.recording_id or args.wav.stem,
            "status": "invalid" if isinstance(error, InvalidAudio) else "error",
            "identity": None,
            "scores": {},
            "reason": str(error),
            "model_id": args.model,
        }
        code = 2 if result["status"] == "invalid" else 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
