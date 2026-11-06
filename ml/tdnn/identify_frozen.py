"""Use the frozen LibriSpeech TDNN and its development Unknown threshold."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from identify import InvalidAudio, SpeakerIdentifier


ROOT = Path(__file__).resolve().parent
RELEASE_SHA256 = "d5e2b2b12faf2a87e930df44e7b5a3fdfb9e6d1d27e70ccc23adf70a326ecd83"


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def release_sha256(release):
    # Ignore JSON formatting changes made by an editor or by Git line endings.
    content = json.dumps(release, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(content).hexdigest()


def load_release(conditions_id):
    release = json.loads((ROOT / "frozen_libri251_fp32.json").read_text(encoding="utf-8"))
    if release_sha256(release) != RELEASE_SHA256:
        raise ValueError("Frozen release configuration changed")
    if conditions_id != release["conditions_id"]:
        raise ValueError("Conditions differ from the frozen development calibration")
    for name, expected in release["source_code_sha256_lf"].items():
        content = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Frozen inference dependency changed: {name}")
    return release


class FrozenTDNN:
    """Load once; identify complete recordings sequentially with a fixed profile."""

    def __init__(self, conditions_id):
        self.release = load_release(conditions_id)
        database_path = ROOT / self.release["template_path"]
        if file_sha256(database_path) != self.release["template_db_sha256"]:
            raise ValueError("Frozen template database changed")
        database = json.loads(database_path.read_text(encoding="utf-8"))
        if (
            database["model_sha256"] != self.release["model_sha256"]
            or database["strategy"] != "multiwindow"
            or list(database["members"]) != self.release["members"]
        ):
            raise ValueError("Model, window strategy or member list differs from the release")
        self.identifier = SpeakerIdentifier(
            "tdnn", database_path,
            model_path=ROOT / self.release["checkpoint_path"],
            threshold=self.release["threshold"],
        )

    def identify(self, wav_path, recording_id=None):
        wav_path = Path(wav_path)
        if "session03" in {part.lower() for part in wav_path.resolve().parts}:
            raise ValueError("Team session03 remains reserved for final evaluation")
        result = self.identifier.identify(wav_path, recording_id)
        result.update(
            release_id=self.release["release_id"],
            inference_branch="fp32",
            conditions_id=self.release["conditions_id"],
            scope=self.release["scope"],
            deployment_approved=False,
        )
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wav", type=Path)
    source.add_argument("--check-release", action="store_true")
    parser.add_argument("--conditions-id", required=True)
    parser.add_argument("--recording-id")
    args = parser.parse_args()
    try:
        if args.check_release:
            release = load_release(args.conditions_id)
            result = dict(
                status="ok", release_id=release["release_id"],
                check="release_metadata_and_existing_source_only",
                runtime_assets_checked=False,
            )
        else:
            result = FrozenTDNN(args.conditions_id).identify(args.wav, args.recording_id)
        code = 0
    except (InvalidAudio, OSError, ValueError, KeyError, RuntimeError, ImportError) as error:
        result = dict(
            status="invalid" if isinstance(error, InvalidAudio) else "error",
            identity=None, scores={}, reason=str(error), model_id="tdnn",
            recording_id=args.recording_id or (args.wav.stem if args.wav else None),
        )
        code = 2 if result["status"] == "invalid" else 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
