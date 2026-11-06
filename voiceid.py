"""CAM++ baseline: inspect recordings, enrol members, and query identity.

This script uses the original model without training CAM++. Start at main().
"""

# 1. Import tools and specify the default model and template database.
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import sherpa_onnx
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent
DEFAULT = (
    ROOT / "models/onnx/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
)
DEFAULT_DATABASE = ROOT / "data/templates.json"


# 2. Read command-line arguments to select the requested operation.
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="CAM++ speaker enrolment and identification"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT)
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser(
        "inspect", help="Inspect audio format and volume"
    )
    inspect_parser.add_argument("wav", type=Path)

    convert_parser = commands.add_parser("convert", help="Save a mono 16kHz WAV copy")
    convert_parser.add_argument("wav", type=Path)
    convert_parser.add_argument("output", type=Path)

    enrol_parser = commands.add_parser("enrol", help="Enrol a member using recordings")
    enrol_parser.add_argument("member")
    enrol_parser.add_argument("wavs", type=Path, nargs="+")
    enrol_parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE)

    query_parser = commands.add_parser("query", help="Identify a new recording")
    query_parser.add_argument("wav", type=Path)
    query_parser.add_argument("--threshold", type=float, required=True)
    query_parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE)
    return parser.parse_args()


# 3. Inspect audio; save conversions separately to preserve the original.
def inspect_or_convert(args):
    audio, sample_rate = sf.read(args.wav, dtype="float32", always_2d=True)
    if len(audio) == 0 or not np.isfinite(audio).all():
        raise ValueError("Empty or nonfinite audio")

    information = {
        "file": str(args.wav),
        "sample_rate": sample_rate,
        "channels": audio.shape[1],
        "samples": len(audio),
        "seconds": len(audio) / sample_rate,
        "peak": float(np.max(np.abs(audio))),
        "rms": float(np.sqrt(np.mean(audio * audio))),
        "clipped_fraction": float(np.mean(np.abs(audio) >= 0.999)),
    }

    if args.command == "convert":
        if args.output.resolve() == args.wav.resolve() or args.output.exists():
            raise ValueError("Output must be a new file")

        mono_audio = audio.mean(axis=1)
        divisor = math.gcd(sample_rate, 16000)
        converted_audio = resample_poly(
            mono_audio, 16000 // divisor, sample_rate // divisor
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(args.output, converted_audio, 16000, subtype="PCM_16")
        information["output"] = str(args.output)

    print(json.dumps(information, indent=2))


# 4. Create a model extractor using two CPU threads.
def create_extractor(model_path):
    configuration = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(model_path), num_threads=2, provider="cpu"
    )
    return sherpa_onnx.SpeakerEmbeddingExtractor(configuration)


# 5. Extract a normalized embedding and record inference time.
def extract(extractor, path):
    audio, sample_rate = sf.read(path, dtype="float32")
    if audio.ndim != 1 or sample_rate != 16000:
        raise ValueError(f"Expected mono 16kHz: {path}")
    if len(audio) == 0 or not np.isfinite(audio).all() or np.max(np.abs(audio)) == 0:
        raise ValueError("Invalid or silent audio")

    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=sample_rate, waveform=audio)
    stream.input_finished()
    if not extractor.is_ready(stream):
        raise ValueError("Not enough audio")

    start_time = time.perf_counter()
    embedding = np.asarray(extractor.compute(stream), dtype=np.float32)
    elapsed_seconds = time.perf_counter() - start_time
    if not np.isfinite(embedding).all() or np.linalg.norm(embedding) == 0:
        raise ValueError("Invalid embedding")

    normalized_embedding = embedding / np.linalg.norm(embedding)
    return normalized_embedding, elapsed_seconds


# 6. Load the template database and verify compatibility with the model.
def load_database(database_path, model_path):
    with model_path.open("rb") as model_file:
        model_hash = hashlib.file_digest(model_file, "sha256").hexdigest()

    if database_path.exists():
        database = json.loads(database_path.read_text(encoding="utf-8"))
    else:
        database = {"model_sha256": model_hash, "members": {}}

    if database["model_sha256"] != model_hash:
        raise ValueError("Template/model mismatch; use a separate database")
    return database


# 7. Enrolment: average embeddings, normalize, and save the member template.
def enrol_member(args, extractor, database):
    embeddings = []
    for recording_path in args.wavs:
        embedding, _ = extract(extractor, recording_path)
        embeddings.append(embedding)

    average_embedding = np.mean(embeddings, axis=0)
    template_norm = np.linalg.norm(average_embedding)
    if template_norm < 1e-8:
        raise ValueError("Degenerate template")

    template = average_embedding / template_norm
    database["members"][args.member] = template.tolist()
    args.db.parent.mkdir(parents=True, exist_ok=True)
    args.db.write_text(json.dumps(database, indent=2), encoding="utf-8")
    print(json.dumps({"enrolled": args.member, "recordings": len(args.wavs)}))


# 8. Query: accept the best match only when its score reaches the threshold.
def query_member(args, extractor, database):
    if not -1 <= args.threshold <= 1:
        raise ValueError("Cosine threshold must be in [-1,1]")
    if not database["members"]:
        raise ValueError("No enrolled members")

    embedding, elapsed_seconds = extract(extractor, args.wav)
    scores = {}
    for member_name, template in database["members"].items():
        scores[member_name] = float(embedding @ np.asarray(template))

    best_member = max(scores, key=scores.get)
    identity = best_member if scores[best_member] >= args.threshold else "Unknown"
    result = {
        "result": identity,
        "scores": scores,
        "threshold": args.threshold,
        "inference_seconds": elapsed_seconds,
        "note": "Threshold must be calibrated on independent validation data; no VAD/spoof protection yet",
    }
    print(json.dumps(result, indent=2))


# 9. Entry point: dispatch the command to the corresponding steps.
def main():
    args = parse_arguments()
    if args.command in ["inspect", "convert"]:
        inspect_or_convert(args)
        return

    database = load_database(args.db, args.model)
    extractor = create_extractor(args.model)
    if args.command == "enrol":
        enrol_member(args, extractor, database)
    else:
        query_member(args, extractor, database)


if __name__ == "__main__":
    main()
