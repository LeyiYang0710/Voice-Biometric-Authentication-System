"""Verify models with public samples; this is not a formal accuracy test."""

# 1. Import tools and reuse embedding extraction from voiceid.py.
from pathlib import Path
import json
import hashlib
import sherpa_onnx

from voiceid import extract

# 2. Locate the project root.
ROOT = Path(__file__).resolve().parent


# 3. Check all ONNX candidates using official public samples only.
def main():
    # Read the official model checksum list.
    expected = {
        line.split()[0]: line.split()[1]
        for line in (ROOT / "models/onnx/checksum.txt").read_text().splitlines()
        if len(line.split()) == 2
    }
    results = []
    # Load each model, extract three embeddings, and compare scores.
    for model in sorted((ROOT / "models/onnx").glob("*.onnx")):
        sha = hashlib.file_digest(model.open("rb"), "sha256").hexdigest()
        if model.name in expected and expected[model.name] != sha:
            raise ValueError("Checksum mismatch: " + model.name)
        conf = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(model), num_threads=2, debug=False, provider="cpu"
        )
        extractor = sherpa_onnx.SpeakerEmbeddingExtractor(conf)
        vectors = []
        times = []
        for name in ["fangjun-sr-1.wav", "fangjun-test-sr-1.wav", "leijun-sr-1.wav"]:
            vector, seconds = extract(extractor, ROOT / "samples" / name)
            vectors.append(vector)
            times.append(seconds)
        result = dict(
            model=model.name,
            sha256=sha,
            upstream_checksum_verified=model.name in expected,
            dimension=len(vectors[0]),
            same_speaker_cosine=float(vectors[0] @ vectors[1]),
            different_speaker_cosine=float(vectors[0] @ vectors[2]),
            inference_seconds=times,
            provider="cpu",
            threads=2,
            note="Public sample smoke test only; threshold uncalibrated; not an accuracy or speed benchmark",
        )
        results.append(result)
        print(json.dumps(result), flush=True)
    # Save the run record for reproducibility.
    (ROOT / "logs/smoke-test.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
