# Batch version of decision_identify_integration.py
#
# Instead of taking one WAV path from terminal, this script reads test cases 
# from verification/integration_manifest.csv and runs
# the same software decision + real CAM++ integration path for each case
#
# Current manifest: 30 Session2 WAV
#
# Keep these two sets of labels distinct: System Outcomes vs Test Harness results
# 1. System outcome (voice biometric result for each recording): memberXX, Unknown, Invalid, Error
# 2. Test harness labels (indicate each test case'result):PASS, INVALID, FAIL, ERROR

# Integration test: verify WAV -> Python reference D/S/C/N -> 
# -> qualityGate -> decisionFlow -> real CAM++ pipeline
# rather than the fake input used in unit tests

# run: python verification\decision_identify_batch.py

from pathlib import Path
import csv
import json
import subprocess
import sys
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1] # .parents[1]: repo root
MANIFEST = ROOT / "verification" / "integration_manifest.csv"

# decisionFlow.py is inside sw/
sys.path.insert(0, str(ROOT / "sw"))

# decide() function
from decisionFlow import decide

def wav_stats(wav_path):
    # Read actual PCM16 samples from WAV file
    # wav_stats 1. Validate WAV format
    # 2. Iterate over all PCM samples; compute D, S, N, and C
    info = sf.info(wav_path)

    if (
        # WAV, signed PCM16, 16 kHz, mono
        info.format != "WAV"
        or info.subtype != "PCM_16"
        or info.samplerate != 16000
        or info.channels != 1
    ):
        raise ValueError(
            f"{wav_path}: expected WAV PCM16 mono 16000 Hz"
        )
    
    # Read PCM samples from WAV file
    audio, _ = sf.read(wav_path, dtype="int16")

    # Software reference implementation of hardware statistics
    # Here D/S/C/N computed by Python from existing WAV, not supplied by FPGA
    D = 0
    S = 0
    C = 0
    for value in audio:
        x = int(value) # Convert NumPy int16 to Python int so arithmetic is not limited to 16 bits
        D += x
        S += x * x
        # PCM16 valid range: -32768 to 32767; clipping boundaries at either extreme
        # C counts samples reaching either boundary
        if x >= 32767 or x <= -32768:
            C += 1
    N = len(audio)
    return D, S, C, N

# Wrapper used by decisionFlow to invoke CAM++ identifier
def identify_campplus(wav_path, recording_id=None):
    command = [
        sys.executable,
        str(ROOT / "identify.py"),
        "--wav",
        str(wav_path),
        "--model",
        "campplus",
    ]

    if recording_id is not None:
        command += ["--recording-id", recording_id]
    # Execute identify.py
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip()
            or f"identify.py failed with return code {completed.returncode}"
        )
    # If identify.py produces no valid stdout output
    # identify.py returncode = 0, but produces no JSON output
    if not completed.stdout.strip():
        raise RuntimeError("identify.py produced no output")
    # Subprocess captures stdout as string, json.loads() parses into Python data structure
    # then returns it to decisionFlow
    return json.loads(completed.stdout)

def main():
    # 4 counters
    total = 0 # number of recordings tested
    passed = 0 # actual identity matches expected identity
    failed = 0 # identity mismatch, system Error outcome, or test exception
    invalid = 0 # recording rejected by qualityGate

    # Read test cases from CSV manifest
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader: # Run one test per recording in the manifest
            relative_path = row["file"]
            expected = row["expected"]

            wav_path = ROOT / relative_path
            total += 1 # Increment total at the start of each test case
            # Unlike single-file tests, batch regression continue despite individual recording failures
            # wrap each test in try/except to prevent 1 bad recording from aborting the entire run
            try:
                D, S, C, N = wav_stats(wav_path)

                result = decide(
                    recordingId=wav_path.stem, # wav_path.stem: filename without its extension
                    stats=(D, S, C, N),
                    # N previously computed directly from WAV file
                    dmaCount=N, # Set DMA count = actual sample count. DMA mismatch is not tested
                    wavPath=wav_path,
                    identifyFn=identify_campplus, # decisionFlow run CAM++ inference if qualityGate Valid
                )

                if result.outcome == expected:
                    passed += 1
                    print(
                        f"PASS  {relative_path}  "
                        f"expected={expected}  "
                        f"got={result.outcome}"
                    )

                # Recording rejected by quality gate
                # Invalid does not necessarily mean FAIL
                # rejecting poor-quality WAV is expected behavior
                # Hence: Track `Invalid` separately for manual review
                elif result.outcome == "Invalid":
                    invalid += 1

                    print(
                        f"INVALID  {relative_path}  "
                        f"expected={expected}  "
                        f"reason={result.reason}"
                    )

                # Outcome matches neither the expected member nor Invalid:
                # Mismatched / Unknown or Sysytem error (got = Error)
                # Note: if the system returns outcome "Error", the harness falls throughto FAIL
                # because "Error" is still a valid system outcome value
                # rather than a Python exception or missing result
                else:
                    failed += 1
                    print(
                        f"FAIL  {relative_path}  "
                        f"expected={expected}  "
                        f"got={result.outcome}  "
                        f"reason={result.reason}"
                    )
            # Python raised an exception during test execution
            # No valid result.outcome obtained (e.g. missing WAV, invalid WAV format, 
            # empty identify.py output, JSON parsing failure, subprocess RuntimeError)
            except Exception as e:
                failed += 1
                print(
                    f"ERROR  {relative_path}  "
                    f"{type(e).__name__}: {e}"
                )

    print()
    print("Integration regression summary")
    print(f"Total   = {total}")
    print(f"Passed  = {passed}")
    print(f"Invalid = {invalid}")
    print(f"Failed  = {failed}")

if __name__ == "__main__":
    main()
