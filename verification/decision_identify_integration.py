# Integration test for the software decision + identification path
#
# This single-file test currently uses Han's Session 2 recording
# and expects the real CAM++ identifier return member05
#
# This uses a real WAV instead of only fake unit-test inputs
# It connects decisionFlow.py to the real CAM++ identify.py 
# And checks that valid audio reaches identification, while invalid quality stops before it
#
# D/S/C/N are still calculated in Py, hardware integration is not covered by this test

from pathlib import Path
import json # parse identify.py's JSON output into Python object
import subprocess # launch identify.py as a separate process
import sys
import soundfile as sf # read WAV metadata and PCM samples

# Run:
# python verification\decision_identify_integration.py data\han\session2\wav\2-1.wav

ROOT = Path(__file__).resolve().parents[1] # .parents[1]: repo root

# decisionFlow.py is inside sw/
sys.path.insert(0, str(ROOT / "sw"))

# decide() function
from decisionFlow import decide

def wav_stats(wav_path):
    # Read the actual PCM16 samples from the WAV file
    # wav_stats 1. Validate WAV format
    # 2. Iterate over all PCM samples; compute D, S, N, and C
    info = sf.info(wav_path) # check WAV metadata

    if (
        # WAV, signed PCM16, 16 kHz, mono
        info.format != "WAV"
        or info.subtype != "PCM_16"
        or info.samplerate != 16000
        or info.channels != 1
    ):
        raise ValueError("Expected WAV, PCM16, mono, 16000 Hz")
    
    # Read PCM samples from WAV file
    audio, _ = sf.read(wav_path, dtype="int16")

    # Convert NumPy int16 to Python int, avoid fixed width int16 overflow
    samples = []

    for value in audio:
        # Convert NumPy int16 to Python int
        # memory usage growing as needed rather than fixed 16-bit limits
        samples.append(int(value))

    # Software reference implementation of hardware statistics
    # Here D/S/C/N computed by Python from existing WAV, not supplied by FPGA
    D = 0
    S = 0
    C = 0
    for x in samples:
        D += x
        S += x * x
        if x >= 32767 or x <= -32768:
            C += 1
    N = len(samples)
    return D, S, C, N # Tuple

# counter: tracks number of actual CAM++ identifier invocations
# Invalid test: use this counter to verify CAM++ inference is never invoked when input is rejected
identify_calls = 0

# Increment counter: decisionFlow invokes this function
def identify_campplus(wav_path, recording_id=None):
    global identify_calls
    identify_calls += 1
    # Equivalent: python identify.py --wav data\han\session2\wav\2-1.wav --model campplus
    command = [
        sys.executable,
        str(ROOT / "identify.py"),
        "--wav",
        str(wav_path),
        "--model",
        "campplus",
    ]

    if recording_id is not None:
        command += [
            "--recording-id",
            recording_id,
        ]
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

if len(sys.argv) != 2:
    print(
        "Run: python verification\\decision_identify_integration.py "
        "data\\han\\session2\\wav\\2-1.wav"
    )
    sys.exit(1)

wav_path = Path(sys.argv[1])

if not wav_path.is_absolute():
    wav_path = ROOT / wav_path

# Case 1: 
# Real WAV -> wav_stats() computes D/S/C/N -> decide() ->
# qualityGate validates input -> decide() calls identify_campplus() ->
# subprocess runs identify.py -> CAM++ identifies member05 -> assert
stats = wav_stats(wav_path)
D, S, C, N = stats
print("Real WAV statistics:")
print(f"D = {D}")
print(f"S = {S}")
print(f"C = {C}")
print(f"N = {N}")
# Record CAM++ invocation count before entering decide()
before_calls = identify_calls
# Call decisionFlow.decide()
result = decide(
    recordingId="real_valid_01",
    stats=stats,
    dmaCount=N,
    wavPath=wav_path,
    identifyFn=identify_campplus,
)

print()
print("Real integration result:")
print(f"Gate = {result.gateResult.status}")
print(f"Outcome = {result.outcome}")
print(f"Reason = {result.reason}")

assert result.gateResult.status == "Valid"
assert identify_calls == before_calls + 1
assert result.identifyResult is not None
# assert result.outcome not in ("Error", "Invalid")
assert result.outcome == "member05", (
    f"Expected member05, got {result.outcome}"
)

print("PASS: Valid audio reached the real CAM++ identifier")

# Case 2: Invalid quality must stop before CAM++ identification
# Construct invalid D/S/C/N -> decide() -> qualityGate: Invalid ->
# identify_campplus() must not be called -> assert no additional CAM++ invocation
before_calls = identify_calls
invalid_stats = (
    0, # D
    0, # S
    0, # C
    48000, # N
)

result = decide(
    recordingId="invalid_quality_01",
    stats=invalid_stats,
    dmaCount=48000,
    # Check whether decisionFlow incorrectly opens the valid WAV and runs CAM++
    # despite stats indicating invalid quality
    wavPath=wav_path,
    identifyFn=identify_campplus,
)

assert result.outcome == "Invalid"
assert result.identifyResult is None
assert identify_calls == before_calls

print("PASS: Invalid quality was rejected without calling CAM++")

print()
print("Software integration verification passed")
