# DMA -> D/S/C/N -> qualityGate -> result

from dataclasses import dataclass

from qualityGate import QualityResult, qualityGate


@dataclass
class Decision:
    recordingId: str
    outcome: str            # "Error", "Invalid", "Unknown" or a member id
    reason: str
    gateResult: QualityResult = None    # result of the quality gate
    identifyResult: dict = None         # result dict of identify(), None if it was not called

 # stats: (D, S, C, N) read from the PL
 # identifyFn: function that identifies the speaker
def decide(recordingId, stats, dmaCount, wavPath,
           identifyFn, invalidExc=ValueError, captureSuccess=True):

    # run the quality gate
    D, S, C, N = stats
    q = qualityGate(D, S, C, N, dmaCount, captureSuccess)

    # If the gate says Invalid or Error, then stop
    if q.status != "Valid":
        return Decision(recordingId, q.status, q.reason, q)

    # if the recording is good, call the identify function
    # recording_id is the keyword name of identify.py's own API
    try:
        result = identifyFn(wavPath, recording_id=recordingId)
        status = result["status"]
    except invalidExc as e:
        # identify says the audio is not acceptable
        return Decision(recordingId, "Invalid", str(e), q)
    except Exception as e:
        # anything else went wrong
        return Decision(recordingId, "Error", repr(e), q)

    # read the result dict
    if status == "invalid":
        return Decision(recordingId, "Invalid", result["reason"], q, result)

    if status != "ok":
        return Decision(recordingId, "Error", result["reason"], q, result)

    return Decision(recordingId, result["identity"], "ok", q, result)
