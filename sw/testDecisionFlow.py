from decisionFlow import decide
from qualityGate import minSamples

from testQualityGate import makeStats, makeTone


# fake identify functions, they do not load any model
def fakeMember(wavPath, recording_id=None):
    return {"status": "ok", "identity": "member02", "reason": None}


def fakeInvalid(wavPath, recording_id=None):
    return {"status": "invalid", "identity": None, "reason": "bad wav"}


def fakeError(wavPath, recording_id=None):
    return {"status": "error", "identity": None, "reason": "model failed"}


def fakeRaise(wavPath, recording_id=None):
    raise ValueError("less than 3 seconds")


def fakeCrash(wavPath, recording_id=None):
    raise RuntimeError("model crashed")


# this one remembers every call, so a test can check the model was not used
calls = []


def fakeSpy(wavPath, recording_id=None):
    calls.append(wavPath)
    return {"status": "ok", "identity": "member02", "reason": None}


def run(samples, identifyFn, dmaCount=0):
    stats = makeStats(samples)
    if dmaCount == 0:
        dmaCount = stats[3]
    return decide("rec1", stats, dmaCount, "rec1.wav", identifyFn)


def testCrash():
    d = run(makeTone(), fakeCrash)
    assert d.outcome == "Error"
    assert "model crashed" in d.reason


def testSilence():
    calls.clear()
    d = run([0] * minSamples, fakeSpy)
    assert d.outcome == "Invalid"
    assert d.identifyResult is None
    assert len(calls) == 0


def testValid():
    d = run(makeTone(), fakeMember)
    assert d.outcome == "member02"
    assert d.gateResult.status == "Valid"


def testInvalidDict():
    d = run(makeTone(), fakeInvalid)
    assert d.outcome == "Invalid"
    assert d.reason == "bad wav"


def testMismatch():
    calls.clear()
    d = run(makeTone(), fakeSpy, dmaCount=minSamples + 7)
    assert d.outcome == "Error"
    assert d.reason == "N != dmaCount"
    assert len(calls) == 0


def testRaise():
    d = run(makeTone(), fakeRaise)
    assert d.outcome == "Invalid"
    assert d.reason == "less than 3 seconds"


def testErrorDict():
    d = run(makeTone(), fakeError)
    assert d.outcome == "Error"
    assert d.reason == "model failed"
