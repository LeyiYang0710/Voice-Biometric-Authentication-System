import math

from qualityGate import qualityGate, joinTo64Bits, minSamples, maxClipFrac


# work out D, S, C, N from samples
def makeStats(samples):
    D = 0
    S = 0
    C = 0
    for x in samples:
        D = D + x
        S = S + x * x
        if x >= 32767 or x <= -32768:
            C = C + 1
    N = len(samples)
    return D, S, C, N


def makeTone(count=minSamples, amplitude=6000):
    samples = []
    for i in range(count):
        value = amplitude * math.sin(2 * math.pi * 300 * i / 16000)
        samples.append(int(value))
    return samples


def runGate(samples, dmaCount=0):
    D, S, C, N = makeStats(samples)
    if dmaCount == 0:
        dmaCount = N
    return qualityGate(D, S, C, N, dmaCount)


def testValid():
    result = runGate(makeTone())
    assert result.status == "Valid"
    assert result.clipFrac == 0


def testSilence():
    samples = [0] * minSamples
    result = runGate(samples)
    assert result.status == "Invalid"
    assert result.reason == "too quiet"


def testClipFractionBoundary():
    limit = int(minSamples * maxClipFrac)
    samples = makeTone()
    for i in range(limit):
        samples[i] = -32768
    assert runGate(samples).status == "Valid"

    # one more clipped sample is over the limit
    samples[limit] = -32768
    assert runGate(samples).status == "Invalid"


def testCountMismatch():
    result = runGate(makeTone(), dmaCount=minSamples + 7)
    assert result.status == "Error"
    assert result.reason == "N != dmaCount"


def testJoinUnsignedKeepsTopBit():
    assert joinTo64Bits(0, 0x80000000, False) == 1 << 63


def testTooShort():
    result = runGate(makeTone(20000))
    assert result.status == "Invalid"
    assert result.reason == "shorter than 3 s"


def testNegativeEnergy():
    result = qualityGate(-70000, 10, 0, minSamples, minSamples)
    assert result.status == "Error"


def testJoinNegative():
    assert joinTo64Bits(0xFFFFFFFB, 0xFFFFFFFF, True) == -5


def testClipping():
    # full scale square wave, every sample is on a clip limit
    samples = []
    for i in range(minSamples):
        if i % 2 == 0:
            samples.append(32767)
        else:
            samples.append(-32768)
    result = runGate(samples)
    assert result.status == "Invalid"
    assert result.reason == "clipping"


def testCaptureFault():
    D, S, C, N = makeStats(makeTone())
    result = qualityGate(D, S, C, N, N, captureSuccess=False)
    assert result.status == "Error"


def testJoinAcrossWords():
    assert joinTo64Bits(7, 2, True) == (2 << 32) + 7


def testZeroSamples():
    result = qualityGate(0, 0, 0, 0, 0)
    assert result.status == "Error"
