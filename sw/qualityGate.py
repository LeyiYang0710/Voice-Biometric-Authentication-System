from dataclasses import dataclass

@dataclass
class QualityResult:
    status: str
    reason: str
    mean: float = 0.0
    energy: float = 0.0
    clipFrac: float = 0.0

minSamples  = 48000     # Shortest recording: 16000(sample rate) * 3 = 3s -> consistence with identify.py
minEnergy   = 1e5
maxClipFrac = 0.01

def joinTo64Bits(lo, hi, isSigned):
    result = ((hi & 0xFFFFFFFF) << 32) | (lo & 0xFFFFFFFF)
    if isSigned and result >= (1 << 63):
        result -= (1 << 64)
    return result

def qualityGate(D, S, C, N, dmaCount, captureSuccess=True):
    # check the integrity and return Error if fails
    if not captureSuccess:
        return QualityResult("Error", "capture fault")
    if N <= 0:
        return QualityResult("Error", "N <= 0")
    # N is from stat register, dmaCount is from AXI DMA, they should be same
    if N != dmaCount:
        return QualityResult("Error", "N != dmaCount")

    # calculate the statistics
    if N * S - D * D < 0:   # Cauchy-schwartz inequality
        return QualityResult("Error", "negative energy")
    mean      = D / N
    energy    = S / N - mean * mean     # mean-removed energy(variance)
    clipFrac = C / N

    # check the quality and return Invalid if fails
    if N < minSamples:
        return QualityResult("Invalid", "shorter than 3 s", mean, energy, clipFrac)
    if energy < minEnergy:
        return QualityResult("Invalid", "too quiet", mean, energy, clipFrac)
    if clipFrac > maxClipFrac:
        return QualityResult("Invalid", "clipping", mean, energy, clipFrac)

    # valid
    return QualityResult("Valid", "ok", mean, energy, clipFrac)