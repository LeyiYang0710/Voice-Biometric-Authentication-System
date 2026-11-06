"""Convert a decoded recording to mono 16 kHz PCM16, preserving its source."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def convert(source, output, channel=None, mixdown=False, allow_clipping=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    report_path = output.with_suffix('.conversion.json')
    if output.exists() or report_path.exists() or source == output:
        raise ValueError('Output WAV and conversion report must be new files; source recordings are preserved')
    if output.suffix.lower() != '.wav':
        raise ValueError('Output must use the .wav extension')
    if channel is not None and mixdown:
        raise ValueError('Choose one channel or mixdown, not both')
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    info = sf.info(source)
    audio, rate = sf.read(source, dtype='float64', always_2d=True)
    if len(audio) != info.frames or not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Empty, truncated, or nonfinite input')
    if channel is not None:
        if not 0 <= channel < info.channels:
            raise ValueError('Channel index is outside the source channel count')
        audio = audio[:, channel]
    elif info.channels == 1:
        audio = audio[:, 0]
    elif mixdown:
        audio = audio.mean(axis=1)
    else:
        raise ValueError('Multichannel audio requires an explicit --channel or --mixdown')
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        audio = resample_poly(audio, 16000 // divisor, rate // divisor)
    pcm = np.rint(audio * 32768)
    clipping = int(np.count_nonzero((pcm < -32768) | (pcm > 32767)))
    if clipping and not allow_clipping:
        raise ValueError(f'{clipping} converted samples would clip; inspect the recording or explicitly allow clipping')
    if len(pcm) < 48000 or not np.any(pcm):
        raise ValueError('Converted input must contain nonzero audio lasting at least three seconds')
    output.parent.mkdir(parents=True, exist_ok=True)
    if (info.format, info.subtype, rate, info.channels) == ('WAV', 'PCM_16', 16000, 1):
        shutil.copyfile(source, output)
    else:
        sf.write(output, np.clip(pcm, -32768, 32767).astype(np.int16), 16000, subtype='PCM_16', format='WAV')
    if hashlib.sha256(source.read_bytes()).hexdigest() != before:
        raise ValueError('Source changed during conversion')
    result = dict(source=str(source), output=str(output), source_sha256=before,
        output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        source_rate=rate, source_channels=info.channels, selected_channel=channel,
        mixdown=mixdown, output_rate=16000, output_channels=1, subtype='PCM_16',
        output_samples=len(pcm), seconds=len(pcm)/16000, converted_clipping_samples=clipping,
        amplitude_normalization=False, source_unchanged=True)
    report_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    g = p.add_mutually_exclusive_group()
    g.add_argument('--channel', type=int)
    g.add_argument('--mixdown', action='store_true')
    p.add_argument('--allow-clipping', action='store_true')
    a = p.parse_args()
    print(json.dumps(convert(a.input, a.output, a.channel, a.mixdown, a.allow_clipping), indent=2))
