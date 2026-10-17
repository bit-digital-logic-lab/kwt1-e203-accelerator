"""Build a small, reproducible KWT-1 MFCC sample set from Speech Commands V2."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import wave

import numpy as np

from run import ROOT, SOURCE


ARCHIVE = ROOT.parent / "资料/数据集/speech_commands_v0.02.tar.gz"
MAX_NUM_WAVS_PER_CLASS = 2**27 - 1


def split_for(name):
    stem = re.sub(r"_nohash_.*$", "", Path(name).name)
    h = int(hashlib.sha1(stem.encode()).hexdigest(), 16)
    percent = (h % (MAX_NUM_WAVS_PER_CLASS + 1)) * (100.0 / MAX_NUM_WAVS_PER_CLASS)
    return "validation" if percent < 10 else "testing" if percent < 20 else "training"


def rank_for(name):
    return hashlib.sha256(name.encode()).hexdigest()


def wav_float(data):
    with wave.open(io.BytesIO(data)) as handle:
        if (handle.getnchannels(), handle.getframerate(), handle.getsampwidth()) != (1, 16000, 2):
            raise ValueError("Expected mono 16 kHz PCM16 WAV")
        audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    output = np.zeros(16000, dtype=np.float32)
    n = min(len(audio), len(output))
    output[:n] = audio[:n].astype(np.float32) / 32768.0
    return output


def mfcc_matrices():
    # Reproduce official Windowing, MagnitudeRDFTmel, mel_table and DCT formulas.
    frame = 480
    fft = 512
    idx = np.arange(frame)
    hann = (0.5 - 0.5 * np.cos(2 * np.pi * idx / frame)).astype(np.float32)
    freq = np.arange(fft // 2 + 1)
    phase = 2 * np.pi * np.outer(idx, freq) / fft
    real = np.cos(phase).astype(np.float32)
    imag = -np.sin(phase).astype(np.float32)

    hertz = np.linspace(0.0, 8000.0, fft // 2 + 1)
    to_mel = lambda hz: 1127.0 * np.log(1.0 + hz / 700.0)
    spectrum_mel = to_mel(hertz)
    edges = np.linspace(to_mel(20.0), to_mel(7600.0), 82)
    mel = np.empty((fft // 2 + 1, 80), dtype=np.float32)
    for i in range(80):
        lower = (spectrum_mel - edges[i]) / (edges[i + 1] - edges[i])
        upper = (edges[i + 2] - spectrum_mel) / (edges[i + 2] - edges[i + 1])
        mel[:, i] = np.maximum(0.0, np.minimum(lower, upper))
    mel[0] = 0
    # Official mel_non_zero_only truncates after the last active FFT bin.
    last = np.flatnonzero(mel[:, -1])[-1] + 1
    dct = 2 * np.cos(np.pi * np.outer(np.arange(80) * 2 + 1, np.arange(40)) / 160)
    dct = (dct / np.sqrt(160)).astype(np.float32)
    return hann, real[:, :last], imag[:, :last], mel[:last], dct


def mfcc_batch(audio, matrices):
    hann, real, imag, mel, dct = matrices
    # 16000 samples, 480 window, 160 stride -> 98 frames.
    frames = np.lib.stride_tricks.sliding_window_view(audio, 480, axis=1)[:, ::160, :]
    frames = frames * hann
    magnitude = np.sqrt((frames @ real)**2 + (frames @ imag)**2)
    features = np.log(np.maximum(magnitude @ mel, np.float32(1e-12))) @ dct
    if features.shape[1:] != (98, 40):
        raise AssertionError(features.shape)
    return features.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-word", type=int, default=2)
    parser.add_argument("--include-special-labels", action="store_true",
                        help="Add unknown speech and one-second zero-waveform silence")
    parser.add_argument("--output", type=Path, default=ROOT / ".build/quant_samples.npz")
    args = parser.parse_args()
    if args.per_word < 1:
        parser.error("--per-word must be positive")
    flags = json.loads((SOURCE / "flags.json").read_text())
    wanted = set(flags["wanted_words"].split(","))
    labels = wanted | ({"_unknown_"} if args.include_special_labels else set())
    selected = {(partition, word): [] for partition in ("training", "validation") for word in labels}
    with tarfile.open(ARCHIVE, "r|gz") as archive:
        for member in archive:
            parts = Path(member.name).parts
            if not member.isfile() or len(parts) != 2 or not parts[1].endswith(".wav"):
                continue
            label = parts[0] if parts[0] in wanted else "_unknown_"
            if label not in labels or parts[0] == "_background_noise_":
                continue
            partition = split_for(member.name)
            if partition not in ("training", "validation"):
                continue
            key = (partition, label)
            rank = rank_for(member.name)
            existing = selected[key]
            if len(existing) == args.per_word and rank >= existing[-1][0]:
                continue
            content = archive.extractfile(member).read()
            existing.append((rank, member.name, content))
            existing.sort(key=lambda item: item[0])
            del existing[args.per_word:]

    entries = []
    for partition in ("training", "validation"):
        for word in sorted(labels):
            items = selected[partition, word]
            if len(items) != args.per_word:
                raise RuntimeError(f"Insufficient samples for {partition}/{word}")
            entries.extend((partition, word, name, wav_float(content)) for _, name, content in items)
        if args.include_special_labels:
            for i in range(args.per_word):
                entries.append((partition, "_silence_", f"_silence_/{partition}_zero_{i:03}.wav",
                                np.zeros(16000, dtype=np.float32)))
    audio = np.stack([item[3] for item in entries])
    features = mfcc_batch(audio, mfcc_matrices())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=features,
                        partitions=np.array([item[0] for item in entries]),
                        words=np.array([item[1] for item in entries]),
                        paths=np.array([item[2] for item in entries]))
    print(f"Saved {len(entries)} audio samples to {args.output}: "
          f"{sum(item[0] == 'training' for item in entries)} training, "
          f"{sum(item[0] == 'validation' for item in entries)} validation")


if __name__ == "__main__":
    main()
