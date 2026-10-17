"""Exercise WAV -> MFCC -> first encoder input on archived real speech."""

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import wave

import numpy as np

from offline_encoder_input import ROOT, PREFIX
from prepare_quant_samples import ARCHIVE, mfcc_batch, mfcc_matrices, split_for, wav_float


TARGETS = {
    "yes/676f8138_nohash_0.wav",
    "no/7e7ca854_nohash_1.wav",
    "go/0ba018fc_nohash_4.wav",
    "yes/ec21c46b_nohash_1.wav",
    "no/45692b02_nohash_3.wav",
    "four/750e3e75_nohash_0.wav",  # unknown target word, validation split
}
OUT = ROOT / "validation/real_input_chain_20261017"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    weights_path = ROOT / "weights/kwt1_fp32.npz"
    with np.load(weights_path, allow_pickle=False) as archive:
        kernel = archive[PREFIX + "patch_proj/kernel"]
        bias = archive[PREFIX + "patch_proj/bias"]
        cls = archive[PREFIX + "class_emb"][0, 0]
        pos = archive[PREFIX + "pos_emb"][0]
    matrices = mfcc_matrices()
    found = {}
    with tarfile.open(ARCHIVE, "r|gz") as archive:
        for member in archive:
            key = member.name.removeprefix("./")
            if key in TARGETS and member.isfile():
                found[key] = archive.extractfile(member).read()
                if len(found) == len(TARGETS):
                    break
    if found.keys() != TARGETS:
        raise RuntimeError(f"Missing WAV members: {sorted(TARGETS - found.keys())}")

    records = []
    for index, path in enumerate(sorted(TARGETS)):
        raw = found[path]
        with wave.open(io.BytesIO(raw)) as handle:
            wav_info = {"channels": handle.getnchannels(), "sample_rate_hz": handle.getframerate(),
                        "sample_width_bytes": handle.getsampwidth(), "frames": handle.getnframes()}
        audio = wav_float(raw)
        mfcc = mfcc_batch(audio[None, :], matrices)[0]
        # Separate FFT implementation checks the magnitude path of MFCC generation.
        frames = np.lib.stride_tricks.sliding_window_view(audio, 480)[::160] * matrices[0]
        spectrum = np.abs(np.fft.rfft(frames, n=512, axis=-1))[:, :matrices[3].shape[0]]
        fft_mfcc = (np.log(np.maximum(spectrum @ matrices[3], 1e-12)) @ matrices[4]).astype(np.float32)
        mfcc_error = float(np.max(np.abs(mfcc - fft_mfcc)))
        if mfcc.shape != (98, 40) or not np.isfinite(mfcc).all() or mfcc_error > 1e-3:
            raise AssertionError(f"MFCC mismatch: {path}: {mfcc_error}")
        stem = f"sample_{index:02d}"
        mfcc_path = OUT / f"{stem}_mfcc.npy"
        encoder_path = OUT / f"{stem}_encoder.npy"
        np.save(mfcc_path, mfcc, allow_pickle=False)
        subprocess.run([sys.executable, str(ROOT / "offline_encoder_input.py"),
                        "--input", str(mfcc_path), "--output", str(encoder_path)], check=True,
                       capture_output=True, text=True)
        actual = np.load(encoder_path, allow_pickle=False)
        # Independent row-wise contraction and explicit token construction.
        reference = np.empty((99, 64), dtype=np.float32)
        reference[0] = cls + pos[0]
        reference[1:] = np.einsum("tf,fd->td", mfcc, kernel, optimize=False)
        reference[1:] += bias
        reference[1:] += pos[1:]
        encoder_error = float(np.max(np.abs(actual - reference)))
        if actual.shape != (99, 64) or actual.dtype != np.float32 or encoder_error > 2e-5:
            raise AssertionError(f"Encoder mismatch: {path}: {encoder_error}")
        partition = split_for(path)
        label = path.split("/", 1)[0]
        if path.startswith("four/"):
            label = "_unknown_"
        records.append({
            "archive_member": path, "label": label, "split": partition,
            "wav_sha256": sha256(raw), "wav_bytes": len(raw), "wav_format": wav_info,
            "mfcc_file": mfcc_path.name, "mfcc_shape": list(mfcc.shape),
            "mfcc_sha256": sha256(mfcc.tobytes()), "mfcc_fft_max_abs_error": mfcc_error,
            "encoder_file": encoder_path.name, "encoder_shape": list(actual.shape),
            "encoder_sha256": sha256(actual.tobytes()),
            "encoder_independent_max_abs_error": encoder_error,
        })
    baseline = np.load(OUT / records[0]["encoder_file"], allow_pickle=False)
    baseline_mfcc = np.load(OUT / records[0]["mfcc_file"], allow_pickle=False)
    metamorphic_max = 0.0
    for row in records[1:]:
        encoder = np.load(OUT / row["encoder_file"], allow_pickle=False)
        mfcc = np.load(OUT / row["mfcc_file"], allow_pickle=False)
        if not np.array_equal(encoder[0], baseline[0]):
            raise AssertionError("Class-token row differs across samples")
        expected_delta = (mfcc - baseline_mfcc) @ kernel
        error = float(np.max(np.abs((encoder[1:] - baseline[1:]) - expected_delta)))
        row["metamorphic_projection_delta_max_abs_error"] = error
        metamorphic_max = max(metamorphic_max, error)
        if error > 3e-6:
            raise AssertionError(f"Projection-delta invariant failed: {row['archive_member']}: {error}")
    flags_path = ROOT.parent / "资料/ARM_KWT-1_237e4097/models_data_v2_12_labels/kwt1/flags.json"
    report = {"archive": str(ARCHIVE), "flags": str(flags_path),
              "flags_sha256": sha256(flags_path.read_bytes()), "weights": str(weights_path),
              "weights_sha256": sha256(weights_path.read_bytes()),
              "metamorphic_projection_delta_max_abs_error": metamorphic_max,
              "class_token_row_identical": True, "samples": records}
    (OUT / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"samples={len(records)} splits={sorted(set(r['split'] for r in records))} "
          f"labels={sorted(set(r['label'] for r in records))}")
    print(f"max_mfcc_fft_error={max(r['mfcc_fft_max_abs_error'] for r in records):.9g}")
    print(f"max_encoder_error={max(r['encoder_independent_max_abs_error'] for r in records):.9g}")
    print(f"class_row_identical=PASS max_projection_delta_error={metamorphic_max:.9g}")
    print(f"manifest={OUT / 'manifest.json'}")


if __name__ == "__main__":
    main()
