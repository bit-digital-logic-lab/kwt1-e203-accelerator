#!/usr/bin/env python3
"""r2_preprocess — independent, element-wise verification of the offline path
WAV -> MFCC(98x40) -> first encoder input(99x64), including the 98->99 filling
(class-token prepend + positional-embedding add) and the .npy serialization.

Scope / honesty boundary
------------------------
* This task verifies ONLY the computer-side offline preprocessing chain. It says
  nothing about RTL bit-exactness, E203 SoC integration, synthesis/timing/area,
  or the official 600-sample acceptance set (all of which are reported below as
  *not proven by this task*).
* The reference values are produced by an INDEPENDENT float64 NumPy reimplementa-
  tion written here (FFT spectrum + mel filterbank + orthonormal DCT-II, and an
  explicit class-token/positional-embedding construction). They are never taken
  from the artefact under test.
* Inputs are read-only: the frozen checkpoint NPZ, the official flags.json, the
  gitignored Speech Commands V2 archive in the main project, and the shipped
  `.npy` artefacts. Nothing outside `graduate/ds_tasks/r2_preprocess/` is written.
* Split discipline: the official `split_for()` hash rule labels every sample as
  training/validation. This task performs NO calibration of any scale/quantisation
  parameter, and every tolerance below is an a-priori float32-rounding constant
  (not fitted on any split). Training and validation samples are treated identi-
  cally, purely as evaluation inputs.

Exit code is 0 only if every gate passes.
"""

import argparse
import ast
import hashlib
import io
import json
import struct
import sys
import tarfile
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WT_ROOT = HERE.parents[2]              # worktree root
SOFTWARE = WT_ROOT / "software"
SOURCE = WT_ROOT / "资料/ARM_KWT-1_237e4097/models_data_v2_12_labels/kwt1"
DEFAULT_ARCHIVE = Path("/home/yang/Desktop/ic设计竞赛/资料/数据集/speech_commands_v0.02.tar.gz")
DEFAULT_WEIGHTS = SOFTWARE / "weights/kwt1_fp32.npz"
DEFAULT_SAMPLES = SOFTWARE / "validation/real_input_chain_20261017"
MANIFEST_NAME = "manifest.json"

# --- a-priori float32-rounding tolerances (NOT fitted on any data split) --------
ATOL_MFCC_E2 = 1e-3          # matches the shipped production guard in validate_real_input_chain.py
ATOL_ENCODER_EXACT = 2e-5    # same input -> only float32 reassociation, shipped production guard
ATOL_ENCODER_E2E = 1e-3      # independent WAV->MFCC->encoder vs stored
ATOL_PADDING_EXACT = 0.0     # class-token row + positional offset must be bit-exact
POS_OFFSET_DISCRIMINATOR = 1e-3  # a wrong positional offset must be far above this

TARGET_MEMBERS = [
    "four/750e3e75_nohash_0.wav",
    "go/0ba018fc_nohash_4.wav",
    "no/45692b02_nohash_3.wav",
    "no/7e7ca854_nohash_1.wav",
    "yes/676f8138_nohash_0.wav",
    "yes/ec21c46b_nohash_1.wav",
]
# Words whose first archive member is additionally used for an independent path
# cross-check (no stored artefact required). Together with TARGET_MEMBERS this
# covers all ten target words.
EXTRA_WORDS = ["up", "down", "left", "right", "on", "off", "stop"]

PREFIX = "layer_with_weights-0/"


# --------------------------------------------------------------------------- #
# independent reimplementations (never read the artefact under test)
# --------------------------------------------------------------------------- #
def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def indep_decode_wav(raw: bytes):
    """Independent WAV parse: mono/16 kHz/PCM16 -> float64 in [-1, 1)."""
    with wave.open(io.BytesIO(raw)) as handle:
        info = {
            "channels": handle.getnchannels(),
            "sample_rate_hz": handle.getframerate(),
            "sample_width_bytes": handle.getsampwidth(),
            "frames": handle.getnframes(),
            "compression": handle.getcomptype(),
        }
        data = handle.readframes(handle.getnframes())
    if (info["channels"], info["sample_rate_hz"], info["sample_width_bytes"]) != (1, 16000, 2):
        raise ValueError(f"unexpected WAV layout {info}")
    pcm = np.frombuffer(data, dtype="<i2")
    audio = np.zeros(16000, dtype=np.float64)
    n = min(len(pcm), 16000)
    audio[:n] = pcm[:n].astype(np.float64) / 32768.0
    return audio, info, int(len(pcm))


def indep_mfcc(audio: np.ndarray) -> np.ndarray:
    """Independent float64 MFCC: rfft spectrum + explicit mel + orthonormal DCT-II."""
    frame, fft, stride = 480, 512, 160
    nframes = (len(audio) - frame) // stride + 1
    idx = np.arange(frame, dtype=np.float64)
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * idx / frame)
    frames = np.lib.stride_tricks.sliding_window_view(audio, frame)[::stride]
    windowed = frames * hann
    spectrum = np.abs(np.fft.rfft(windowed, n=fft, axis=-1))              # (98, 257)

    nfft = fft // 2 + 1
    hertz = np.linspace(0.0, 8000.0, nfft)
    to_mel = lambda hz: 1127.0 * np.log1p(hz / 700.0)                     # noqa: E731
    mel_hz = to_mel(hertz)
    edges = np.linspace(to_mel(20.0), to_mel(7600.0), 82)
    mel = np.zeros((nfft, 80), dtype=np.float64)
    for i in range(80):
        lower = (mel_hz - edges[i]) / (edges[i + 1] - edges[i])
        upper = (edges[i + 2] - mel_hz) / (edges[i + 2] - edges[i + 1])
        mel[:, i] = np.maximum(0.0, np.minimum(lower, upper))
    mel[0, :] = 0.0
    last = int(np.flatnonzero(mel[:, -1])[-1]) + 1                        # mel_non_zero_only
    logmel = np.log(np.maximum(spectrum[:, :last] @ mel[:last], 1e-12))   # (98, 80)

    m = np.arange(80)[:, None]
    k = np.arange(40)[None, :]
    dct = np.sqrt(2.0 / 80.0) * np.cos(np.pi * (2 * m + 1) * k / 160.0)   # (80, 40)
    return (logmel @ dct).astype(np.float32), {"nframes": nframes, "mel_last_bin": last}


def indep_encoder(mfcc_f32: np.ndarray, kernel, bias, cls_row, pos) -> np.ndarray:
    """Independent 98->99 construction in float64, cast back to float32.

    row 0            = class_emb + pos_emb[0]
    rows 1..98       = mfcc @ kernel + bias + pos_emb[1..98]
    """
    projected = mfcc_f32.astype(np.float64) @ kernel.astype(np.float64)
    projected = projected + bias.astype(np.float64)
    out = np.empty((99, 64), dtype=np.float64)
    out[0] = cls_row.astype(np.float64) + pos[0].astype(np.float64)
    out[1:] = projected + pos[1:].astype(np.float64)
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# .npy serialization inspection
# --------------------------------------------------------------------------- #
def parse_npy(raw: bytes) -> dict:
    if raw[:6] != b"\x93NUMPY":
        raise ValueError("bad npy magic")
    major, minor = raw[6], raw[7]
    if major == 1:
        hlen = struct.unpack("<H", raw[8:10])[0]
        data_off, header = 10 + hlen, raw[10:10 + hlen]
    elif major in (2, 3):
        hlen = struct.unpack("<I", raw[8:12])[0]
        data_off, header = 12 + hlen, raw[12:12 + hlen]
    else:
        raise ValueError(f"unsupported npy version {major}.{minor}")
    return {
        "version": f"{major}.{minor}",
        "header_len": hlen,
        "header": header.decode("latin1").strip(),
        "fields": ast.literal_eval(header.decode("latin1")),
        "data_offset": data_off,
    }


def npy_roundtrip(array: np.ndarray):
    """Re-serialize with the production writer and confirm lossless byte identity."""
    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=False)
    reloaded = np.load(io.BytesIO(buf.getvalue()), allow_pickle=False)
    return buf.getvalue(), reloaded


def rel_nrmse(a: np.ndarray, b: np.ndarray) -> float:
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    denom = np.sqrt(np.mean(b64 ** 2))
    return float(np.sqrt(np.mean((a64 - b64) ** 2)) / denom) if denom else float("nan")


# --------------------------------------------------------------------------- #
def stream_archive(archive_path: Path, targets, extra_words):
    """Single streaming pass; return target payloads and one extra WAV per word."""
    found, extras, scanned = {}, {}, 0
    wanted = set(targets)
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            scanned += 1
            key = member.name.removeprefix("./")
            if not member.isfile() or not key.endswith(".wav"):
                continue
            if key in wanted and key not in found:
                found[key] = archive.extractfile(member).read()
            else:
                word = key.split("/", 1)[0]
                if word in extra_words and word not in extras:
                    extras[word] = (key, archive.extractfile(member).read())
            if len(found) == len(wanted) and len(extras) == len(extra_words):
                break
    return found, extras, scanned


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()

    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    gates = []

    def gate(name, ok, detail):
        gates.append({"name": name, "pass": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        return ok

    # ---- frozen inputs ----------------------------------------------------
    manifest_path = args.samples_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    weights_sha = sha256(args.weights.read_bytes())
    flags_path = SOURCE / "flags.json"
    flags_sha = sha256(flags_path.read_bytes())
    flags = json.loads(flags_path.read_text())

    # ---- production reference path (read-only import) ---------------------
    sys.path.insert(0, str(SOFTWARE))
    import prepare_quant_samples as PQS
    from prepare_quant_samples import mfcc_batch, mfcc_matrices, split_for, wav_float
    import offline_encoder_input as OEI
    from offline_encoder_input import convert

    matrices = mfcc_matrices()

    with np.load(args.weights, allow_pickle=False) as archive:
        kernel = archive[PREFIX + "patch_proj/kernel"].copy()
        bias = archive[PREFIX + "patch_proj/bias"].copy()
        cls_row = archive[PREFIX + "class_emb"][0, 0].copy()
        pos = archive[PREFIX + "pos_emb"][0].copy()

    print(f"archive={args.archive} exists={args.archive.exists()}")
    print(f"flags sha256={flags_sha}")
    print(f"weights sha256={weights_sha}")
    print(f"config spectrogram_length={flags['spectrogram_length']} dct_num_features={flags['dct_num_features']} "
          f"d_model={flags['d_model']} num_layers={flags['num_layers']} attention_type={flags['attention_type']}")

    # ---- frame-count derivation (why 98 -> 99) ----------------------------
    frame, stride, samples = flags["window_size_samples"], flags["window_stride_samples"], flags["desired_samples"]
    derived_frames = (samples - frame) // stride + 1
    derivation = {"desired_samples": samples, "window_size_samples": frame,
                  "window_stride_samples": stride, "derived_mfcc_rows": derived_frames,
                  "class_token_rows": 1, "encoder_rows": derived_frames + 1,
                  "config_spectrogram_length": flags["spectrogram_length"]}
    gate("frame_count_derivation", derived_frames == 98 and derived_frames + 1 == 99,
         json.dumps(derivation, ensure_ascii=False))

    # ---- single archive pass ---------------------------------------------
    t0 = time.time()
    found, extras, scanned = stream_archive(args.archive, TARGET_MEMBERS, EXTRA_WORDS)
    print(f"archive_members_scanned={scanned} targets_found={len(found)} extras_found={len(extras)} "
          f"secs={time.time() - t0:.1f}")
    gate("archive_all_targets_found", set(found) == set(TARGET_MEMBERS),
         f"found={sorted(found)}")

    manifest_by_member = {row["archive_member"]: row for row in manifest["samples"]}

    # ---- per-sample verification on the six stored artefacts -------------
    samples = []
    for index, member in enumerate(TARGET_MEMBERS):
        row = manifest_by_member.get(member)
        if row is None:
            gate(f"manifest_row::{member}", False, "missing from manifest")
            continue
        raw = found[member]
        wav_sha = sha256(raw)
        audio64, wav_info, pcm_len = indep_decode_wav(raw)
        mfcc_ind, mfcc_meta = indep_mfcc(audio64)

        stored_mfcc_path = args.samples_dir / row["mfcc_file"]
        stored_enc_path = args.samples_dir / row["encoder_file"]
        raw_mfcc_file = stored_mfcc_path.read_bytes()
        raw_enc_file = stored_enc_path.read_bytes()
        stored_mfcc = np.load(io.BytesIO(raw_mfcc_file), allow_pickle=False)
        stored_enc = np.load(io.BytesIO(raw_enc_file), allow_pickle=False)

        # (1) provenance: WAV bytes hash against the manifest
        wav_ok = (wav_sha == row["wav_sha256"]
                  and all(wav_info.get(k) == v for k, v in row["wav_format"].items())
                  and pcm_len == 16000)
        # (2) MFCC: independent float64 vs stored, element-wise
        mfcc_err = float(np.max(np.abs(mfcc_ind.astype(np.float64) - stored_mfcc.astype(np.float64))))
        mfcc_exact = int(np.count_nonzero(mfcc_ind == stored_mfcc))
        mfcc_ok = (stored_mfcc.shape == (98, 40) and stored_mfcc.dtype == np.float32
                   and stored_mfcc.flags["C_CONTIGUOUS"] and np.isfinite(stored_mfcc).all()
                   and mfcc_ind.shape == (98, 40) and mfcc_err <= ATOL_MFCC_E2)
        # (2b) production MFCC path reproduces the stored artefact (self-consistency)
        mfcc_prod = mfcc_batch(wav_float(raw)[None, :], matrices)[0]
        mfcc_prod_err = float(np.max(np.abs(mfcc_prod - stored_mfcc)))

        # (3) strict 99x64 reconstruction: SAME stored MFCC through production convert()
        enc_exact = convert(stored_mfcc, {PREFIX + s: v for s, v in
                                          (("patch_proj/kernel", kernel), ("patch_proj/bias", bias),
                                           ("class_emb", cls_row[None, None, :]), ("pos_emb", pos[None]))})
        enc_exact_err = float(np.max(np.abs(enc_exact - stored_enc)))
        # production writer round-trip == the stored file bytes
        enc_file_bytes, _ = npy_roundtrip(enc_exact)
        enc_file_identical = enc_file_bytes == raw_enc_file

        # (4) independent 99x64 reconstruction (own MFCC + own construction)
        enc_ind = indep_encoder(mfcc_ind, kernel, bias, cls_row, pos)
        enc_e2e_err = float(np.max(np.abs(enc_ind.astype(np.float64) - stored_enc.astype(np.float64))))
        enc_e2e_nrmse = rel_nrmse(enc_ind, stored_enc)

        # (5) 98->99 filling: class-token row and positional-embedding offset
        cls_expected = (cls_row + pos[0]).astype(np.float32)
        cls_err = float(np.max(np.abs(stored_enc[0].astype(np.float64) - cls_expected.astype(np.float64))))
        ideal_body = enc_exact[1:].astype(np.float64) - pos[1:].astype(np.float64)   # projected+bias
        pos_correct = float(np.max(np.abs((ideal_body + pos[1:]) - stored_enc[1:].astype(np.float64))))
        pos_shifted = float(np.max(np.abs((ideal_body + pos[0:98]) - stored_enc[1:].astype(np.float64))))
        pos_absent = float(np.max(np.abs(ideal_body - stored_enc[1:].astype(np.float64))))
        cls_vs_frame0 = float(np.max(np.abs(stored_enc[0].astype(np.float64)
                                            - (ideal_body[0] + pos[0]))))
        padding_ok = (cls_err <= ATOL_PADDING_EXACT and pos_correct <= ATOL_PADDING_EXACT
                      and pos_shifted > POS_OFFSET_DISCRIMINATOR and pos_absent > POS_OFFSET_DISCRIMINATOR)

        # (6) .npy serialization of both files
        ser = {}
        for tag, raw_file, arr, expect_shape, hashkey in (
                ("mfcc", raw_mfcc_file, stored_mfcc, (98, 40), "mfcc_sha256"),
                ("encoder", raw_enc_file, stored_enc, (99, 64), "encoder_sha256")):
            hdr = parse_npy(raw_file)
            fields = hdr["fields"]
            nbytes = int(np.prod(expect_shape)) * 4
            rt_bytes, rt_arr = npy_roundtrip(arr)
            ser[tag] = {
                "file_bytes": len(raw_file),
                "file_sha256": sha256(raw_file),
                "npy_version": hdr["version"],
                "descr": fields["descr"],
                "fortran_order": bool(fields["fortran_order"]),
                "header_shape": list(fields["shape"]),
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "c_contiguous": bool(arr.flags["C_CONTIGUOUS"]),
                "data_offset": hdr["data_offset"],
                "payload_bytes": len(raw_file) - hdr["data_offset"],
                "expected_payload_bytes": nbytes,
                "array_bytes_sha256": sha256(arr.tobytes()),
                "manifest_array_sha256": row[hashkey],
                "array_bytes_match_manifest": sha256(arr.tobytes()) == row[hashkey],
                "roundtrip_byte_identical": rt_bytes == raw_file,
                "roundtrip_array_equal": bool(np.array_equal(rt_arr, arr)),
                "finite": bool(np.isfinite(arr).all()),
            }
        ser_ok = all(
            s["descr"] == "<f4" and s["fortran_order"] is False and s["c_contiguous"] is True
            and s["payload_bytes"] == s["expected_payload_bytes"]
            and s["array_bytes_match_manifest"] and s["roundtrip_byte_identical"]
            and s["roundtrip_array_equal"] and s["finite"]
            and s["shape"] == s["header_shape"] for s in ser.values())

        split = split_for(member)
        label = "_unknown_" if member.startswith("four/") else member.split("/", 1)[0]
        record = {
            "index": index, "archive_member": member, "split": split, "label": label,
            "wav_sha256": wav_sha, "wav_sha256_matches_manifest": wav_sha == row["wav_sha256"],
            "wav_format": wav_info, "wav_pcm_frames": pcm_len,
            "mfcc_meta": mfcc_meta,
            "mfcc_independent_max_abs": mfcc_err, "mfcc_exact_elements": mfcc_exact,
            "mfcc_total_elements": int(stored_mfcc.size),
            "mfcc_production_vs_stored_max_abs": mfcc_prod_err,
            "encoder_same_input_max_abs": enc_exact_err,
            "encoder_file_byte_identical": enc_file_identical,
            "encoder_independent_max_abs": enc_e2e_err,
            "encoder_independent_nrmse": enc_e2e_nrmse,
            "padding": {
                "class_token_row_max_abs": cls_err,
                "pos_offset_correct_max_abs": pos_correct,
                "pos_offset_shifted_max_abs": pos_shifted,
                "pos_absent_max_abs": pos_absent,
                "class_token_vs_frame0_max_abs": cls_vs_frame0,
            },
            "serialization": ser,
            "class_token_row_sha256": sha256(stored_enc[0].tobytes()),
        }
        samples.append(record)

        ok = (wav_ok and mfcc_ok and enc_exact_err <= ATOL_ENCODER_EXACT and enc_file_identical
              and enc_e2e_err <= ATOL_ENCODER_E2E and padding_ok and ser_ok)
        gate(f"sample::{'/'.join(member.split('/'))}",
             ok, f"split={split} mfcc_err={mfcc_err:.3g} enc_same={enc_exact_err:.3g} "
                 f"enc_e2e={enc_e2e_err:.3g} nrmse={enc_e2e_nrmse:.3g} "
                 f"cls={cls_err:.3g} pos_shift={pos_shifted:.3g}")

    # class-token row identical across all six stored samples
    cls_hashes = {s["class_token_row_sha256"] for s in samples}
    gate("class_token_row_identical_across_samples", len(cls_hashes) == 1, f"{len(cls_hashes)} distinct")

    # ---- independent path cross-check on extra real samples --------------
    extra_records = []
    for word in EXTRA_WORDS:
        if word not in extras:
            continue
        member, raw = extras[word]
        audio64, wav_info, pcm_len = indep_decode_wav(raw)
        mfcc_ind, _ = indep_mfcc(audio64)
        mfcc_prod = mfcc_batch(wav_float(raw)[None, :], matrices)[0]
        enc_ind = indep_encoder(mfcc_ind, kernel, bias, cls_row, pos)
        enc_prod = convert(mfcc_prod, {PREFIX + s: v for s, v in
                                       (("patch_proj/kernel", kernel), ("patch_proj/bias", bias),
                                        ("class_emb", cls_row[None, None, :]), ("pos_emb", pos[None]))})
        mfcc_err = float(np.max(np.abs(mfcc_ind.astype(np.float64) - mfcc_prod.astype(np.float64))))
        enc_err = float(np.max(np.abs(enc_ind.astype(np.float64) - enc_prod.astype(np.float64))))
        extra_records.append({
            "archive_member": member, "label": word, "split": split_for(member),
            "wav_sha256": sha256(raw), "wav_format": wav_info, "wav_pcm_frames": pcm_len,
            "mfcc_independent_vs_production_max_abs": mfcc_err,
            "encoder_independent_vs_production_max_abs": enc_err,
        })

    if extra_records:
        worst_mfcc = max(r["mfcc_independent_vs_production_max_abs"] for r in extra_records)
        worst_enc = max(r["encoder_independent_vs_production_max_abs"] for r in extra_records)
    else:
        worst_mfcc = worst_enc = float("nan")
    gate("extra_real_samples_path_cross_check", len(extra_records) >= 1
         and worst_mfcc <= ATOL_MFCC_E2 and worst_enc <= ATOL_MFCC_E2,
         f"n={len(extra_records)} worst_mfcc={worst_mfcc:.3g} worst_encoder={worst_enc:.3g}")

    # ---- split discipline -------------------------------------------------
    n_train = sum(1 for s in samples if s["split"] == "training")
    n_val = sum(1 for s in samples if s["split"] == "validation")
    split_note = {
        "rule": "official Speech Commands hash rule (prepare_quant_samples.split_for)",
        "stored_samples_training": n_train,
        "stored_samples_validation": n_val,
        "calibration_performed": False,
        "thresholds_source": "a-priori float32-rounding constants; not fitted on any split",
        "validation_used_for": "evaluation only; no scale/threshold derived from validation samples",
    }
    gate("split_discipline", n_train >= 1 and n_val >= 1 and n_train + n_val == len(samples),
         json.dumps(split_note, ensure_ascii=False))

    # ---- evidence levels --------------------------------------------------
    evidence = {
        "software_reference": {
            "status": "PROVEN (this task)",
            "level": "E2 independent reimplementation + E3 metamorphic/padding-discriminating + E4 serialization self-consistency",
            "detail": "independent float64 NumPy WAV->MFCC->99x64 vs shipped artefacts on 6 real samples "
                      "with stored artifacts, plus independent-vs-production cross-check on extra real samples",
        },
        "rtl_bit_exact": {"status": "NOT PROVEN BY THIS TASK", "level": "n/a",
                          "detail": "offline preprocessing is computer-side; no RTL in this task"},
        "e203_soc": {"status": "NOT PROVEN BY THIS TASK", "level": "n/a",
                     "detail": "no E203 instruction stream / System Bus / NICE integration exercised"},
        "synthesis": {"status": "NOT PROVEN BY THIS TASK", "level": "n/a",
                      "detail": "no PDK/SDC; no 150 MHz timing, area or power sign-off"},
        "official_acceptance": {"status": "NOT PROVEN BY THIS TASK", "level": "n/a",
                                "detail": "6 real samples with stored artifacts (+7 extras); not the official 600-sample set"},
    }

    all_pass = all(g["pass"] for g in gates)
    result = {
        "task": "r2_preprocess",
        "started": started,
        "all_gates_pass": all_pass,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "cwd": str(Path.cwd()),
        },
        "inputs": {
            "archive": str(args.archive),
            "archive_bytes": args.archive.stat().st_size if args.archive.exists() else None,
            "weights": str(args.weights), "weights_sha256": weights_sha,
            "flags": str(flags_path), "flags_sha256": flags_sha,
            "samples_dir": str(args.samples_dir), "manifest": str(manifest_path),
            "manifest_weights_sha256": manifest.get("weights_sha256"),
        },
        "frame_count_derivation": derivation,
        "counts": {
            "stored_real_samples": len(samples),
            "extra_real_samples": len(extra_records),
            "total_real_samples": len(samples) + len(extra_records),
            "training": n_train, "validation": n_val,
            "mfcc_elements_checked": sum(s["mfcc_total_elements"] for s in samples),
            "encoder_elements_checked": sum(99 * 64 for s in samples),
        },
        "aggregate": {
            "mfcc_independent_max_abs": max(s["mfcc_independent_max_abs"] for s in samples),
            "encoder_same_input_max_abs": max(s["encoder_same_input_max_abs"] for s in samples),
            "encoder_independent_max_abs": max(s["encoder_independent_max_abs"] for s in samples),
            "encoder_independent_worst_nrmse": max(s["encoder_independent_nrmse"] for s in samples),
            "padding_class_token_max_abs": max(s["padding"]["class_token_row_max_abs"] for s in samples),
            "padding_pos_shifted_min_max_abs": min(s["padding"]["pos_offset_shifted_max_abs"] for s in samples),
            "extra_worst_mfcc": worst_mfcc, "extra_worst_encoder": worst_enc,
        },
        "split_discipline": split_note,
        "evidence_levels": evidence,
        "samples": samples,
        "extra_samples": extra_records,
        "gates": gates,
    }
    (out_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"gates={sum(g['pass'] for g in gates)}/{len(gates)} all_pass={all_pass}")
    print(f"results={out_dir / 'results.json'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
