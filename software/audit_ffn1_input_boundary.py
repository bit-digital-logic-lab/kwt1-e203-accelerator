"""Audit the desktop 98x40 -> 99x64 path and the per-layer FFN1 boundary.

Contest scope: the twelve hardware FFN1 inputs are the post-attention,
residual and *first* LayerNorm tensors of encoder layers 0..11.  The first
encoder input (99x64) is a different tensor and must not be reused as an FFN1
input.  This script checks both boundaries for KWT-1 without touching the RTL
or any quantization configuration.

Evidence design (no golden is read back from the implementation under test):

* Independent NumPy float64 re-implementation.  Raw WAV -> MFCC and the
  twelve-layer FP32 forward are recomputed from the frozen weight artifact
  and the raw Speech Commands V2 waveform.  ``offline_encoder_input.convert``
  and ``run.KWT1.forward(..., capture_ffn=True)`` are only ever the candidate,
  never the reference.
* Metamorphic invariants (no golden at all): class-token invariance, affine
  additivity and homogeneity of the 98x40 -> 99x64 projection, and token
  permutation equivariance of the captured FFN1 tensors.
* Boundary discrimination: the captured tensor is compared against wrong
  boundary candidates (pre-LN1 residual, FFN hidden, post-LN2) produced by the
  independent loop, so a capture taken at the wrong point cannot pass.

CPU only.  Shared inputs are read-only::

    资料/数据集/speech_commands_v0.02.tar.gz
    software/weights/kwt1_fp32.npz
    /home/yang/Desktop/ic设计竞赛/software/.build/quant_samples_12labels_40.npz
"""

import argparse
import hashlib
import io
import json
import math
import platform
import tarfile
from pathlib import Path
import wave

import numpy as np


ROOT = Path(__file__).resolve().parent
# The worktree does not copy the 2.3 GB archive; read the shared one read-only.
_SHARED_DATA = Path("/home/yang/Desktop/ic设计竞赛/资料/数据集")
_LOCAL_DATA = ROOT.parent / "资料/数据集"
ARCHIVE = ((_LOCAL_DATA if (_LOCAL_DATA / "speech_commands_v0.02.tar.gz").exists()
            else _SHARED_DATA) / "speech_commands_v0.02.tar.gz")
WEIGHTS = ROOT / "weights/kwt1_fp32.npz"
# Real, already-preprocessed samples (read-only shared cache) used by the fast
# unittest path.  Never written to.
SHARED_CACHE = Path("/home/yang/Desktop/ic设计竞赛/software/.build/quant_samples_12labels_40.npz")
OUT = ROOT / "validation/ffn1_input_boundary_audit"

PREFIX = "layer_with_weights-0/"
N_LAYERS = 12
D_MODEL = 64
N_TOKENS = 99
N_FRAMES = 98
N_MFCC = 40

# Two different real categories: one target word and one unknown-word clip.
RAW_SAMPLES = (
    ("yes/676f8138_nohash_0.wav", "yes"),
    ("four/750e3e75_nohash_0.wav", "_unknown_"),
)

# Candidate tolerances.  The implementation is float32 on CPU; the reference is
# float64.  Tolerances are reported with the measured values and are deliberately
# loose enough not to be brittle while still rejecting a wrong tensor.
TOL_MFCC = 1e-3          # independent FFT MFCC vs the project RDFT MFCC
TOL_ENCODER = 2e-5       # 98x40 -> 99x64 candidate vs independent float64 golden
TOL_FFN1 = 5e-5          # captured FFN1 input vs independent float64 golden
TOL_METAMORPHIC = 1e-3   # additivity / homogeneity / permutation invariants
MARGIN = 1e-2            # a wrong boundary must be at least this far away


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Independent reference (E2): NumPy float64, own feature and forward code.
# --------------------------------------------------------------------------
def independent_mfcc(audio: np.ndarray) -> np.ndarray:
    """98x40 log-mel DCT features, computed with NumPy's own FFT.

    Re-implements the pinned ``flags.json`` recipe (16 kHz, 480-sample Hann
    window, 160-sample stride, 512-point DFT, 80 mel bins, 40 DCT outputs)
    without using ``prepare_quant_samples.mfcc_matrices``.
    """
    frame, fft, hop, sample_rate = 480, 512, 160, 16000
    audio = np.asarray(audio, dtype=np.float64)
    if audio.shape != (16000,):
        raise ValueError(f"Expected 16000 mono samples, got {audio.shape}")
    frames = np.stack([audio[i * hop:i * hop + frame] for i in range(N_FRAMES)])
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(frame) / frame)
    spectrum = np.abs(np.fft.rfft(frames * window, n=fft, axis=1))

    def hz_to_mel(hz):
        return 1127.0 * np.log1p(hz / 700.0)

    freqs = np.fft.rfftfreq(fft, 1.0 / sample_rate)
    spectrum_mel = hz_to_mel(freqs)
    edges = np.linspace(hz_to_mel(20.0), hz_to_mel(7600.0), 82)
    bank = np.zeros((fft // 2 + 1, 80))
    for i in range(80):
        lower = (spectrum_mel - edges[i]) / (edges[i + 1] - edges[i])
        upper = (edges[i + 2] - spectrum_mel) / (edges[i + 2] - edges[i + 1])
        bank[:, i] = np.maximum(0.0, np.minimum(lower, upper))
    log_mel = np.log(np.maximum(spectrum @ bank, 1e-12))
    dct_k = np.arange(N_MFCC)
    dct = 2.0 * np.cos(np.pi * (2 * np.arange(80)[:, None] + 1) * dct_k[None, :]
                       / (2.0 * 80)) / np.sqrt(2.0 * 80)
    return (log_mel @ dct).astype(np.float32)


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Keras population-variance LayerNormalization with epsilon 1e-6."""
    mean = x.mean(axis=-1, keepdims=True)
    variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + 1e-6) * gamma + beta


def independent_encode(mfcc: np.ndarray, w: dict) -> np.ndarray:
    """Independent 98x40 -> 99x64 projection with CLS token and positions."""
    out = np.empty((N_TOKENS, D_MODEL), dtype=np.float64)
    out[0] = w[PREFIX + "class_emb"][0, 0] + w[PREFIX + "pos_emb"][0, 0]
    out[1:] = (mfcc.astype(np.float64) @ w[PREFIX + "patch_proj/kernel"].astype(np.float64)
               + w[PREFIX + "patch_proj/bias"])
    out[1:] += w[PREFIX + "pos_emb"][0, 1:]
    return out


def independent_forward(x: np.ndarray, w: dict) -> dict:
    """Independent twelve-layer forward capturing candidate boundaries.

    ``ffn1`` is the tensor under audit.  ``residual`` (pre-LN1), ``combined``
    (attention output) and ``ln2`` (post-FFN LayerNorm) are the wrong-boundary
    candidates used for discrimination; ``layer_input`` is the tensor entering
    the layer.
    """
    x = x.astype(np.float64)
    audit = {"ffn1": [], "residual": [], "combined": [], "layer_input": [], "ln2": []}
    for i in range(N_LAYERS):
        p = f"{PREFIX}enc_layers/{i}"
        a = p + "/att"
        layer_input = x
        q = x @ w[a + "/query_dense/kernel"].astype(np.float64)
        k = x @ w[a + "/key_dense/kernel"].astype(np.float64)
        v = x @ w[a + "/value_dense/kernel"].astype(np.float64)
        scores = q @ k.T / math.sqrt(D_MODEL)
        scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        attn = scores / scores.sum(axis=1, keepdims=True)
        combined = attn @ v @ w[a + "/combine_heads/kernel"].astype(np.float64) \
            + w[a + "/combine_heads/bias"]
        residual = x + combined
        ffn1 = layer_norm(residual, w[p + "/layernorm1/gamma"].astype(np.float64),
                          w[p + "/layernorm1/beta"].astype(np.float64))
        hidden = ffn1 @ w[p + "/ffn/layer_with_weights-0/kernel"].astype(np.float64) \
            + w[p + "/ffn/layer_with_weights-0/bias"]
        hidden = 0.5 * hidden * (1.0 + np.vectorize(math.erf)(hidden / math.sqrt(2.0)))
        projected = hidden @ w[p + "/ffn/layer_with_weights-1/kernel"].astype(np.float64) \
            + w[p + "/ffn/layer_with_weights-1/bias"]
        ln2 = layer_norm(ffn1 + projected, w[p + "/layernorm2/gamma"].astype(np.float64),
                         w[p + "/layernorm2/beta"].astype(np.float64))
        audit["ffn1"].append(ffn1)
        audit["residual"].append(residual)
        audit["combined"].append(combined)
        audit["layer_input"].append(layer_input)
        audit["ln2"].append(ln2)
        x = ln2
    logits = x[0] @ w["layer_with_weights-1/layer_with_weights-0/kernel"].astype(np.float64) \
        + w["layer_with_weights-1/layer_with_weights-0/bias"]
    audit["encoder"] = x
    audit["logits"] = logits
    return audit


# --------------------------------------------------------------------------
# Samples.
# --------------------------------------------------------------------------
def wav_to_audio(raw: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(raw)) as handle:
        if (handle.getnchannels(), handle.getframerate(), handle.getsampwidth()) != (1, 16000, 2):
            raise ValueError("Expected mono 16 kHz PCM16 WAV")
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    audio = np.zeros(16000, dtype=np.float64)
    audio[:min(len(samples), 16000)] = samples[:16000].astype(np.float64) / 32768.0
    return audio.astype(np.float32)


def read_raw_samples(paths):
    found = {}
    with tarfile.open(ARCHIVE, "r|gz") as archive:
        for member in archive:
            key = member.name.removeprefix("./")
            if key in paths and member.isfile():
                found[key] = archive.extractfile(member).read()
                if len(found) == len(paths):
                    break
    missing = set(paths) - set(found)
    if missing:
        raise RuntimeError(f"Missing WAV archive members: {sorted(missing)}")
    return found


def read_cache_samples(categories, partition="validation"):
    if not SHARED_CACHE.exists():
        raise FileNotFoundError(f"Shared sample cache is unavailable: {SHARED_CACHE}")
    samples = []
    with np.load(SHARED_CACHE, allow_pickle=False) as data:
        features, words, partitions, paths = (data["features"], data["words"],
                                              data["partitions"], data["paths"])
        for category in categories:
            match = np.flatnonzero((words == category) & (partitions == partition))
            if not len(match):
                raise RuntimeError(f"No {partition} sample for category {category!r}")
            index = int(match[0])
            samples.append((str(paths[index]), category, features[index].copy()))
    return samples


def load_weights(path=WEIGHTS):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def projection_weights(w):
    """The four tensors consumed by the 98x40 -> 99x64 projection."""
    return {PREFIX + s: w[PREFIX + s] for s in
            ("patch_proj/kernel", "patch_proj/bias", "class_emb", "pos_emb")}


# --------------------------------------------------------------------------
# Checks.
# --------------------------------------------------------------------------
def check_sample_metadata(mfcc, path):
    if mfcc.shape != (N_FRAMES, N_MFCC) or mfcc.dtype != np.float32 or not np.isfinite(mfcc).all():
        raise AssertionError(f"Bad MFCC for {path}: {mfcc.shape} {mfcc.dtype}")


def run_sample(name, label, mfcc_independent, project_mfcc, w, torch_mod, model, convert):
    """Audit one real sample.  Returns a JSON-serialisable record."""
    check_sample_metadata(mfcc_independent, name)
    mfcc_project_error = float(np.max(np.abs(mfcc_independent - project_mfcc))) \
        if project_mfcc is not None else None
    if mfcc_project_error is not None and mfcc_project_error > TOL_MFCC:
        raise AssertionError(f"Independent MFCC mismatch for {name}: {mfcc_project_error}")

    # 98x40 -> 99x64 boundary.  The golden is the independent float64 tensor.
    golden_encoder = independent_encode(mfcc_independent, w)
    actual_encoder = convert(mfcc_independent, projection_weights(w))
    encoder_error = float(np.max(np.abs(actual_encoder - golden_encoder)))
    if actual_encoder.shape != (N_TOKENS, D_MODEL) or actual_encoder.dtype != np.float32:
        raise AssertionError(f"Bad encoder shape for {name}: {actual_encoder.shape} {actual_encoder.dtype}")
    if encoder_error > TOL_ENCODER:
        raise AssertionError(f"98x40 -> 99x64 mismatch for {name}: {encoder_error}")

    # Class token and first frame, checked explicitly against the raw tensors.
    cls = w[PREFIX + "class_emb"][0, 0]
    pos = w[PREFIX + "pos_emb"][0]
    np.testing.assert_allclose(actual_encoder[0], cls + pos[0], rtol=0, atol=0)
    first = mfcc_independent[0] @ w[PREFIX + "patch_proj/kernel"] + w[PREFIX + "patch_proj/bias"]
    np.testing.assert_allclose(actual_encoder[1], first + pos[1], rtol=0, atol=TOL_ENCODER)

    # Per-layer FFN1 input boundary.  Candidate == implementation capture.
    golden = independent_forward(golden_encoder, w)
    with torch_mod.inference_mode():
        x = torch_mod.from_numpy(actual_encoder).to(torch_mod.device("cpu"))
        _, _, caps = model.forward(x, capture_ffn=True)
    if len(caps) != N_LAYERS:
        raise AssertionError(f"Expected {N_LAYERS} captured FFN1 inputs, got {len(caps)}")
    layers = []
    for i in range(N_LAYERS):
        captured = caps[i].numpy()
        if captured.shape != (N_TOKENS, D_MODEL) or captured.dtype != np.float32:
            raise AssertionError(f"Bad FFN1 capture {i}: {captured.shape} {captured.dtype}")
        errors = {key: float(np.max(np.abs(captured - golden[key][i])))
                  for key in ("ffn1", "residual", "combined", "layer_input", "ln2")}
        if errors["ffn1"] > TOL_FFN1:
            raise AssertionError(f"FFN1 layer {i} mismatch for {name}: {errors['ffn1']}")
        wrong = min(errors[key] for key in ("residual", "combined", "layer_input", "ln2"))
        if errors["ffn1"] + MARGIN > wrong:
            raise AssertionError(f"FFN1 layer {i} boundary not discriminated for {name}: "
                                 f"ffn1={errors['ffn1']:.3g} nearest_wrong={wrong:.3g}")
        layers.append({"layer": i,
                       "capture_sha256": sha256(captured.tobytes()),
                       "ffn1_max_abs_error": errors["ffn1"],
                       "wrong_boundary_max_abs_error": {k: errors[k] for k in
                                                        ("residual", "combined",
                                                         "layer_input", "ln2")}})
    # The twelve tensors must be distinct; a stale capture would repeat layer 0.
    distinct = min(float(np.max(np.abs(caps[i].numpy() - caps[j].numpy())))
                   for i in range(N_LAYERS) for j in range(i + 1, N_LAYERS))
    if distinct <= 0.0:
        raise AssertionError(f"Duplicate FFN1 captures for {name}")

    return {
        "sample": name,
        "label": label,
        "mfcc_shape": list(mfcc_independent.shape),
        "mfcc_sha256": sha256(mfcc_independent.tobytes()),
        "mfcc_independent_vs_project_max_abs_error": mfcc_project_error,
        "encoder_shape": list(actual_encoder.shape),
        "encoder_sha256": sha256(actual_encoder.tobytes()),
        "encoder_independent_max_abs_error": encoder_error,
        "ffn1_inputs_shape": [list(c.shape) for c in caps],
        "ffn1_distinct_min_max_abs": distinct,
        "layers": layers,
    }


def metamorphic_checks(mfcc_a, mfcc_b, w, torch_mod, model, convert):
    """Invariants of the 98x40 -> 99x64 map and of the FFN1 capture."""
    bias = w[PREFIX + "patch_proj/bias"]
    pos1 = w[PREFIX + "pos_emb"][0, 1:]
    cls = w[PREFIX + "class_emb"][0, 0]
    pos0 = w[PREFIX + "pos_emb"][0, 0]

    enc_a = convert(mfcc_a, projection_weights(w))
    enc_b = convert(mfcc_b, projection_weights(w))
    enc_sum = convert((mfcc_a.astype(np.float64) + mfcc_b).astype(np.float32),
                      projection_weights(w))
    additivity = float(np.max(np.abs(enc_sum[1:] - enc_a[1:] - enc_b[1:] + bias + pos1)))
    if additivity > TOL_METAMORPHIC:
        raise AssertionError(f"Projection additivity failed: {additivity}")

    scale = np.float32(0.5)
    enc_half = convert((scale * mfcc_a).astype(np.float32), projection_weights(w))
    expected_half = scale * (enc_a[1:] - bias - pos1) + bias + pos1
    homogeneity = float(np.max(np.abs(enc_half[1:] - expected_half)))
    if homogeneity > TOL_METAMORPHIC:
        raise AssertionError(f"Projection homogeneity failed: {homogeneity}")

    # Class token is identical for every sample and equals class_emb + pos_emb[0].
    class_identical = bool(np.array_equal(enc_a[0], enc_b[0]))
    if not class_identical or not np.array_equal(enc_a[0], cls + pos0):
        raise AssertionError("Class token row invariant failed")

    # Token permutation equivariance of the captured FFN1 tensors.
    idx = np.random.default_rng(20261018).permutation(N_TOKENS)
    with torch_mod.inference_mode():
        _, _, caps = model.forward(torch_mod.from_numpy(enc_a).to(torch_mod.device("cpu")),
                                   capture_ffn=True)
        _, _, caps_perm = model.forward(torch_mod.from_numpy(enc_a[idx]).to(torch_mod.device("cpu")),
                                        capture_ffn=True)
    permutation = max(float(np.max(np.abs(caps_perm[i].numpy() - caps[i].numpy()[idx])))
                      for i in range(N_LAYERS))
    if permutation > TOL_METAMORPHIC:
        raise AssertionError(f"FFN1 permutation equivariance failed: {permutation}")

    return {"projection_additivity_max_abs_error": additivity,
            "projection_homogeneity_max_abs_error": homogeneity,
            "class_token_row_identical": class_identical,
            "ffn1_token_permutation_max_abs_error": permutation}


def sweep(count, w, torch_mod, model, convert, seed=20261018):
    """Broader numeric audit over real cached samples of all 12 categories."""
    if not SHARED_CACHE.exists():
        raise FileNotFoundError(f"Shared sample cache is unavailable: {SHARED_CACHE}")
    with np.load(SHARED_CACHE, allow_pickle=False) as data:
        features, words, partitions = data["features"], data["words"], data["partitions"]
    candidates = np.flatnonzero(partitions == "validation")
    rng = np.random.default_rng(seed)
    picked = rng.choice(candidates, size=min(count, len(candidates)), replace=False)
    max_encoder, max_ffn1, min_margin = 0.0, 0.0, float("inf")
    categories = set()
    for index in picked:
        mfcc = features[index].astype(np.float32)
        categories.add(str(words[index]))
        golden_encoder = independent_encode(mfcc, w)
        actual_encoder = convert(mfcc, projection_weights(w))
        encoder_error = float(np.max(np.abs(actual_encoder - golden_encoder)))
        golden = independent_forward(golden_encoder, w)
        with torch_mod.inference_mode():
            _, _, caps = model.forward(torch_mod.from_numpy(actual_encoder).to(torch_mod.device("cpu")),
                                       capture_ffn=True)
        for layer in range(N_LAYERS):
            captured = caps[layer].numpy()
            ffn1_error = float(np.max(np.abs(captured - golden["ffn1"][layer])))
            wrong = min(float(np.max(np.abs(captured - golden[key][layer])))
                        for key in ("residual", "combined", "layer_input", "ln2"))
            max_encoder = max(max_encoder, encoder_error)
            max_ffn1 = max(max_ffn1, ffn1_error)
            min_margin = min(min_margin, wrong)
    if max_encoder > TOL_ENCODER:
        raise AssertionError(f"Sweep encoder error too large: {max_encoder}")
    if max_ffn1 > TOL_FFN1:
        raise AssertionError(f"Sweep FFN1 error too large: {max_ffn1}")
    if max_ffn1 + MARGIN > min_margin:
        raise AssertionError(f"Sweep boundary not discriminated: {max_ffn1} vs {min_margin}")
    return {"samples": int(len(picked)), "seed": seed,
            "categories": sorted(categories),
            "encoder_independent_max_abs_error": max_encoder,
            "ffn1_independent_max_abs_error": max_ffn1,
            "wrong_boundary_min_max_abs_error": min_margin}


def environment():
    import torch
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "device_used": "cpu",
        "tensorflow_available": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--cache", action="store_true",
                        help="Use the read-only shared sample cache instead of the WAV archive")
    parser.add_argument("--sweep", type=int, default=0, metavar="N",
                        help="Additionally audit N random cached validation samples")
    args = parser.parse_args()

    import torch
    import sys
    sys.path.insert(0, str(ROOT))
    from offline_encoder_input import convert
    from run import KWT1
    from prepare_quant_samples import mfcc_batch, mfcc_matrices

    args.output.mkdir(parents=True, exist_ok=True)
    w = load_weights()
    model = KWT1(WEIGHTS, torch.device("cpu"))
    matrices = mfcc_matrices()

    if args.cache:
        cache_samples = read_cache_samples([label for _, label in RAW_SAMPLES])
        samples = []
        for path, label, mfcc in cache_samples:
            samples.append({"name": path, "label": label, "mfcc_independent": mfcc,
                            "mfcc_project": None, "wav_sha256": None, "wav_bytes": None})
        source = {"kind": "shared_cache", "path": str(SHARED_CACHE),
                  "sha256": sha256_file(SHARED_CACHE)}
    else:
        members = [path for path, _ in RAW_SAMPLES]
        payloads = read_raw_samples(members)
        samples = []
        for path, label in RAW_SAMPLES:
            audio = wav_to_audio(payloads[path])
            samples.append({"name": path, "label": label,
                            "mfcc_independent": independent_mfcc(audio),
                            "mfcc_project": mfcc_batch(audio[None, :], matrices)[0],
                            "wav_sha256": sha256(payloads[path]),
                            "wav_bytes": len(payloads[path])})
        source = {"kind": "wav_archive", "path": str(ARCHIVE), "bytes": ARCHIVE.stat().st_size}

    records = []
    for sample in samples:
        record = run_sample(sample["name"], sample["label"], sample["mfcc_independent"],
                            sample["mfcc_project"], w, torch, model, convert)
        record["wav_sha256"] = sample["wav_sha256"]
        record["wav_bytes"] = sample["wav_bytes"]
        records.append(record)
        print(f"{sample['name']} encoder_err={record['encoder_independent_max_abs_error']:.9g} "
              f"ffn1_err_max={max(l['ffn1_max_abs_error'] for l in record['layers']):.9g} "
              f"wrong_boundary_min={min(min(l['wrong_boundary_max_abs_error'].values()) for l in record['layers']):.9g}")

    metamorphic = metamorphic_checks(samples[0]["mfcc_independent"],
                                     samples[1]["mfcc_independent"], w, torch, model, convert)

    report = {
        "scope": "98x40 MFCC -> 99x64 first encoder input, and per-layer FFN1 input boundary",
        "weights": str(WEIGHTS), "weights_sha256": sha256_file(WEIGHTS),
        "source": source,
        "tolerances": {"mfcc": TOL_MFCC, "encoder": TOL_ENCODER, "ffn1": TOL_FFN1,
                       "metamorphic": TOL_METAMORPHIC, "boundary_margin": MARGIN},
        "environment": environment(),
        "samples": records,
        "metamorphic": metamorphic,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    if args.sweep:
        report["sweep"] = sweep(args.sweep, w, torch, model, convert)
        (args.output / "manifest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"sweep={report['sweep']}")

    max_encoder = max(r["encoder_independent_max_abs_error"] for r in records)
    max_ffn1 = max(l["ffn1_max_abs_error"] for r in records for l in r["layers"])
    print(f"PASS samples={len(records)} max_encoder_error={max_encoder:.9g} "
          f"max_ffn1_error={max_ffn1:.9g} "
          f"metamorphic_permutation={metamorphic['ffn1_token_permutation_max_abs_error']:.9g}")
    print(f"manifest={args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
