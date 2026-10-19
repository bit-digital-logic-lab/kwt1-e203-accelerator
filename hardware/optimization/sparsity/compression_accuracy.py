#!/usr/bin/env python3
"""Measure FFN1 weight-stream compression and its validation-set accuracy."""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "software"))
from quantize_ffn1 import collect_activations  # noqa: E402
from run import KWT1, PREFIX  # noqa: E402

K, N, LAYERS = 64, 256, 12
THRESHOLDS = (0, 1, 2, 4, 8, 16, 32, 64)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def encode(w):
    """Per-(K,16-column) u16 bitmap, u16 byte offsets, byte-padded W10 stream."""
    assert w.shape == (K, N) and np.max(np.abs(w)) <= 511
    masks = bytearray()
    data = bytearray()
    offsets = [0]
    for k in range(K):
        for c in range(0, N, 16):
            vals = w[k, c:c + 16]
            mask = sum(1 << j for j, v in enumerate(vals) if v != 0)
            masks += struct.pack("<H", mask)
            bits = 0
            count = 0
            for v in vals:
                if v != 0:
                    bits |= (int(v) & 1023) << count
                    count += 10
            data += bits.to_bytes((count + 7) // 8, "little")
            offsets.append(len(data))
    assert offsets[-1] <= 65535
    return bytes(masks + b"".join(struct.pack("<H", v) for v in offsets) + data)


def decode(blob):
    blocks = K * (N // 16)
    masks = struct.unpack_from(f"<{blocks}H", blob, 0)
    offsets = struct.unpack_from(f"<{blocks + 1}H", blob, blocks * 2)
    start = blocks * 2 + (blocks + 1) * 2
    out = np.zeros((K, N), dtype=np.int16)
    assert offsets[0] == 0 and start + offsets[-1] == len(blob)
    for i, mask in enumerate(masks):
        raw = int.from_bytes(blob[start + offsets[i]:start + offsets[i + 1]], "little")
        shift = 0
        for j in range(16):
            if mask & (1 << j):
                v = (raw >> shift) & 1023
                out[i // 16, (i % 16) * 16 + j] = v - 1024 if v >= 512 else v
                shift += 10
        assert offsets[i + 1] - offsets[i] == (shift + 7) // 8
    return out


def nrmse_sum(approx, ref):
    delta = approx.astype(np.float64) - ref.astype(np.float64)
    base = ref.astype(np.float64)
    return float(np.sum(delta * delta)), float(np.sum(base * base))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=Path, default=ROOT / "software/.build/quant_samples_12labels_40.npz")
    p.add_argument("--pack", type=Path, default=ROOT / "software/results/ffn1_a12w10_static_sym_12labels_40.npz")
    p.add_argument("--weights", type=Path, default=ROOT / "software/weights/kwt1_fp32.npz")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path(__file__).with_name("compression_accuracy.json"))
    args = p.parse_args()
    torch.set_num_threads(4)
    with np.load(args.samples, allow_pickle=False) as z:
        ids = np.flatnonzero(z["partitions"] == "validation")
        if args.limit:
            ids = ids[:args.limit]
        features = z["features"][ids]
    if not len(features) or features.shape[1:] != (98, 40):
        raise ValueError("expected validation MFCC [samples,98,40]")
    with np.load(args.pack, allow_pickle=False) as z:
        qweights = [z[f"layer{i:02}_weight"].copy() for i in range(1, 13)]
        qbiases = [z[f"layer{i:02}_bias"].copy() for i in range(1, 13)]
        xscales = [float(z[f"layer{i:02}_activation_scale"]) for i in range(1, 13)]
        wscales = [z[f"layer{i:02}_weight_scales"].copy() for i in range(1, 13)]
    model = KWT1(args.weights, torch.device("cpu"))
    x = collect_activations(features, model)
    assert x.shape == (len(features), LAYERS, 99, K)
    rows = {str(t): {"threshold": t, "layers": [], "encoded_weight_bytes": 0,
                     "zero_weights": 0} for t in THRESHOLDS}
    for layer in range(LAYERS):
        stem = f"{PREFIX}/enc_layers/{layer}/ffn/layer_with_weights-0"
        fw = model.weight(stem + "/kernel").cpu().numpy()
        fb = model.weight(stem + "/bias").cpu().numpy()
        qx = np.clip(np.rint(x[:, layer].astype(np.float64) / xscales[layer]),
                     -2047, 2047).astype(np.int32)
        # Every integer product/sum below is <2^53, so FP64 BLAS gives exact
        # integer accumulators. Check against an independent integer dot sample.
        spot = qx[0, 0].astype(np.int64) @ qweights[layer].astype(np.int64)
        check = qx[0, 0].astype(np.float64) @ qweights[layer].astype(np.float64)
        assert np.array_equal(spot, check)
        ref = x[:, layer] @ fw + fb
        xflat = qx.reshape(-1, K).astype(np.float64)
        for t in THRESHOLDS:
            qw = qweights[layer].copy()
            qw[np.abs(qw.astype(np.int32)) <= t] = 0
            blob = encode(qw)
            assert np.array_equal(decode(blob), qw), (layer, t)
            acc = xflat @ qw.astype(np.float64) + qbiases[layer]
            assert np.max(np.abs(acc)) < 2**31
            approx = (acc.reshape(len(features), 99, N) *
                      (xscales[layer] * wscales[layer])).astype(np.float32)
            err, den = nrmse_sum(approx, ref)
            rows[str(t)]["layers"].append({"layer": layer + 1, "nrmse_pct":
                100 * (err / max(den, 1e-16)) ** 0.5, "encoded_weight_bytes": len(blob)})
            rows[str(t)]["encoded_weight_bytes"] += len(blob)
            rows[str(t)]["zero_weights"] += int(np.count_nonzero(qw == 0))
        print(f"layer {layer + 1:02}/12 done", flush=True)
    dense_w10 = LAYERS * K * N * 10 // 8
    dense_w16 = LAYERS * K * N * 2
    for row in rows.values():
        row["worst_nrmse_pct"] = max(v["nrmse_pct"] for v in row["layers"])
        row["zero_weight_pct"] = 100 * row["zero_weights"] / (LAYERS * K * N)
        row["vs_dense_w10_pct"] = 100 * (row["encoded_weight_bytes"] / dense_w10 - 1)
        row["vs_dense_w16_pct"] = 100 * (row["encoded_weight_bytes"] / dense_w16 - 1)
    result = {"samples": len(features), "partition": "validation",
              "inputs_sha256": {str(path): sha256(path) for path in (args.samples, args.pack, args.weights)},
              "format": "1024 blocks/layer; u16 mask, 1025 u16 byte offsets, byte-padded signed W10 values",
              "dense_w10_bytes": dense_w10, "dense_w16_bytes": dense_w16,
              "metric": "max over 12 layers of sqrt(sum((approx-fp32)^2)/max(sum(fp32^2),1e-16))",
              "rows": list(rows.values())}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for row in rows.values():
        print(f"threshold={row['threshold']:2} zero={row['zero_weight_pct']:.3f}% "
              f"bytes={row['encoded_weight_bytes']} NRMSE={row['worst_nrmse_pct']:.5f}%", flush=True)


if __name__ == "__main__":
    main()
