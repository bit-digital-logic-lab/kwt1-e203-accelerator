#!/usr/bin/env python3
"""Read-only sparsity profile of the frozen undergraduate FFN1 integer path."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "software"))
from quantize_ffn1 import collect_activations  # noqa: E402
from run import KWT1  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=Path, default=ROOT / "software/.build/quant_samples_12labels_40.npz")
    p.add_argument("--pack", type=Path, default=ROOT / "software/results/ffn1_a12w10_static_sym_12labels_40.npz")
    p.add_argument("--weights", type=Path, default=ROOT / "software/weights/kwt1_fp32.npz")
    p.add_argument("--output", type=Path, default=Path(__file__).with_name("profile.json"))
    p.add_argument("--partition", choices=("validation", "training"), default="validation")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    torch.set_num_threads(4)
    with np.load(a.samples, allow_pickle=False) as z:
        ids = np.flatnonzero(z["partitions"] == a.partition)
        if a.limit:
            ids = ids[:a.limit]
        features = z["features"][ids]
    if features.shape[1:] != (98, 40) or not len(features):
        raise ValueError("missing MFCC samples or wrong shape")
    with np.load(a.pack, allow_pickle=False) as z:
        weights = [z[f"layer{i:02}_weight"].copy() for i in range(1, 13)]
        scales = [float(z[f"layer{i:02}_activation_scale"]) for i in range(1, 13)]
    for w in weights:
        if w.shape != (64, 256):
            raise ValueError("wrong FFN1 weight shape")
    model = KWT1(a.weights, torch.device("cpu"))
    x = collect_activations(features, model)
    if x.shape != (len(ids), 12, 99, 64):
        raise ValueError(f"wrong activation shape {x.shape}")
    qx = np.stack([np.clip(np.rint(x[:, i].astype(np.float64) / scales[i]), -2047, 2047).astype(np.int16) for i in range(12)], axis=1)
    # Independent spot check of the encoder-to-integer conversion path.
    for i in (0, len(ids) - 1):
        for l in (0, 11):
            assert np.array_equal(qx[i, l], np.clip(np.rint(x[i, l] / scales[l]), -2047, 2047).astype(np.int16))
    w = np.stack(weights)
    wz = w == 0
    xz = qx == 0
    total_mac = int(len(ids) * 12 * 99 * 64 * 256)
    weight_zero_mac = int(wz.sum(dtype=np.int64) * len(ids) * 99)
    activation_zero_mac = int(xz.sum(dtype=np.int64) * 256)
    both_zero_mac = 0
    near_zero_mac = {"1": 0, "2": 0}
    tile_k_skips = 0
    total_tile_k = len(ids) * 12 * 7 * 16 * 64  # ceil(99/16) * ceil(256/16) * K
    block_skips = {str(b): 0 for b in (4, 8, 16)}
    total_blocks = {str(b): len(ids) * 12 * 7 * 16 * (64 // b) for b in (4, 8, 16)}
    row_zero = 0
    col_zero = int(np.all(wz, axis=1).sum())
    for sample in range(len(ids)):
        for layer in range(12):
            az = xz[sample, layer]
            bz = wz[layer]
            both_zero_mac += int((az.sum(axis=0, dtype=np.int64) * bz.sum(axis=1, dtype=np.int64)).sum())
            for threshold in (1, 2):
                an = np.abs(qx[sample, layer].astype(np.int32)) <= threshold
                bn = np.abs(w[layer].astype(np.int32)) <= threshold
                both = int((an.sum(axis=0, dtype=np.int64) * bn.sum(axis=1, dtype=np.int64)).sum())
                near_zero_mac[str(threshold)] += int(an.sum() * 256 + bn.sum() * 99 - both)
            row_zero += int(np.all(az, axis=1).sum())
            for rb in range(0, 99, 16):
                a_live = np.any(~az[rb:rb+16], axis=0)
                for cb in range(0, 256, 16):
                    active = a_live & np.any(~bz[:, cb:cb+16], axis=1)
                    tile_k_skips += int((~active).sum())
                    for size in (4, 8, 16):
                        block_skips[str(size)] += int((~active.reshape(-1, size).any(axis=1)).sum())
    zero_product = weight_zero_mac + activation_zero_mac - both_zero_mac
    brute = (qx[0, 0, :, :, None] == 0) | (w[0, None, :, :] == 0)
    formula = int(xz[0, 0].sum() * 256 + wz[0].sum() * 99 - (xz[0, 0].sum(axis=0, dtype=np.int64) * wz[0].sum(axis=1, dtype=np.int64)).sum())
    assert int(brute.sum()) == formula
    rec = {
        "method": "read-only, A12W10 exact integer zero; no pruning/rounding applied",
        "partition": a.partition, "samples": len(ids), "layers": 12,
        "inputs_sha256": {str(p): sha(p) for p in (a.samples, a.pack, a.weights)},
        "activation_zero_fraction": float(xz.mean()),
        "weight_zero_fraction": float(wz.mean()),
        "weight_abs_le_1_fraction": float((np.abs(w.astype(np.int32)) <= 1).mean()),
        "weight_abs_le_2_fraction": float((np.abs(w.astype(np.int32)) <= 2).mean()),
        "activation_abs_le_1_fraction": float((np.abs(qx.astype(np.int32)) <= 1).mean()),
        "activation_abs_le_2_fraction": float((np.abs(qx.astype(np.int32)) <= 2).mean()),
        "total_mac": total_mac,
        "exact_zero_product_mac": zero_product,
        "exact_zero_product_mac_fraction": zero_product / total_mac,
        "counterfactual_abs_le_threshold_product_mac_fraction": {k: v / total_mac for k, v in near_zero_mac.items()},
        "fully_zero_activation_rows": row_zero,
        "activation_rows": int(len(ids) * 12 * 99),
        "fully_zero_weight_columns": col_zero,
        "weight_columns": 12 * 256,
        "tile_16x16_k_skip": tile_k_skips,
        "tile_16x16_k_total": total_tile_k,
        "tile_16x16_k_skip_fraction": tile_k_skips / total_tile_k,
        "tile_16x16_k_block_skips": block_skips,
        "tile_16x16_k_block_totals": total_blocks,
        "bound_note": "MAC gating fraction is an operation/energy upper bound, not cycle speedup. Tile-K skips require per-cycle all-lane zero detection and dynamic K suppression; block skips require block metadata or scanning. Load/write/bias cycles remain.",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: rec[k] for k in ("samples", "activation_zero_fraction", "weight_zero_fraction", "exact_zero_product_mac_fraction", "tile_16x16_k_skip_fraction")}, indent=2))


if __name__ == "__main__":
    main()
