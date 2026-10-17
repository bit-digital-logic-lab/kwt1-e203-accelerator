"""Export the FP32 KWT-1 first encoder input from one 98x40 MFCC matrix.

The contest's twelve FFN1 inputs are later, layer-specific post-attention
LayerNorm tensors. This command only produces the input to encoder layer 0.
"""

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
FLAGS = ROOT.parent / "资料/ARM_KWT-1_237e4097/models_data_v2_12_labels/kwt1/flags.json"
PREFIX = "layer_with_weights-0/"


def convert(mfcc: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    """Apply the checkpoint's 40->64 projection, prepend CLS, add positions."""
    if mfcc.shape != (98, 40) or mfcc.dtype != np.float32 or not np.isfinite(mfcc).all():
        raise ValueError("MFCC must be finite FP32 with shape (98,40)")
    expected = {
        "patch_proj/kernel": (40, 64),
        "patch_proj/bias": (64,),
        "class_emb": (1, 1, 64),
        "pos_emb": (1, 99, 64),
    }
    for suffix, shape in expected.items():
        value = weights[PREFIX + suffix]
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"Invalid checkpoint tensor {PREFIX + suffix}: {value.shape}, {value.dtype}")
    projected = mfcc @ weights[PREFIX + "patch_proj/kernel"]
    projected += weights[PREFIX + "patch_proj/bias"]
    tokens = np.concatenate((weights[PREFIX + "class_emb"][0], projected), axis=0)
    output = tokens + weights[PREFIX + "pos_emb"][0]
    assert output.shape == (99, 64) and output.dtype == np.float32
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="One FP32 MFCC .npy, shape (98,40)")
    parser.add_argument("--output", type=Path, required=True, help="FP32 encoder input .npy, shape (99,64)")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/kwt1_fp32.npz")
    args = parser.parse_args()
    flags = json.loads(FLAGS.read_text())
    if (flags["spectrogram_length"], flags["dct_num_features"], flags["d_model"],
            flags["num_layers"], flags["attention_type"]) != (98, 40, 64, 12, "time"):
        raise ValueError("Unexpected KWT-1 checkpoint configuration")
    mfcc = np.load(args.input, allow_pickle=False)
    with np.load(args.weights, allow_pickle=False) as archive:
        weights = {PREFIX + suffix: archive[PREFIX + suffix] for suffix in
                   ("patch_proj/kernel", "patch_proj/bias", "class_emb", "pos_emb")}
    output = convert(mfcc, weights)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output, allow_pickle=False)
    print(f"saved={args.output} shape={output.shape} dtype={output.dtype} bytes={output.nbytes}")


if __name__ == "__main__":
    main()
