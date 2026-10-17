"""Export the specified ARM KWT-1 TensorFlow checkpoint for the FP32 runner."""

from pathlib import Path
import argparse
import json

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "资料/ARM_KWT-1_237e4097/models_data_v2_12_labels/kwt1"
SUFFIX = "/.ATTRIBUTES/VARIABLE_VALUE"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "weights/kwt1_fp32.npz")
    args = parser.parse_args()

    import tensorflow as tf

    flags = json.loads((SOURCE / "flags.json").read_text())
    assert (flags["num_layers"], flags["d_model"], flags["mlp_dim"], flags["heads"]) == (12, 64, 256, 1)
    reader = tf.train.load_checkpoint(str(SOURCE / "best_weights"))
    arrays = {}
    for name, shape in tf.train.list_variables(str(SOURCE / "best_weights")):
        if not name.endswith(SUFFIX) or "OPTIMIZER_SLOT" in name or name.startswith("optimizer/"):
            continue
        key = name.removesuffix(SUFFIX)
        value = reader.get_tensor(name)
        if value.dtype != np.float32:
            raise ValueError(f"Expected FP32 weight: {name} {value.dtype}")
        arrays[key] = value

    expected = {"layer_with_weights-0/class_emb", "layer_with_weights-0/pos_emb",
                "layer_with_weights-0/patch_proj/kernel", "layer_with_weights-0/patch_proj/bias",
                "layer_with_weights-1/layer_with_weights-0/kernel",
                "layer_with_weights-1/layer_with_weights-0/bias"}
    for i in range(12):
        p = f"layer_with_weights-0/enc_layers/{i}"
        expected.update(f"{p}/att/{n}/kernel" for n in
                        ("query_dense", "key_dense", "value_dense", "combine_heads"))
        expected.add(f"{p}/att/combine_heads/bias")
        for n in ("layer_with_weights-0", "layer_with_weights-1"):
            expected.update((f"{p}/ffn/{n}/kernel", f"{p}/ffn/{n}/bias"))
        for n in ("layernorm1", "layernorm2"):
            expected.update((f"{p}/{n}/gamma", f"{p}/{n}/beta"))
    if set(arrays) != expected:
        raise ValueError(f"Checkpoint mismatch: missing={sorted(expected-set(arrays))}, extra={sorted(set(arrays)-expected)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)
    print(f"Exported {len(arrays)} FP32 tensors to {args.output}")
    print(f"Parameter count: {sum(a.size for a in arrays.values()):,}")


if __name__ == "__main__":
    main()
