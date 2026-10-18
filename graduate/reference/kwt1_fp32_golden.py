"""Independent pure-NumPy FP32 golden reference for KWT-1 graduate items 4-6.

Item 4 single-head attention, item 5 single PostNorm encoder layer, item 6 the
full 12-layer encoder with per-layer parameters plus the software 12-way
classification head.

The operation order is taken verbatim from the official Arm KWT-1 sources that
ship with this repository:

  * ``资料/ARM_KWT-1_237e4097/kws_streaming/models/transformer_utils.py``
      ``MultiHeadSelfAttention.attention``  -> ``q @ k^T``, ``/ sqrt(64)``,
      ``softmax(axis=-1)``, ``weights @ v``; ``combine_heads`` output projection.
      ``TransformerBlock.call`` (postnorm branch, the default) ->
        attn; out1 = LN1(inputs + attn); ffn; output = LN2(out1 + ffn).
  * ``资料/ARM_KWT-1_237e4097/kws_streaming/models/kws_transformer.py``
      Dense head ``label_count`` applied to the class token ``x[:, 0]``.

This module only needs NumPy; scipy and TensorFlow are intentionally not used so
that the unit tests can run on the plain system interpreter.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

REFERENCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = REFERENCE_DIR.parents[1]

WEIGHTS_DEFAULT = REPO_ROOT / "software/weights/kwt1_fp32.npz"
FLAGS_DEFAULT = (
    REPO_ROOT
    / "资料/ARM_KWT-1_237e4097/models_data_v2_12_labels/kwt1/flags.json"
)
ENCODER_SAMPLES_DEFAULT = REPO_ROOT / "software/validation/real_input_chain_20261017"
PREFIX = "layer_with_weights-0"
HEAD = "layer_with_weights-1/layer_with_weights-0"

# Official semantics fixed by flags.json (prenorm and approximate_gelu are absent
# from flags.json, so the model_parameters defaults apply: postnorm, exact GELU).
EXPECTED_LAYERS = 12
EXPECTED_D_MODEL = 64
EXPECTED_MLP_DIM = 256
EXPECTED_HEADS = 1
EXPECTED_TOKENS = 99          # 98 time frames + 1 class token
EXPECTED_MFCC = (98, 40)
LN_EPSILON = 1e-6
GELU_INV_SQRT2 = 1.0 / math.sqrt(2.0)

_ERF = np.frompyfunc(math.erf, 1, 1)


def load_flags(path: Path = FLAGS_DEFAULT) -> dict:
    return json.loads(Path(path).read_text())


def check_flags(flags: dict) -> dict:
    """Assert the semantics this golden implements and return the fixed spec."""
    expected = {
        "num_layers": EXPECTED_LAYERS,
        "d_model": EXPECTED_D_MODEL,
        "mlp_dim": EXPECTED_MLP_DIM,
        "heads": EXPECTED_HEADS,
        "attention_type": "time",
        "spectrogram_length": EXPECTED_MFCC[0],
        "dct_num_features": EXPECTED_MFCC[1],
        "label_count": 12,
    }
    for key, value in expected.items():
        if flags.get(key) != value:
            raise ValueError(f"Unexpected KWT-1 flag {key}={flags.get(key)!r}, expected {value!r}")
    if flags.get("prenorm", False):
        raise ValueError("flags request prenorm; this golden implements postnorm")
    if flags.get("approximate_gelu", False):
        raise ValueError("flags request approximate GELU; this golden implements exact GELU")
    return {
        "num_layers": EXPECTED_LAYERS,
        "d_model": EXPECTED_D_MODEL,
        "mlp_dim": EXPECTED_MLP_DIM,
        "heads": EXPECTED_HEADS,
        "num_tokens": EXPECTED_TOKENS,
        "attention_scale": math.sqrt(EXPECTED_D_MODEL),
        "norm": "postnorm",
        "gelu": "exact",
        "layernorm_epsilon": LN_EPSILON,
    }


def load_weights(path: Path = WEIGHTS_DEFAULT) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        weights = {name: np.ascontiguousarray(archive[name], dtype=np.float32)
                   for name in archive.files}
    for name, value in weights.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite FP32 weight: {name}")
    return weights


def sha256(tensor: np.ndarray) -> str:
    array = np.ascontiguousarray(tensor, dtype=np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest()


def describe(name: str, tensor: np.ndarray) -> dict:
    array = np.asarray(tensor, dtype=np.float32)
    return {
        "name": name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "abs_max": float(np.abs(array).max()),
        "finite": bool(np.isfinite(array).all()),
        "sha256": sha256(array),
    }


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = LN_EPSILON) -> np.ndarray:
    """Keras LayerNormalization: population variance, epsilon after the var."""
    mean = x.mean(axis=-1, keepdims=True, dtype=np.float32)
    var = x.var(axis=-1, keepdims=True, dtype=np.float32)
    return (x - mean) * np.reciprocal(np.sqrt(var + eps)).astype(np.float32) * gamma + beta


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact GELU = 0.5*x*(1+erf(x/sqrt(2))), matching tfa.layers.GELU and tf.nn.gelu."""
    scaled = (x * np.float32(GELU_INV_SQRT2)).astype(np.float32)
    erf = _ERF(scaled).astype(np.float32)
    return np.float32(0.5) * x * (np.float32(1.0) + erf)


def linear(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray | None) -> np.ndarray:
    y = x @ kernel
    return y if bias is None else y + bias


def attention(x: np.ndarray, weights: dict, layer: int, capture: dict | None = None,
              arrays: dict | None = None) -> np.ndarray:
    """Item 4: single-head Q/K/V projection, scaled QK^T, row softmax, weight V."""
    p = f"{PREFIX}/enc_layers/{layer}/att"
    query = linear(x, weights[f"{p}/query_dense/kernel"], None)
    key = linear(x, weights[f"{p}/key_dense/kernel"], None)
    value = linear(x, weights[f"{p}/value_dense/kernel"], None)
    scale = np.float32(math.sqrt(EXPECTED_D_MODEL))
    score = (query @ key.T) / scale
    probs = np.exp(score - score.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)
    attended = probs @ value
    projected = linear(attended, weights[f"{p}/combine_heads/kernel"],
                       weights[f"{p}/combine_heads/bias"])
    if arrays is not None:
        arrays.update({"q": query, "k": key, "v": value, "score": score, "softmax": probs,
                       "attended_v": attended, "attn_projected": projected})
    if capture is not None:
        capture.update({
            "q": describe("q", query), "k": describe("k", key), "v": describe("v", value),
            "score": describe("score", score), "softmax": describe("softmax", probs),
            "attended_v": describe("attended_v", attended),
            "attn_projected": describe("attn_projected", projected),
            "softmax_rowsum_max_dev": float(np.abs(probs.sum(axis=-1) - 1.0).max()),
        })
    return projected


def encoder_layer(x: np.ndarray, weights: dict, layer: int, capture: dict | None = None,
                  arrays: dict | None = None) -> np.ndarray:
    """Item 5: PostNorm encoder layer (attention, LN1, FFN 64->256->64, LN2)."""
    p = f"{PREFIX}/enc_layers/{layer}"
    projected = attention(x, weights, layer, capture, arrays)
    residual1 = x + projected
    out1 = layer_norm(residual1, weights[f"{p}/layernorm1/gamma"], weights[f"{p}/layernorm1/beta"])
    hidden = linear(out1, weights[f"{p}/ffn/layer_with_weights-0/kernel"],
                    weights[f"{p}/ffn/layer_with_weights-0/bias"])
    activated = gelu(hidden)
    ffn_out = linear(activated, weights[f"{p}/ffn/layer_with_weights-1/kernel"],
                     weights[f"{p}/ffn/layer_with_weights-1/bias"])
    output = layer_norm(out1 + ffn_out, weights[f"{p}/layernorm2/gamma"],
                        weights[f"{p}/layernorm2/beta"])
    if arrays is not None:
        arrays.update({"residual1": residual1, "ln1": out1, "ffn_hidden": hidden,
                       "gelu": activated, "ffn_out": ffn_out, "residual2": out1 + ffn_out,
                       "layer_output": output})
    if capture is not None:
        capture.update({
            "residual1": describe("residual1", residual1),
            "ln1": describe("ln1", out1),
            "ffn_hidden": describe("ffn_hidden", hidden),
            "gelu": describe("gelu", activated),
            "ffn_out": describe("ffn_out", ffn_out),
            "residual2": describe("residual2", out1 + ffn_out),
            "ln2": describe("ln2", output),
            "ln1_rowmean_max_abs": float(np.abs(out1.mean(axis=-1)).max()),
            "ln2_rowmean_max_abs": float(np.abs(output.mean(axis=-1)).max()),
        })
    return output


def forward(x0: np.ndarray, weights: dict, num_layers: int = EXPECTED_LAYERS,
            capture: bool = False, arrays: list | None = None):
    """Item 6: run all layers with per-layer weights, return logits and stages.

    ``x0`` is the (99, 64) FP32 encoder input (PC preprocessing output; the PC
    MFCC chain is deliberately outside the accelerator boundary).
    ``arrays`` (a list) receives the raw per-layer intermediate tensors.
    """
    x = np.ascontiguousarray(x0, dtype=np.float32)
    if x.shape != (EXPECTED_TOKENS, EXPECTED_D_MODEL):
        raise ValueError(f"Expected encoder input {(EXPECTED_TOKENS, EXPECTED_D_MODEL)}, got {x.shape}")
    stages = []
    for layer in range(num_layers):
        record = {"layer": layer} if capture else None
        layer_arrays = {} if arrays is not None else None
        x = encoder_layer(x, weights, layer, record, layer_arrays)
        if layer_arrays is not None:
            arrays.append(layer_arrays)
        if record is not None:
            record["layer_output"] = describe("layer_output", x)
            stages.append(record)
    head_logits = linear(x[0], weights[f"{HEAD}/kernel"], weights[f"{HEAD}/bias"])
    return head_logits, x, stages


def layer_parameter_signatures(weights: dict, num_layers: int = EXPECTED_LAYERS):
    """Per-layer parameter SHA-256, to prove the 12 layers use distinct weights."""
    signatures = []
    for layer in range(num_layers):
        p = f"{PREFIX}/enc_layers/{layer}"
        digest = hashlib.sha256()
        for suffix in (
            "att/query_dense/kernel", "att/key_dense/kernel", "att/value_dense/kernel",
            "att/combine_heads/kernel", "att/combine_heads/bias",
            "ffn/layer_with_weights-0/kernel", "ffn/layer_with_weights-0/bias",
            "ffn/layer_with_weights-1/kernel", "ffn/layer_with_weights-1/bias",
            "layernorm1/gamma", "layernorm1/beta", "layernorm2/gamma", "layernorm2/beta",
        ):
            digest.update(np.ascontiguousarray(weights[f"{p}/{suffix}"], dtype=np.float32).tobytes())
        signatures.append(digest.hexdigest())
    return signatures


if __name__ == "__main__":
    flags = check_flags(load_flags())
    print(json.dumps(flags, indent=2, ensure_ascii=False))
