"""Compare the PyTorch FP32 path with TensorFlow operations on the same weights."""

import numpy as np
import tensorflow as tf
import torch

from run import KWT1, PREFIX, ROOT


def main():
    path = ROOT / "weights/kwt1_fp32.npz"
    with np.load(path, allow_pickle=False) as data:
        w = {key: tf.constant(data[key]) for key in data.files}

    def linear(x, base, bias=True):
        y = tf.linalg.matvec(w[base + "/kernel"], x, transpose_a=True) if x.shape.rank == 1 else tf.linalg.matmul(x, w[base + "/kernel"])
        return y + w[base + "/bias"] if bias else y

    def norm(x, base):
        mean, variance = tf.nn.moments(x, axes=[-1], keepdims=True)
        return (x - mean) * tf.math.rsqrt(variance + 1e-6) * w[base + "/gamma"] + w[base + "/beta"]

    features = np.random.default_rng(42).normal(size=(98, 40)).astype(np.float32)
    model = KWT1(path, torch.device("cpu"))
    with torch.inference_mode():
        px = model.encoder_input(torch.from_numpy(features))
        plogits, pout, ffn_inputs = model.forward(px, capture_ffn=True)

    x = linear(tf.constant(features), PREFIX + "/patch_proj")
    x = tf.concat((w[PREFIX + "/class_emb"][0], x), axis=0) + w[PREFIX + "/pos_emb"][0]
    assert np.max(np.abs(x.numpy() - px.numpy())) < 1e-5
    max_ffn_error = 0.0
    for i in range(12):
        p = f"{PREFIX}/enc_layers/{i}"
        a = p + "/att"
        q = linear(x, a + "/query_dense", False)
        k = linear(x, a + "/key_dense", False)
        v = linear(x, a + "/value_dense", False)
        attn = tf.linalg.matmul(tf.nn.softmax(tf.linalg.matmul(q, k, transpose_b=True) / tf.sqrt(64.0)), v)
        out1 = norm(x + linear(attn, a + "/combine_heads"), p + "/layernorm1")
        max_ffn_error = max(max_ffn_error, float(np.max(np.abs(out1.numpy() - ffn_inputs[i].numpy()))))
        h = linear(out1, p + "/ffn/layer_with_weights-0")
        h = tf.nn.gelu(h, approximate=False)
        x = norm(out1 + linear(h, p + "/ffn/layer_with_weights-1"), p + "/layernorm2")
    logits = linear(x[0], "layer_with_weights-1/layer_with_weights-0")
    encoder_error = float(np.max(np.abs(x.numpy() - pout.numpy())))
    logits_error = float(np.max(np.abs(logits.numpy() - plogits.numpy())))
    print(f"max_abs_error ffn_inputs={max_ffn_error:.8g} encoder={encoder_error:.8g} logits={logits_error:.8g}")
    if max(max_ffn_error, encoder_error, logits_error) > 1e-3:
        raise AssertionError("FP32 backend mismatch exceeds 1e-3")


if __name__ == "__main__":
    main()
