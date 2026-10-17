"""Run the specified KWT-1 FP32 encoder and classifier with PyTorch."""

from pathlib import Path
import argparse
import json
import math

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "资料/ARM_KWT-1_237e4097/models_data_v2_12_labels/kwt1"
PREFIX = "layer_with_weights-0"


class KWT1:
    def __init__(self, weight_path: Path, device: torch.device):
        with np.load(weight_path, allow_pickle=False) as data:
            self.w = {k: torch.from_numpy(data[k].copy()).to(device) for k in data.files}
        self.device = device
        if len(self.w) != 162:
            raise ValueError(f"Expected 162 model tensors, got {len(self.w)}")

    def weight(self, name):
        return self.w[name]

    def linear(self, x, base, bias=True):
        y = x @ self.weight(base + "/kernel")
        return y + self.weight(base + "/bias") if bias else y

    def layer_norm(self, x, base):
        # Keras LayerNormalization uses population variance and epsilon=1e-6.
        return F.layer_norm(x, (64,), self.weight(base + "/gamma"),
                            self.weight(base + "/beta"), eps=1e-6)

    def encoder_input(self, features):
        if tuple(features.shape) != (98, 40):
            raise ValueError(f"Expected MFCC features (98, 40), got {tuple(features.shape)}")
        x = self.linear(features, PREFIX + "/patch_proj")
        x = torch.cat((self.weight(PREFIX + "/class_emb")[0], x), dim=0)
        return x + self.weight(PREFIX + "/pos_emb")[0]

    def forward(self, x, capture_ffn=False):
        if tuple(x.shape) != (99, 64):
            raise ValueError(f"Expected encoder input (99, 64), got {tuple(x.shape)}")
        ffn_inputs = []
        for i in range(12):
            p = f"{PREFIX}/enc_layers/{i}"
            a = p + "/att"
            q = self.linear(x, a + "/query_dense", False)
            k = self.linear(x, a + "/key_dense", False)
            v = self.linear(x, a + "/value_dense", False)
            score = (q @ k.T) / math.sqrt(64.0)
            attn = torch.softmax(score, dim=-1) @ v
            out1 = self.layer_norm(x + self.linear(attn, a + "/combine_heads"), p + "/layernorm1")
            if capture_ffn:
                ffn_inputs.append(out1)
            h = self.linear(out1, p + "/ffn/layer_with_weights-0")
            h = F.gelu(h, approximate="none")
            h = self.linear(h, p + "/ffn/layer_with_weights-1")
            x = self.layer_norm(out1 + h, p + "/layernorm2")
        logits = self.linear(x[0], "layer_with_weights-1/layer_with_weights-0")
        return logits, x, ffn_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help=".npy FP32 input: (98,40) MFCC or (99,64) encoder input")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/kwt1_fp32.npz")
    parser.add_argument("--output", type=Path, help="Save logits and final encoder output as .npz")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", action="store_true", help="Use seeded synthetic MFCC input; only checks execution")
    args = parser.parse_args()
    if args.smoke == bool(args.input):
        parser.error("Specify exactly one of --input and --smoke")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    flags = json.loads((SOURCE / "flags.json").read_text())
    if (flags["num_layers"], flags["d_model"], flags["mlp_dim"], flags["heads"]) != (12, 64, 256, 1):
        raise ValueError("Unexpected KWT-1 flags")
    model = KWT1(args.weights, device)
    array = np.random.default_rng(0).normal(size=(98, 40)).astype(np.float32) if args.smoke else np.load(args.input, allow_pickle=False)
    if array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError("Input must contain finite FP32 values")
    with torch.inference_mode():
        x = torch.from_numpy(array).to(device)
        if tuple(x.shape) == (98, 40):
            x = model.encoder_input(x)
        logits, encoder, _ = model.forward(x)
    labels = (SOURCE / "labels.txt").read_text().splitlines()
    scores = logits.cpu().numpy()
    prediction = int(scores.argmax())
    print(f"device={device} input={tuple(array.shape)} encoder={tuple(encoder.shape)}")
    print(f"top1={prediction} label={labels[prediction]} logits={np.array2string(scores, precision=5)}")
    if args.smoke:
        print("SMOKE ONLY: synthetic input is not an accuracy measurement")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output, logits=scores, encoder=encoder.cpu().numpy())
        print(f"saved={args.output}")


if __name__ == "__main__":
    main()
