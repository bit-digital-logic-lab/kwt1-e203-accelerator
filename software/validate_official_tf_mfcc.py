"""Compare real WAV MFCC against ARM's pinned SpeechFeatures layer.

Run with PYTHONPATH=/tmp/kwt-official-237e4097 software/.venv/bin/python.
"""

import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import numpy as np
import tensorflow as tf

from kws_streaming.layers import modes, speech_features


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "validation/real_input_chain_20261017"
ARCHIVE = ROOT.parent / "资料/数据集/speech_commands_v0.02.tar.gz"
FLAGS = ROOT.parent / "资料/ARM_KWT-1_237e4097/models_data_v2_12_labels/kwt1/flags.json"


def main():
    report = json.loads((OUT / "manifest.json").read_text())
    wanted = {row["archive_member"] for row in report["samples"]}
    payloads = {}
    with tarfile.open(ARCHIVE, "r|gz") as archive:
        for member in archive:
            key = member.name.removeprefix("./")
            if key in wanted and member.isfile():
                payloads[key] = archive.extractfile(member).read()
                if len(payloads) == len(wanted):
                    break
    if payloads.keys() != wanted:
        raise RuntimeError(f"Missing WAV: {wanted - payloads.keys()}")
    flags = SimpleNamespace(**json.loads(FLAGS.read_text()))
    params = speech_features.SpeechFeatures.get_params(flags)
    layer = speech_features.SpeechFeatures(params, mode=modes.Modes.NON_STREAM_INFERENCE)
    maximum = 0.0
    for row in report["samples"]:
        pcm, sample_rate = tf.audio.decode_wav(payloads[row["archive_member"]], desired_channels=1)
        if int(sample_rate) != 16000:
            raise ValueError("Unexpected sample rate")
        pcm = tf.reshape(pcm, [-1])[:16000]
        pcm = tf.pad(pcm, [[0, 16000 - tf.shape(pcm)[0]]])
        official = layer(pcm[None, :]).numpy()[0]
        ours = np.load(OUT / row["mfcc_file"], allow_pickle=False)
        error = float(np.max(np.abs(official - ours)))
        rmse = float(np.sqrt(np.mean((official.astype(np.float64) - ours) ** 2)))
        row["mfcc_official_tf_max_abs_error"] = error
        row["mfcc_official_tf_rmse"] = rmse
        maximum = max(maximum, error)
        if official.shape != (98, 40) or not np.isfinite(official).all() or error > 1e-3:
            raise AssertionError(f"Official MFCC mismatch: {row['archive_member']}: {official.shape}, {error}")
        print(f"{row['archive_member']} max_abs_error={error:.9g} rmse={rmse:.9g}")
    report["mfcc_official_tf_max_abs_error"] = maximum
    report["official_source_commit"] = "237e4097afd4f6851e0b82f0a8d65c2dfe8eae30"
    (OUT / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"samples={len(report['samples'])} max_official_tf_mfcc_error={maximum:.9g}")


if __name__ == "__main__":
    main()
