#!/usr/bin/env python3
"""从指定 GitHub blob 分块下载并校验赛题指定的 KWT-1 权重。"""

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time


SHA = "9c38f9653119383f7a303a94689c85a2a1c79b6b"
URL = f"https://api.github.com/repos/ARM-software/keyword-transformer/git/blobs/{SHA}"
DEST = (
    Path(__file__).resolve().parent
    / "ARM_KWT-1_237e4097/models_data_v2_12_labels/kwt1"
    / "best_weights.data-00000-of-00001"
)
CHUNK_SIZE = 1024 * 1024


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def request_range(start: int, end: int, folder: Path, total: int | None = None) -> tuple[Path, int]:
    output = folder / f"{start:08d}.bin"
    headers = folder / f"{start:08d}.headers"
    for attempt in range(4):
        result = subprocess.run(
            [
                "curl", "--noproxy", "*", "--http1.1", "-fLsS",
                "--connect-timeout", "15", "--max-time", "90",
                "-H", f"Range: bytes={start}-{end}",
                "-D", str(headers), "-o", str(output), URL,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and output.exists() and output.stat().st_size == end - start + 1:
            match = re.search(
                r"(?im)^content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)",
                headers.read_text(),
            )
            if match:
                got_start, got_end, got_total = map(int, match.groups())
                if (got_start, got_end) == (start, end) and (total is None or got_total == total):
                    return output, got_total
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"下载字节 {start}-{end} 失败：{result.stderr.strip()}")


def main() -> None:
    if DEST.exists() and blob_sha(DEST.read_bytes()) == SHA:
        print(f"权重已存在且校验通过：{DEST}")
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kwt1-blob-") as temp:
        folder = Path(temp)
        _, total = request_range(0, 0, folder)
        ranges = [(start, min(start + CHUNK_SIZE - 1, total - 1)) for start in range(0, total, CHUNK_SIZE)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(request_range, start, end, folder, total): start for start, end in ranges}
            for future in as_completed(futures):
                future.result()
                print(f"已下载区块 {futures[future] // CHUNK_SIZE + 1}/{len(ranges)}", flush=True)
        payload = b"".join((folder / f"{start:08d}.bin").read_bytes() for start, _ in ranges)
        blob = json.loads(payload)
        if blob["sha"] != SHA:
            raise RuntimeError("GitHub 元数据中的 SHA 与指定权重不一致")
        data = base64.b64decode(blob["content"])
        if blob_sha(data) != SHA:
            raise RuntimeError("权重内容的 Git blob SHA 校验失败")
        partial = DEST.with_suffix(DEST.suffix + ".part")
        partial.write_bytes(data)
        partial.replace(DEST)
    print(f"权重已下载并校验：{DEST}")


if __name__ == "__main__":
    main()
