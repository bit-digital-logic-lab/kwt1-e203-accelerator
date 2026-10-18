#!/usr/bin/env bash
# 研究生组第 4-7 项环境检测入口：依次运行只读盘点与真实 smoke test。
# 脚本不安装/不下载/不改仓库；任一子脚本缺失会以可读错误退出。
set -uo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "$here/../.." && pwd)"
cd "$repo" || exit 1

status=0
echo "== 1/2 环境盘点 =="
python3 "$here/check_environment.py" || status=$?
echo
echo "== 2/2 工具链 smoke test =="
python3 "$here/smoke_toolchain.py" || status=$?
echo
echo "== 完成，聚合退出码: $status =="
echo "产物："
ls -1 "$here"/*.json "$here"/*.md 2>/dev/null
exit "$status"
