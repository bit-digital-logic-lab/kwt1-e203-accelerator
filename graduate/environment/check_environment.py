#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""研究生组第 4-7 项：环境与接口约定检测（只读探测，不改工程文件）。

职责与边界
==========
本脚本只做三件事，并且对缺失项返回可读状态而不是伪 PASS：

1. 探测宿主软件工具链：Python / TensorFlow / PyTorch、Verilator / Icarus /
   Yosys / STA / RV32 交叉编译等，记录真实路径、版本字符串与命令退出码。
2. 校验官方资料哈希：按 ``资料/来源校验.json`` 逐文件比对 SHA-1，并给权重与
   数据集补充 SHA-256 / MD5。
3. 检查"真实资产"是否存在：E203 RTL、PDK、SRAM 宏、SDC、官方评测集等，并
   明确区分 in-repo / external-local / official-missing 三类来源。

输出 ``environment_report.json``（机器可读）与 ``environment_report.md``（简短
文档）。脚本本身不安装软件、不下载数据、不修改仓库，退出码恒为 0；缺失与否体现
在 JSON 的 ``present``/``status`` 字段。真正的工具链 smoke test 见
``smoke_toolchain.py``。

用法::

    python3 graduate/environment/check_environment.py
    python3 graduate/environment/check_environment.py --json-only
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1

# 官方赛题声明的固定条件（截图 4/5/6/13）：官方平台应为这些资产负责。
OFFICIAL_REQUIREMENTS = [
    {
        "id": "official_soc_platform",
        "name": "组委会官方 E203 SoC 平台（System Bus / NICE 接线、地址映射、共享 512KB SRAM）",
        "why_it_matters": "第 3 项 SoC 集成、第 6 项十二层调度、第 7 项综合与性能都以官方平台为验收对象。",
    },
    {
        "id": "official_pdk",
        "name": "组委会指定工艺 PDK 与标准单元库（用于 150 MHz / 面积签核）",
        "why_it_matters": "官方给出的 150 MHz 与 ≤2.25 mm² 只有在指定工艺库下才有意义。",
    },
    {
        "id": "official_sram_macros",
        "name": "指定内部 SRAM 宏：16 bits × 128 / 1024 / 4096 单口或双口",
        "why_it_matters": "面积、端口冲突和每拍读写能力必须按指定宏核算，不能拿通用 fakeram 代替。",
    },
    {
        "id": "official_sdc",
        "name": "官方 150 MHz 时序约束（SDC）与综合脚本",
        "why_it_matters": "没有官方时钟/输入输出延迟定义就无法做可比的 150 MHz 时序检查。",
    },
    {
        "id": "official_eval_set",
        "name": "官方统一评测集（600 条样本）与官方评测脚本",
        "why_it_matters": "本科/研究生数值门槛与准确率必须按官方 600 条口径复算。",
    },
]


# --------------------------------------------------------------------------- #
# 通用工具函数
# --------------------------------------------------------------------------- #
def run(cmd: list[str], timeout: int = 30) -> dict:
    """执行命令，返回 {ok, exit_code, stdout, stderr, error}。绝不抛异常。"""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="replace",
        )
        return {
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "error": None,
        }
    except FileNotFoundError:
        return {"ok": False, "exit_code": None, "stdout": "", "stderr": "", "error": "not found"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "exit_code": None, "stdout": "", "stderr": "", "error": "timeout"}
    except Exception as exc:  # pragma: no cover - 兜底
        return {"ok": False, "exit_code": None, "stdout": "", "stderr": "", "error": str(exc)}


def first_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def git_blob_sha1(path: Path) -> str:
    """Git blob 哈希：sha1(b"blob <size>\\0" + content)。

    ``资料/来源校验.json`` 的 ``sha`` 字段与 ``资料/补全大文件.sh`` 的
    ``git hash-object`` 一致，是 Git blob 哈希，而不是裸文件内容的 SHA-1。
    误用裸 SHA-1 会造成 25/25 全 mismatch 的假象。
    """
    size = path.stat().st_size
    h = hashlib.sha1()
    h.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_module(module: str, python_exe: str) -> dict:
    """在指定解释器里查模块，并用 importlib.metadata 读版本（不真正 import 重型库）。"""
    if python_exe == sys.executable:
        found = importlib.util.find_spec(module) is not None
    else:
        code = (
            "import importlib.util,sys,json\n"
            f"spec=importlib.util.find_spec({module!r})\n"
            "print(json.dumps(bool(spec)))\n"
        )
        res = run([python_exe, "-c", code])
        found = res["stdout"].strip() == "true"
    version = None
    if found:
        vcode = (
            "import importlib.metadata as m, json\n"
            f"try:\n    print(json.dumps(m.version({module!r})))\n"
            "except Exception:\n    print('null')\n"
        )
        if python_exe == sys.executable:
            try:
                import importlib.metadata as m

                version = m.version(module)
            except Exception:
                version = None
        else:
            res = run([python_exe, "-c", vcode])
            try:
                version = json.loads(res["stdout"])
            except Exception:
                version = None
    return {"found": found, "version": version}


# --------------------------------------------------------------------------- #
# 工具链探测
# --------------------------------------------------------------------------- #
def candidate_tools(repo: Path) -> list[dict]:
    oss = os.environ.get("OSS_CAD_SUITE", "/home/yang/yosys-sta/oss-cad-suite")
    return [
        {
            "id": "python3",
            "category": "python",
            "candidates": ["python3"],
            "version_cmd": ["--version"],
            "required": True,
        },
        {
            "id": "verilator",
            "category": "hdl_sim",
            "candidates": ["verilator"],
            "version_cmd": ["--version"],
            "required": True,
        },
        {
            "id": "iverilog",
            "category": "hdl_sim",
            "candidates": ["iverilog"],
            "version_cmd": ["-V"],
            "required": True,
        },
        {
            "id": "yosys_system",
            "category": "synthesis",
            "candidates": ["/usr/bin/yosys"],
            "version_cmd": ["-V"],
            "required": False,
        },
        {
            "id": "yosys_oss_cad",
            "category": "synthesis",
            "candidates": [f"{oss}/bin/yosys", "yosys"],
            "version_cmd": ["-V"],
            "required": True,
        },
        {
            "id": "sta_ieda",
            "category": "sta",
            "candidates": ["/home/yang/yosys-sta/bin/iEDA", "iEDA"],
            "version_cmd": ["-version"],
            "required": False,
        },
        {
            "id": "sta_opensta",
            "category": "sta",
            "candidates": ["sta", "OpenSTA", "opensta", f"{oss}/bin/sta"],
            "version_cmd": ["-version"],
            "required": False,
        },
        {
            "id": "rv32_gcc",
            "category": "rv32",
            "candidates": [
                "riscv64-linux-gnu-gcc",
                "riscv32-unknown-elf-gcc",
                "riscv64-unknown-elf-gcc",
                "riscv-none-embed-gcc",
            ],
            "version_cmd": ["--version"],
            "required": True,
        },
        {
            "id": "objdump_rv",
            "category": "rv32",
            "candidates": ["riscv64-linux-gnu-objdump"],
            "version_cmd": ["--version"],
            "required": False,
        },
        {
            "id": "gtkwave",
            "category": "waveform",
            "candidates": ["gtkwave"],
            "version_cmd": ["--version"],
            "required": False,
        },
    ]


def probe_tool(spec: dict) -> dict:
    chosen = None
    for cand in spec["candidates"]:
        path = shutil.which(cand) if os.sep not in cand else (cand if Path(cand).is_file() else None)
        if path:
            chosen = path
            break
    entry = {
        "id": spec["id"],
        "category": spec["category"],
        "required": spec["required"],
        "present": chosen is not None,
        "path": chosen,
        "version": None,
        "version_exit": None,
        "version_raw": None,
        "note": None,
    }
    if chosen is None:
        entry["note"] = "未找到可执行文件（PATH 与已知候选均无）"
        return entry
    res = run([chosen] + spec["version_cmd"])
    entry["version_exit"] = res["exit_code"]
    raw = first_line(res["stdout"]) or first_line(res["stderr"])
    entry["version_raw"] = raw or None
    entry["version"] = raw or None
    if not raw and res["exit_code"] == 0:
        entry["note"] = "命令退出码 0 但无版本字符串输出"
    elif res["error"]:
        entry["note"] = f"运行失败: {res['error']}"
    elif res["exit_code"] not in (0, None):
        entry["note"] = f"版本命令退出码 {res['exit_code']}"
    return entry


# --------------------------------------------------------------------------- #
# Python 运行时探测
# --------------------------------------------------------------------------- #
def probe_python_runtimes() -> list[dict]:
    cands = [
        ("system", sys.executable),
        ("conda_base", "/home/yang/miniconda3/bin/python3"),
        ("conda_eda", "/home/yang/miniconda3/envs/eda/bin/python3"),
        ("conda_rag", "/home/yang/miniconda3/envs/rag/bin/python3"),
        ("conda_qpanda", "/home/yang/miniconda3/envs/qpanda/bin/python3"),
    ]
    runtimes = []
    for name, exe in cands:
        real = shutil.which(exe) if os.sep not in exe else (exe if Path(exe).is_file() else None)
        entry = {"name": name, "path": real, "version": None, "modules": {}}
        if real is None:
            entry["note"] = "解释器不存在"
            runtimes.append(entry)
            continue
        res = run([real, "--version"])
        entry["version"] = first_line(res["stdout"]) or first_line(res["stderr"])
        for module in ("tensorflow", "torch", "numpy", "onnx", "onnxruntime"):
            entry["modules"][module] = probe_module(module, real)
        runtimes.append(entry)
    return runtimes


# --------------------------------------------------------------------------- #
# 官方资料哈希校验
# --------------------------------------------------------------------------- #
def verify_source_hashes(repo: Path) -> dict:
    manifest_path = repo / "资料" / "来源校验.json"
    result = {
        "manifest": str(manifest_path.relative_to(repo)),
        "hash_algorithm": "git blob sha1 (sha1 of 'blob <size>\\0' + content)",
        "source_commits": {},
        "checked": 0,
        "match": 0,
        "mismatch": 0,
        "missing": 0,
        "files": [],
    }
    if not manifest_path.is_file():
        result["error"] = "来源校验.json 不存在"
        return result
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result["source_commits"] = manifest.get("source_commits", {})
    for item in manifest.get("files", []):
        rel = item["path"]
        target = repo / "资料" / rel
        entry = {
            "path": rel,
            "expected_sha1": item.get("sha"),
            "expected_size": item.get("size"),
            "exists": target.is_file(),
            "actual_sha1": None,
            "actual_size": None,
            "status": None,
        }
        if target.is_file():
            entry["actual_size"] = target.stat().st_size
            entry["actual_sha1"] = git_blob_sha1(target)
            if entry["actual_sha1"] == entry["expected_sha1"]:
                entry["status"] = "match"
                result["match"] += 1
            else:
                entry["status"] = "mismatch"
                result["mismatch"] += 1
        else:
            entry["status"] = "missing"
            result["missing"] += 1
        result["checked"] += 1
        result["files"].append(entry)
    return result


# --------------------------------------------------------------------------- #
# 资产与接口约定检查
# --------------------------------------------------------------------------- #
def parse_flags(repo: Path) -> dict:
    flags_path = repo / "资料" / "ARM_KWT-1_237e4097" / "models_data_v2_12_labels" / "kwt1" / "flags.json"
    if not flags_path.is_file():
        return {"present": False, "path": str(flags_path.relative_to(repo))}
    flags = json.loads(flags_path.read_text(encoding="utf-8"))
    keep = {
        "model_name": flags.get("model_name"),
        "num_layers": flags.get("num_layers"),
        "d_model": flags.get("d_model"),
        "mlp_dim": flags.get("mlp_dim"),
        "heads": flags.get("heads"),
        "dropout1": flags.get("dropout1"),
        "attention_type": flags.get("attention_type"),
        "prenorm": flags.get("prenorm"),
        "label_count": flags.get("label_count"),
        "wanted_words": flags.get("wanted_words"),
        "sample_rate": flags.get("sample_rate"),
        "clip_duration_ms": flags.get("clip_duration_ms"),
        "window_size_ms": flags.get("window_size_ms"),
        "window_stride_ms": flags.get("window_stride_ms"),
        "spectrogram_length": flags.get("spectrogram_length"),
        "dct_num_features": flags.get("dct_num_features"),
        "feature_type": flags.get("feature_type"),
        "attention_type": flags.get("attention_type"),
        "train": flags.get("train"),
        "return_softmax": flags.get("return_softmax"),
    }
    return {"present": True, "path": str(flags_path.relative_to(repo)), "flags": keep}


def check_asset_tree(repo: Path) -> dict:
    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(repo))
        except ValueError:
            return str(p)

    e203_subset = repo / "资料" / "E203_Hummingbirdv2_9f3e34f"
    e203_full = repo / "资料" / "e203_hbirdv2_full"
    kwt1 = repo / "资料" / "ARM_KWT-1_237e4097" / "models_data_v2_12_labels" / "kwt1"
    weights_index = kwt1 / "best_weights.index"
    weights_data = kwt1 / "best_weights.data-00000-of-00001"
    labels = kwt1 / "labels.txt"
    npz = repo / "software" / "weights" / "kwt1_fp32.npz"
    dataset = repo / "资料" / "数据集" / "speech_commands_v0.02.tar.gz"

    # E203：in-repo 子集 + external-local 完整上游
    subset_core = e203_subset / "rtl" / "e203" / "core" / "e203_exu_nice.v"
    full_resolves = e203_full.exists()
    full_core = e203_full / "rtl" / "e203" / "core" / "e203_exu_nice.v"
    full_commit = None
    if full_core.is_file():
        res = run(["git", "-C", str(e203_full), "rev-parse", "HEAD"])
        full_commit = res["stdout"] or None

    # PDK / SRAM 宏：external-local（学术库），非官方工艺
    nangate_lib_dir = Path("/home/yang/yosys-sta/pdk/nangate45/lib")
    icsprout = Path("/home/yang/yosys-sta/pdk/icsprout55")
    fakeram = sorted(p.name for p in nangate_lib_dir.glob("fakeram*.lib")) if nangate_lib_dir.is_dir() else []

    # SDC
    sdc_in_repo = sorted(rel(p) for p in repo.rglob("*.sdc"))
    sdc_external = sorted(
        str(p) for p in [Path("/home/yang/yosys-sta/abc.sdc")] if p.is_file()
    )

    return {
        "model_source": {
            "path": rel(repo / "资料" / "ARM_KWT-1_237e4097"),
            "present": (repo / "资料" / "ARM_KWT-1_237e4097").is_dir(),
            "pinned_commit": "237e4097afd4f6851e0b82f0a8d65c2dfe8eae30",
            "scope": "in-repo",
            "note": "比赛锁定提交的源码快照（非 git 仓库），只读参考。",
        },
        "flags": parse_flags(repo),
        "weights": {
            "index": {"path": rel(weights_index), "present": weights_index.is_file()},
            "data": {"path": rel(weights_data), "present": weights_data.is_file()},
            "labels": {"path": rel(labels), "present": labels.is_file()},
            "software_npz": {
                "path": rel(npz),
                "present": npz.is_file(),
                "sha256": sha256_of(npz) if npz.is_file() else None,
                "note": "由 TF checkpoint 导出的全 162 张量 FP32 基线，不是官方评测输入。",
            },
            "scope": "in-repo",
        },
        "dataset": {
            "path": rel(dataset),
            "present": dataset.is_file(),
            "md5": md5_of(dataset) if dataset.is_file() else None,
            "expected_md5": "6b74f3901214cb2c2934e98196829835",
            "note": "Google Speech Commands v0.02；被 .gitignore 排除，需按资料/补全大文件.sh --dataset 下载。",
        },
        "e203": {
            "subset": {
                "path": rel(e203_subset),
                "present": subset_core.is_file(),
                "pinned_commit": "9f3e34f337f5d0a1236a285871ffa2032491617a",
                "scope": "in-repo",
                "note": "官方锁定的 Hummingbird E203 关键文件子集（含 nice_core），用于对齐接口。",
            },
            "full_upstream": {
                "path": rel(e203_full),
                "present": full_core.is_file(),
                "resolves": full_resolves,
                "detected_commit": full_commit,
                "scope": "external-local",
                "note": "本机独立管理的完整上游仓库（symlink 到 /home/yang/Desktop/ic设计竞赛/资料），不属于本 worktree。",
            },
            "official_soc_platform": {
                "present": False,
                "scope": "official-missing",
                "note": "委员会打包的 E203 SoC 平台（固定地址映射、共享 512KB SRAM、总线/NICE 接线与外设）不在仓库内。",
            },
        },
        "pdk": {
            "nangate45": {
                "path": str(nangate_lib_dir.parent),
                "present": nangate_lib_dir.is_dir(),
                "libs": sorted(p.name for p in nangate_lib_dir.glob("*.lib")) if nangate_lib_dir.is_dir() else [],
                "scope": "external-local",
                "note": "学术 Nangate45 库，仅作通用逻辑映射/粗综合代理，不是赛题指定工艺。",
            },
            "icsprout55": {
                "path": str(icsprout),
                "present": icsprout.is_dir(),
                "scope": "external-local",
                "note": "iEDA 自带 55nm 练习工艺，非赛题指定。",
            },
            "official": {"present": False, "scope": "official-missing", "note": "赛题指定 PDK 缺失。"},
        },
        "sram_macros": {
            "available_generic": {
                "present": bool(fakeram),
                "libs": fakeram,
                "path": str(nangate_lib_dir),
                "scope": "external-local",
                "note": "fakeram 通用宏（45nm 学术库），不等于题目指定的 16bit×128/1024/4096。",
            },
            "official_sizes": {
                "present": False,
                "required_sizes": ["16bit x 128", "16bit x 1024", "16bit x 4096"],
                "scope": "official-missing",
                "note": "题目指定的内部 SRAM 宏缺失，无法据此核算面积/端口/冲突。",
            },
            "in_repo_models": {
                "present": (repo / "hardware" / "rtl" / "sram").is_dir(),
                "files": sorted(rel(p) for p in (repo / "hardware" / "rtl" / "sram").glob("*"))
                if (repo / "hardware" / "rtl" / "sram").is_dir()
                else [],
                "scope": "in-repo",
                "note": "RTL 级行为缓冲模型，不是可综合宏。",
            },
        },
        "sdc": {
            "in_repo": sdc_in_repo,
            "in_repo_present": bool(sdc_in_repo),
            "external_examples": sdc_external,
            "official_150mhz": {
                "present": False,
                "scope": "official-missing",
                "note": "官方 150 MHz SDC 缺失；仓库内没有任何 .sdc。",
            },
        },
    }


def official_requirements_status(assets: dict) -> list[dict]:
    status_map = {
        "official_soc_platform": assets["e203"]["official_soc_platform"]["present"],
        "official_pdk": assets["pdk"]["official"]["present"],
        "official_sram_macros": assets["sram_macros"]["official_sizes"]["present"],
        "official_sdc": assets["sdc"]["official_150mhz"]["present"],
        "official_eval_set": (assets.get("dataset", {}).get("present", False))
        and False,  # 数据集 tar 也不是官方 600 条评测集
    }
    out = []
    for req in OFFICIAL_REQUIREMENTS:
        present = bool(status_map.get(req["id"], False))
        out.append(
            {
                "id": req["id"],
                "name": req["name"],
                "present": present,
                "status": "present" if present else "missing",
                "why_it_matters": req["why_it_matters"],
                "blocking_claim": None
                if present
                else "在缺少该官方资产时，任何依赖它的性能/时序/面积/精度结论都不能宣称已达到官方口径。",
            }
        )
    return out


# --------------------------------------------------------------------------- #
# 能力上限（可验证到哪一层）
# --------------------------------------------------------------------------- #
def capability_levels() -> list[dict]:
    return [
        {
            "id": "L1_kernel_functional",
            "name": "计算核/输出级功能仿真（Verilator/Icarus，逐位对照软件整数模型）",
            "status": "achievable",
            "evidence": "工具链 present 且 RTL/测试平台在本仓库内；smoke test 见 smoke_test.json。",
            "reason": "不依赖官方 SoC/PDK/SDC。",
        },
        {
            "id": "L2_module_synthesis_proxy",
            "name": "模块级逻辑综合（Yosys 通用库，或 iEDA + Nangate45 代理库）",
            "status": "achievable",
            "evidence": "Yosys 与 iEDA 可执行；Nangate45 lib 作为代理库存在。",
            "reason": "结果为代理工艺下的相对参考，不是赛题指定 PDK 的签核数据。",
        },
        {
            "id": "L3_e203_rtl_simulation",
            "name": "E203 RTL 仿真接入（Hummingbird E203 固定提交，含 NICE/总线仿真模型）",
            "status": "partial",
            "evidence": "E203 子集与完整上游 RTL 存在，可用 Icarus 跑 CPU 级仿真。",
            "reason": "缺少委员会官方 SoC 封装（地址映射、共享 SRAM、外设），不等于官方平台仿真。",
        },
        {
            "id": "L4_official_timing_signoff",
            "name": "150 MHz 时序签核",
            "status": "blocked",
            "evidence": "缺少官方 PDK、标准单元库、官方 SDC 与完整 SoC 网表。",
            "reason": "代理库/示例 SDC 下的频率数字不能当作 150 MHz 达标。",
        },
        {
            "id": "L5_area_power_signoff",
            "name": "≤2.25 mm² 面积与功耗/能耗签核",
            "status": "blocked",
            "evidence": "缺少指定 SRAM 宏与官方工艺；无 SAIF/VCD 统一条件。",
            "reason": "不能用通用 fakeram 面积或不同配置拼出面积/功耗结论。",
        },
        {
            "id": "L6_official_eval_accuracy",
            "name": "官方 600 条评测集上的 NRMSE / 准确率",
            "status": "blocked",
            "evidence": "官方评测集与评测脚本未发布；本地 240 条验证样本不是 600 条口径。",
            "reason": "480 软件 / 240 RTL / 官方 600 是三个不同统计，不可混用或外推。",
        },
    ]


def build_claims(assets: dict) -> dict:
    return {
        "can_claim": [
            "在 Verilator/Icarus 上的模块级与计算核级功能仿真结果，以及逐位对照软件整数模型的一致性。",
            "在 Yosys 通用库或 iEDA+Nangate45 代理库下的相对逻辑综合/面积量级，注明为代理工艺。",
            "官方锁定提交（KWT-1 237e4097 / E203 9f3e34f）源码快照与权重的哈希一致性。",
            "Hummingbird E203（固定上游）RTL 级 CPU 仿真中的指令流/接口行为。",
        ],
        "cannot_claim": [
            "在官方 E203 SoC 平台完成 SoC 集成仿真——官方 SoC 封装缺失。",
            "达到 150 MHz 时序——无官方 PDK/库/SDC，代理约束不成立。",
            "满足 ≤2.25 mm² 面积与功耗签核——无指定 SRAM 宏与官方工艺。",
            "官方 600 条评测集上的 NRMSE/准确率——评测集未发布。",
            "任何基于模拟模型/代理库的数字代表真实 SoC 或真实工艺结果。",
        ],
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]), help="仓库根目录")
    parser.add_argument("--outdir", default=str(Path(__file__).resolve().parent), help="输出目录")
    parser.add_argument("--json-only", action="store_true", help="只写 JSON，不写 Markdown")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    tools = [probe_tool(spec) for spec in candidate_tools(repo)]
    runtimes = probe_python_runtimes()
    hashes = verify_source_hashes(repo)
    assets = check_asset_tree(repo)
    official = official_requirements_status(assets)
    levels = capability_levels()
    claims = build_claims(assets)

    tool_present = sum(1 for t in tools if t["present"])
    tool_missing = [t["id"] for t in tools if not t["present"]]
    official_missing = [o["id"] for o in official if not o["present"]]

    summary = {
        "tools_present": tool_present,
        "tools_total": len(tools),
        "tools_missing": tool_missing,
        "hash_match": hashes["match"],
        "hash_mismatch": hashes["mismatch"],
        "hash_missing": hashes["missing"],
        "official_missing": official_missing,
        "capability_achievable": [l["id"] for l in levels if l["status"] == "achievable"],
        "capability_partial": [l["id"] for l in levels if l["status"] == "partial"],
        "capability_blocked": [l["id"] for l in levels if l["status"] == "blocked"],
        "verification_ceiling": (
            "在缺官方 SoC/PDK/SRAM 宏/SDC/评测集的前提下，本环境最高只能给出"
            " L1 功能仿真与 L2 代理工艺粗综合证据；不得据此宣称真实 SoC 或 150 MHz。"
        ),
    }

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "cwd": os.getcwd(),
        },
        "python_runtimes": runtimes,
        "tools": tools,
        "assets": assets,
        "hashes": hashes,
        "official_requirements": official,
        "capability_levels": levels,
        "claims": claims,
        "summary": summary,
    }

    json_path = outdir / "environment_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not args.json_only:
        md_path = outdir / "environment_report.md"
        md_path.write_text(render_markdown(report), encoding="utf-8")

    print(f"[env] json -> {json_path}")
    print(
        f"[env] tools {tool_present}/{len(tools)} present; "
        f"hash match={hashes['match']} mismatch={hashes['mismatch']} missing={hashes['missing']}; "
        f"official_missing={official_missing}"
    )
    return 0


def render_markdown(report: dict) -> str:
    r = report
    lines: list[str] = []
    add = lines.append
    add("# 研究生组环境检测报告（自动生成）")
    add("")
    add(f"- 生成时间（UTC）：{r['generated_at']}")
    add(f"- 仓库根：`{r['repo_root']}`")
    add(f"- 宿主：{r['host']['platform']} / {r['host']['machine']} / python {r['host']['python']}")
    add(f"- 结论上限：{r['summary']['verification_ceiling']}")
    add("")

    add("## 1. 软件工具链")
    add("")
    add("| 工具 | 类别 | 存在 | 路径 | 版本/输出 | 退出码 | 备注 |")
    add("| --- | --- | :-: | --- | --- | :-: | --- |")
    for t in r["tools"]:
        add(
            f"| `{t['id']}` | {t['category']} | {'是' if t['present'] else '否'} | "
            f"{('`'+t['path']+'`') if t['path'] else '—'} | "
            f"{(t['version'] or '—').replace('|','/')} | "
            f"{t['version_exit'] if t['version_exit'] is not None else '—'} | {t['note'] or ''} |"
        )
    add("")

    add("## 2. Python 运行时与模型库")
    add("")
    add("| 运行时 | 路径 | 版本 | tensorflow | torch | numpy | onnx |")
    add("| --- | --- | --- | :-: | :-: | :-: | :-: |")
    for rt in r["python_runtimes"]:
        mods = rt.get("modules", {})

        def cell(name: str) -> str:
            m = mods.get(name, {})
            if not m.get("found"):
                return "否"
            return m.get("version") or "有(版本未知)"

        add(
            f"| {rt['name']} | `{rt.get('path') or '—'}` | {rt.get('version') or '—'} | "
            f"{cell('tensorflow')} | {cell('torch')} | {cell('numpy')} | {cell('onnx')} |"
        )
    add("")

    add("## 3. 官方资料哈希（对照 `资料/来源校验.json`）")
    add("")
    h = r["hashes"]
    add(f"- 哈希算法：{h.get('hash_algorithm', 'git blob sha1')}（与 `资料/补全大文件.sh` 的 `git hash-object` 口径一致）")
    add(f"- 来源提交：{json.dumps(h['source_commits'], ensure_ascii=False)}")
    add(f"- 检查 {h['checked']} 个：match={h['match']}，mismatch={h['mismatch']}，missing={h['missing']}")
    add("")
    add("| 文件 | 期望 blob 哈希 | 实际 blob 哈希 | 状态 |")
    add("| --- | --- | --- | :-: |")
    for f in h["files"]:
        if f["status"] == "match":
            continue
        add(f"| `{f['path']}` | `{f['expected_sha1']}` | `{f['actual_sha1'] or '—'}` | **{f['status']}** |")
    if h["mismatch"] == 0 and h["missing"] == 0:
        add("| (全部一致) | | | ✅ |")
    add("")

    add("## 4. 真实资产与接口约定")
    add("")
    a = r["assets"]
    rows = [
        ("官方模型源码 (237e4097)", a["model_source"]["present"], a["model_source"]["scope"], a["model_source"]["note"]),
        ("KWT-1 best_weights", a["weights"]["data"]["present"], a["weights"]["scope"], "TF checkpoint 权重数据文件"),
        ("FP32 导出 npz", a["weights"]["software_npz"]["present"], "in-repo", a["weights"]["software_npz"]["note"]),
        ("Speech Commands v0.02", a["dataset"]["present"], "external-local/missing", a["dataset"]["note"]),
        ("E203 子集 (9f3e34f)", a["e203"]["subset"]["present"], a["e203"]["subset"]["scope"], a["e203"]["subset"]["note"]),
        ("E203 完整上游", a["e203"]["full_upstream"]["present"], a["e203"]["full_upstream"]["scope"], a["e203"]["full_upstream"]["note"]),
        ("官方 E203 SoC 平台", a["e203"]["official_soc_platform"]["present"], "official-missing", a["e203"]["official_soc_platform"]["note"]),
        ("Nangate45 PDK", a["pdk"]["nangate45"]["present"], a["pdk"]["nangate45"]["scope"], a["pdk"]["nangate45"]["note"]),
        ("官方 PDK", a["pdk"]["official"]["present"], "official-missing", a["pdk"]["official"]["note"]),
        ("fakeram 通用宏", a["sram_macros"]["available_generic"]["present"], "external-local", a["sram_macros"]["available_generic"]["note"]),
        ("指定 SRAM 宏 16bit×128/1024/4096", a["sram_macros"]["official_sizes"]["present"], "official-missing", a["sram_macros"]["official_sizes"]["note"]),
        ("官方 150MHz SDC", a["sdc"]["official_150mhz"]["present"], "official-missing", a["sdc"]["official_150mhz"]["note"]),
    ]
    add("| 资产 | 存在 | 来源 | 说明 |")
    add("| --- | :-: | --- | --- |")
    for name, present, scope, note in rows:
        add(f"| {name} | {'是' if present else '否'} | {scope} | {note} |")
    add("")

    add("## 5. 官方平台缺失项")
    add("")
    add("| 需求 | 存在 | 说明 |")
    add("| --- | :-: | --- |")
    for o in r["official_requirements"]:
        add(f"| {o['name']} | {'是' if o['present'] else '否'} | {o['why_it_matters']} |")
    add("")

    add("## 6. 可验证到哪一层")
    add("")
    add("| 层级 | 能力 | 状态 | 依据 |")
    add("| --- | --- | :-: | --- |")
    for l in r["capability_levels"]:
        add(f"| {l['id']} | {l['name']} | {l['status']} | {l['reason']} |")
    add("")

    add("## 7. 口径声明")
    add("")
    add("**可以宣称：**")
    for c in r["claims"]["can_claim"]:
        add(f"- {c}")
    add("")
    add("**不能宣称：**")
    for c in r["claims"]["cannot_claim"]:
        add(f"- {c}")
    add("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
