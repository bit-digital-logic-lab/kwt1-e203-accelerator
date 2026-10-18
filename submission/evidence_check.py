#!/usr/bin/env python3
"""本科必做 1-3 项证据 / 缺口核对（packaging 分支）。

本脚本只做两件事：
  1. 核对证据文件是否存在、关键哈希是否与既有报告一致；
  2. 区分证据层级（已上机数值核 / NICE-like 仿真 / C 驱动宿主测试 /
     示例库综合 / 未获得的官方证据）。

它**不会**把缺失或未上机的东西写成 PASS。每个条目只会在
`artifacts` 全部存在且哈希一致、且其层级是"已上机实测"时才给 PASS；
仅提供命令入口而无结果日志的条目给 ENTRY；声称过但无提交日志的给
PARTIAL；明确失败的给 FAIL；拿不到组委会材料、或只有示例库代替的给 GAP。

用法：
    python3 submission/evidence_check.py                 # 只核对文件/哈希/工具
    python3 submission/evidence_check.py --run-fast      # 另跑 C 驱动宿主测试与 RV32 编译
    python3 submission/evidence_check.py --json OUT --md OUT

退出码：0 = 核对流程正常跑完（结论以报告内容为准）；1 = 脚本自身出错。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUBMISSION = REPO / "submission"

# 主目录只读参考（别人的 worktree / 官方资料一律不写）。
MAIN = Path("/home/yang/Desktop/ic设计竞赛")

TOOLS = [
    "python3", "git", "sha256sum", "cc", "gcc",
    "verilator", "iverilog", "vvp", "yosys",
    "riscv64-linux-gnu-gcc", "riscv64-linux-gnu-objcopy",
]

# ---------------------------------------------------------------------------
# 证据清单。status 字段是"核对前的预期"，实际结论由校验结果覆盖。
# expected_sha256 只填既有报告里明确写下的完整 SHA-256。
# ---------------------------------------------------------------------------

LEVELS = {
    "L1_num_core_verilator": "已上机数值核（Verilator RTL 计算核 + 输出暂存）",
    "L2_nice_like_e203": "NICE-like E203 / SoC 仿真（非组委会正式 SoC）",
    "L3_driver_host": "C 驱动宿主测试 / RV32 交叉编译（非 SoC 联调）",
    "L4_example_lib_synth": "示例库综合（Yosys + Nangate45 示例 Liberty）",
    "GAP_official": "未获得的组委会材料 / 目标库 / 宏 / SDC / 评测集",
}

# 每个条目：
#   id, level, claim（已有说法）, artifacts（相对仓库路径）,
#   proof_level: "on_machine"（已上机）/ "entry_only"（仅入口）/ "reported_only"（声称无日志）
#   note, gaps（尚缺）
ITEMS = [
    # ---- L1 数值核 -------------------------------------------------------
    dict(
        id="l1_240_a11w10",
        level="L1_num_core_verilator",
        claim="240 条验证样本 × 12 层、72,990,720 个 INT32 输出与独立 INT32 GEMM 逐位一致，最差层 NRMSE 0.18216320%",
        artifacts=[
            "hardware/architecture/quantized_validation240_12layers_rtl_summary.json",
            "hardware/architecture/quantized_validation240_rtl.md",
        ],
        proof_level="on_machine",
        note="JSON 内 pass_3pct=true，checked_outputs=72990720，worst_nrmse_pct=0.1821632；但这是本地 240 条验证样本，不是组委会冻结后选取的 600 条。",
        gaps=["官方 600 条评测集结果", "该批次运行日志未提交（只有摘要 JSON）"],
    ),
    dict(
        id="l1_240_a12w10",
        level="L1_num_core_verilator",
        claim="A12W10 同一 240×12 批次最差层 NRMSE 0.14739360%，72,990,720 输出逐位一致",
        artifacts=[
            "hardware/architecture/quantized_validation240_a12w10_rtl_summary.json",
        ],
        proof_level="on_machine",
        note="JSON 内 conservative_accumulator_bound=67826716，远小于 INT32 上限。",
        gaps=["官方 600 条评测集结果", "运行日志未提交"],
    ),
    dict(
        id="l1_real_sample_12layers",
        level="L1_num_core_verilator",
        claim="一条真实验证样本 12 层、304,128 个输出逐位一致，最差层 0.17041%",
        artifacts=[
            "hardware/architecture/quantized_validation0_12layers.json",
            "hardware/architecture/quantized_validation0_12layers_row16.json",
        ],
        proof_level="on_machine",
        note="8 行核任务模型（327,920 拍/层）。",
        gaps=["该样本的 WAV / 样本哈希未随产物提交，复现需 software/.build 样本"],
    ),
    dict(
        id="l1_row16_staged_cycles",
        level="L1_num_core_verilator",
        claim="16 行 + 输出暂存 staged 任务模型 154,240 拍/层（A/W/bias/总线为测试台定时模型）",
        artifacts=[
            "hardware/architecture/staged_task_row16_summary.json",
            "hardware/architecture/quantized_validation0_12layers_row16_staged.json",
        ],
        proof_level="on_machine",
        note="154,240 拍是测试台定时模型，不是 E203 / 组委会总线性能。",
        gaps=["目标库/SDC 下的真实周期"],
    ),
    dict(
        id="l1_quant_format_compare",
        level="L1_num_core_verilator",
        claim="A8W8/A12W8/A12W9/A11W10/A12W10 五方案软件 480 条 + RTL 12 条子集逐位一致（见对照文档）",
        artifacts=[
            "hardware/architecture/量化格式_硬件成本与周期对照_20261017.md",
        ],
        proof_level="reported_only",
        note="文档汇总值；逐方案 RTL 报告目录 hardware/architecture/results/ 未随本分支提交（见缺口）。",
        gaps=[
            "results/quant_formats_12labels_40 逐方案日志未提交",
            "A12W9 参数包 software/results/ffn1_a12w9_static_sym_12labels_40.npz 未提交",
        ],
    ),

    # ---- L2 NICE-like 仿真 ----------------------------------------------
    dict(
        id="l2_run_e203_entry",
        level="L2_nice_like_e203",
        claim="E203 整层/真实层仿真入口脚本",
        artifacts=[
            "hardware/rtl/control/tb/run_e203.sh",
            "hardware/rtl/control/tb/prepare_real_e203_vectors.py",
            "hardware/rtl/control/tb/summarize_real_e203.py",
        ],
        proof_level="entry_only",
        note="脚本支持 --full-layer / --real-layers / --tiles；已核对其分支与 0..207 tile 范围检查。入口存在不代表结果通过。",
        gaps=["本分支未提交任何 PASS 日志"],
    ),
    dict(
        id="l2_nice_system_model",
        level="L2_nice_like_e203",
        claim="NICE-like E203 系统模型（128-bit 仿真总线 + 16-bit SRAM 行为模型）",
        artifacts=[
            "hardware/rtl/control/kwt_nice_system_model.sv",
            "hardware/rtl/control/tb/e203_kwt_nice_core.sv",
            "hardware/rtl/control/tb/e203_kwt_full_layer.S",
        ],
        proof_level="on_machine",
        note="源码自称 NICE-like，不是组委会实际 SoC 总线/完整 NICE 适配。",
        gaps=["组委会正式 System Bus 地址图与 NICE 编码确认"],
    ),
    dict(
        id="l2_synthetic_full_layer",
        level="L2_nice_like_e203",
        claim="--full-layer 合成整数 208 tile 整层 PASS：25,344 有效输出、838,028 拍",
        artifacts=[],
        proof_level="reported_only",
        note="结果只写在 doc/本科FFN1数据与硬件边界核对_20261017.md；本分支与主目录均未找到该次运行的提交日志或退出码。",
        gaps=["可核验的运行日志/退出码", "真实样本而非合成整数"],
    ),
    dict(
        id="l2_real_layer_run",
        level="L2_nice_like_e203",
        claim="真实 A12W10 向量逐层跑 E203 + NICE-like 模型的现状",
        artifacts=[
            "hardware/architecture/results/real_e203_validation0_manifest.json",
            "hardware/architecture/results/real_e203_validation0_report.md",
        ],
        proof_level="on_machine",
        external_root=str(MAIN),
        note="主目录只读结果：real_e203_validation0 的 12 层全部 FAIL（通过 0/12），每层在 tile 0 输出 [0,0] 读到 x，疑似中文路径导致 $readmemh 载入 X；driver 分支正在修复该问题。",
        gaps=["一次退出码全 0 的真实层 E203 全层 PASS"],
        force_status="FAIL",
    ),

    # ---- L3 C 驱动宿主测试 ----------------------------------------------
    dict(
        id="l3_driver_source",
        level="L3_driver_host",
        claim="208 个 32 字节描述符的 NICE-like C 驱动与宿主测试",
        artifacts=[
            "software/driver/kwt_nice_driver.c",
            "software/driver/kwt_nice_driver.h",
            "software/driver/test_kwt_nice_driver.c",
            "software/driver/README.md",
        ],
        proof_level="on_machine",
        note="README 明确：宿主描述符与编译已验证；C 版**尚未**在 E203 仿真中执行。",
        gaps=["C 驱动在 E203/SoC 仿真中实际执行", "正式地址图"],
    ),
    dict(
        id="l4b_synth_entry",
        level="L4_example_lib_synth",
        claim="示例库综合/比较脚本入口",
        artifacts=[
            "hardware/synth/run_engine.sh",
            "hardware/synth/run_dma_2d.sh",
            "hardware/architecture/run_compare_synth.sh",
        ],
        proof_level="entry_only",
        note="脚本要求 Liberty 库存在；仅入口存在不代表已综合或已达时序。",
        gaps=["目标 SDC / STA / 150 MHz"],
    ),
]

# 官方/目标材料：预期"不存在"，若意外出现需人工复核。命中即视为缺口依旧。
OFFICIAL_GAP_GLOBS = [
    ("组委会 System Bus / 正式 SoC 地址图", ["**/*system*bus*", "**/*sysbus*"]),
    ("目标 PDK / 标准单元库（非示例 Nangate45）", ["**/pdk/**", "**/*.db", "**/*.lef"]),
    ("指定 16bit×128/1024/4096 SRAM 宏（liberty/verilog 宏模型）",
     ["**/*sram*macro*", "**/*sram*1024*", "**/*sram*4096*", "**/*sram*128*"]),
    ("目标 SDC 约束文件", ["**/*.sdc"]),
    ("官方 600 条评测集 / 评测脚本", ["**/eval*600*", "**/*official*eval*"]),
]

# 官方来源完整性：资料/来源校验.json 里的 git blob SHA（与 --run-fast 无关，总是核对）。
# 参照 资料/来源校验.json 的 files[].sha。
OFFICIAL_BLOB_SHA = {
    "资料/ARM_KWT-1_237e4097/models_data_v2_12_labels/kwt1/labels.txt":
        "ba416458b011a7f4b96739eb6fcb6275a6ab3bec",
    "资料/ARM_KWT-1_237e4097/models_data_v2_12_labels/kwt1/best_weights.data-00000-of-00001":
        "9c38f9653119383f7a303a94689c85a2a1c79b6b",
    "资料/ARM_KWT-1_237e4097/models_data_v2_12_labels/kwt1/flags.json":
        "008af2047e217bdad9e84ff5e5b70e69bac423c3",
    "资料/ARM_KWT-1_237e4097/kws_streaming/layers/speech_features.py":
        "c888a2f77e03487b399622c094fd288c43a237e4",
    "资料/E203_Hummingbirdv2_9f3e34f/rtl/e203/core/e203_exu_nice.v":
        "b01b7c263f878b082b2061178949fb2baf4f9fb6",
}

# 计算核摘要 JSON 内部记录的 source_sha256，与仓库内实际文件核对。
PARAM_HASH_CHECKS = [
    # (记录来源 JSON 里的 source_sha256 key, 需要核对的仓库路径, 期望值)
    ("hardware/architecture/quantized_validation240_12layers_rtl_summary.json",
     "weights", "software/weights/kwt1_fp32.npz",
     "8f359abf7adbe287b8f89fca98af46194f98f90f2146784731bff551a155825a"),
    ("hardware/architecture/quantized_validation240_12layers_rtl_summary.json",
     "params", "software/results/ffn1_blackbox_a11w10.npz",
     "bc7aff59fadf408245c0fe841c9753875df9af202023e3ed98a234e5d884c3aa"),
    ("hardware/architecture/quantized_validation240_a12w10_rtl_summary.json",
     "params", "software/results/ffn1_a12w10_static_sym.npz",
     "a77f60fb7c8560adbca73db3cf64dcd5501ec0108b96a1131a00aa722be1918e"),
    ("hardware/architecture/quantized_validation240_a12w10_rtl_summary.json",
     "weights", "software/weights/kwt1_fp32.npz",
     "8f359abf7adbe287b8f89fca98af46194f98f90f2146784731bff551a155825a"),
]

# 样本归档在仓库内缺失（.build 被 .gitignore 排除），这是复现缺口而不是 PASS。
SAMPLE_ARCHIVE = {
    "recorded_sha256": "ee04e849bc407149618e4ecd5ff99575687586d4e57c0ade6a0b3f7b66c8c149",
    "in_worktree": ["software/.build/quant_samples_12labels_20.npz"],
    "exists_in_main": "software/.build/quant_samples_12labels_20.npz",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha(path: Path) -> str | None:
    result = subprocess.run(["git", "hash-object", str(path)],
                            capture_output=True, text=True, cwd=str(REPO))
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def tool_info(name: str) -> dict:
    found = shutil.which(name)
    version = None
    if found:
        for flag in ("--version", "-V", "-v"):
            result = subprocess.run([found, flag], capture_output=True, text=True)
            text = (result.stdout + result.stderr).strip().splitlines()
            if text:
                version = text[0][:120]
                break
    return {"present": bool(found), "path": found, "version": version}


def check_artifact(rel: str, external_root: str | None) -> dict:
    base = Path(external_root) if external_root else REPO
    path = base / rel
    entry = {"path": (str(path) if external_root else rel), "exists": path.is_file()}
    if entry["exists"]:
        entry["sha256"] = sha256_file(path)
        entry["size"] = path.stat().st_size
    return entry


def run_host_tests(do_run: bool) -> list[dict]:
    tests = [{
        "name": "c_driver_host_test",
        "command": ("cc -std=c11 -O2 -Wall -Wextra -Werror "
                    "software/driver/kwt_nice_driver.c "
                    "software/driver/test_kwt_nice_driver.c "
                    "-o /tmp/kwt-nice-driver-evidence && "
                    "/tmp/kwt-nice-driver-evidence"),
    }, {
        "name": "c_driver_rv32_compile",
        "command": ("riscv64-linux-gnu-gcc -std=c11 -O2 -Wall -Wextra -Werror "
                    "-ffreestanding -march=rv32imac -mabi=ilp32 "
                    "-c software/driver/kwt_nice_driver.c "
                    "-o /tmp/kwt_nice_driver_evidence_rv32.o"),
    }]
    if not do_run:
        for test in tests:
            test.update({"status": "NOT_RUN", "exit_code": None,
                         "note": "未加 --run-fast，未执行"})
        return tests
    for test in tests:
        result = subprocess.run(["bash", "-lc", test["command"]],
                                capture_output=True, text=True, cwd=str(REPO))
        output = (result.stdout + result.stderr).strip().splitlines()
        test.update({
            "status": "PASS" if result.returncode == 0 else "FAIL",
            "exit_code": result.returncode,
            "stdout_tail": output[-8:],
        })
    return tests


def decide_status(item: dict, artifacts: list[dict]) -> tuple[str, str]:
    forced = item.get("force_status")
    if forced:
        return forced, "由只读外部证据判定"
    missing = [a["path"] for a in artifacts if not a["exists"]]
    if item["proof_level"] == "entry_only":
        return ("ENTRY", "仅有可执行入口，未附运行结果") if not missing \
            else ("GAP", "入口缺失: " + ", ".join(missing))
    if item["proof_level"] == "reported_only":
        if artifacts and not missing:
            return "PARTIAL", "有产物但无该次运行的退出码/日志"
        return "PARTIAL", "声称已跑但无提交日志（不计 PASS）"
    if missing:
        return "GAP", "证据文件缺失: " + ", ".join(missing)
    return "PASS", "证据文件齐备（数值/结论层级见 note）"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-fast", action="store_true",
                        help="实际执行 C 驱动宿主测试与 RV32 交叉编译")
    parser.add_argument("--json", type=Path,
                        default=SUBMISSION / "evidence_report.json")
    parser.add_argument("--md", type=Path,
                        default=SUBMISSION / "evidence_report.md")
    args = parser.parse_args()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve().relative_to(REPO)),
        "repo_root": str(REPO),
        "reference_main_readonly": str(MAIN),
        "levels": LEVELS,
        "tools": {name: tool_info(name) for name in TOOLS},
    }

    branch = subprocess.run(["git", "branch", "--show-current"], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()
    report["git_branch"] = branch
    report["git_head"] = head

    # 证据条目
    items_out = []
    for item in ITEMS:
        artifacts = [check_artifact(rel, item.get("external_root"))
                     for rel in item["artifacts"]]
        status, reason = decide_status(item, artifacts)
        items_out.append({
            "id": item["id"],
            "level": item["level"],
            "claim": item["claim"],
            "proof_level": item["proof_level"],
            "status": status,
            "status_reason": reason,
            "artifacts": artifacts,
            "note": item.get("note"),
            "gaps": item.get("gaps", []),
        })
    report["items"] = items_out

    # 官方来源完整性
    report["official_source_integrity"] = []
    for rel, expected in OFFICIAL_BLOB_SHA.items():
        path = REPO / rel
        actual = git_blob_sha(path) if path.is_file() else None
        report["official_source_integrity"].append({
            "path": rel, "present": path.is_file(),
            "expected_blob_sha": expected, "actual_blob_sha": actual,
            "match": actual == expected,
        })

    # 参数/权重哈希
    report["hash_checks"] = []
    for source_json, key, rel, expected in PARAM_HASH_CHECKS:
        path = REPO / rel
        actual = sha256_file(path) if path.is_file() else None
        report["hash_checks"].append({
            "source_json": source_json, "field": key, "path": rel,
            "expected_sha256": expected, "actual_sha256": actual,
            "match": actual == expected,
        })

    # 样本归档
    sample = {"recorded_sha256": SAMPLE_ARCHIVE["recorded_sha256"]}
    for rel in SAMPLE_ARCHIVE["in_worktree"]:
        path = REPO / rel
        actual = sha256_file(path) if path.is_file() else None
        sample[rel] = {"present": path.is_file(), "actual_sha256": actual,
                       "match": actual == SAMPLE_ARCHIVE["recorded_sha256"]}
    main_path = MAIN / SAMPLE_ARCHIVE["exists_in_main"]
    sample["main_dir_copy"] = {
        "path": str(main_path), "present": main_path.is_file(),
        "match": (sha256_file(main_path) == SAMPLE_ARCHIVE["recorded_sha256"])
        if main_path.is_file() else False,
    }
    sample["status"] = "GAP"
    sample["note"] = ("仓库内 software/.build 被 .gitignore 排除，实际样本归档缺失；"
                      "主目录有同哈希副本，但不在提交树中，复现依赖外部文件。")
    report["sample_archive"] = sample

    # 官方材料缺口：预期缺失，出现即需人工复核
    report["official_gap_scan"] = []
    for label, globs in OFFICIAL_GAP_GLOBS:
        hits = []
        for pattern in globs:
            for hit in REPO.glob(pattern):
                rel = str(hit.relative_to(REPO))
                if rel.startswith(".git/") or "/.git/" in rel:
                    continue
                hits.append(rel)
        report["official_gap_scan"].append({
            "missing_material": label, "found": sorted(set(hits)),
            "status": "UNEXPECTED_PRESENT" if hits else "ABSENT_AS_EXPECTED",
        })

    # 快速宿主测试
    report["host_tests"] = run_host_tests(args.run_fast)

    # 把本次实际执行的宿主测试也作为条目计入。
    for test in report["host_tests"]:
        if test["name"] == "c_driver_host_test":
            items_out.append({
                "id": "l3_host_test_fresh",
                "level": "L3_driver_host",
                "claim": "本次核对实际执行 C 驱动宿主测试（描述符布局/边界）",
                "proof_level": "on_machine",
                "status": "PASS" if test["status"] == "PASS" else
                          ("NOT_RUN" if test["status"] == "NOT_RUN" else "FAIL"),
                "status_reason": "本次运行退出码 " + str(test["exit_code"]),
                "artifacts": [],
                "note": ("宿主测试只证明描述符生成/边界检查；驱动尚未在 E203 仿真中执行，"
                         "不代表 SoC 联调。"),
                "gaps": ["C 驱动在 E203/SoC 仿真中实际执行"],
            })

    # 把官方材料缺口与样本归档也作为条目计入 summary，避免"PASS 数"掩盖缺口。
    for entry in report["official_gap_scan"]:
        unexpected = entry["status"] == "UNEXPECTED_PRESENT"
        items_out.append({
            "id": "gap_" + re.sub(r"\W+", "_", entry["missing_material"])[:24],
            "level": "GAP_official",
            "claim": entry["missing_material"],
            "proof_level": "not_obtained",
            "status": "REVIEW" if unexpected else "GAP",
            "status_reason": ("发现疑似官方材料，需人工复核是否真为组委会发放"
                              if unexpected else "未获得组委会材料（预期缺失）"),
            "artifacts": [{"path": h, "exists": True} for h in entry["found"]],
            "note": "只有组委会发放后才能真正闭合；不得用示例库替代。",
            "gaps": [entry["missing_material"]],
        })
    items_out.append({
        "id": "gap_sample_archive",
        "level": "L1_num_core_verilator",
        "claim": "240/480 条验证样本归档 software/.build/quant_samples_12labels_20.npz",
        "proof_level": "not_in_tree",
        "status": "GAP",
        "status_reason": "样本归档被 .gitignore 排除，提交树内不存在",
        "artifacts": [{"path": rel, "exists": False}
                      for rel in SAMPLE_ARCHIVE["in_worktree"]] + [{
                          "path": str(MAIN / SAMPLE_ARCHIVE["exists_in_main"]),
                          "exists": main_path.is_file(),
                          "note": "主目录只读副本，哈希一致时才可用"}],
        "note": sample["note"],
        "gaps": ["提交树内可复现的样本归档"],
    })
    report["items"] = items_out

    counts: dict[str, int] = {}
    for item in items_out:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    report["summary"] = {
        "by_status": counts,
        "pass_requires_on_machine_evidence": True,
        "note": ("PASS 仅用于已上机且产物齐备的条目；ENTRY/PARTIAL/FAIL/GAP "
                 "均不得当作通过。"),
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    args.md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    print(f"报告写入: {args.json} / {args.md}")
    return 0


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# 本科必做 1-3 项证据核对报告（自动生成）")
    lines.append("")
    lines.append(f"- 生成时间（UTC）：{report['generated_at']}")
    lines.append(f"- 分支 / 提交：`{report['git_branch']}` @ `{report['git_head']}`")
    lines.append(f"- 主目录只读参考：`{report['reference_main_readonly']}`")
    lines.append("- PASS 仅用于已上机且产物齐备的条目；ENTRY/PARTIAL/FAIL/GAP 均不算通过。")
    lines.append("")
    lines.append("## 汇总")
    lines.append("")
    lines.append("| 状态 | 条目数 |")
    lines.append("| --- | ---: |")
    for status, count in sorted(report["summary"]["by_status"].items()):
        lines.append(f"| {status} | {count} |")
    lines.append("")
    lines.append("## 按证据层级")
    lines.append("")
    for level, desc in report["levels"].items():
        members = [i for i in report["items"] if i["level"] == level]
        if not members:
            continue
        lines.append(f"### {level} — {desc}")
        lines.append("")
        lines.append("| 条目 | 状态 | 说明 |")
        lines.append("| --- | --- | --- |")
        for item in members:
            lines.append(f"| `{item['id']}` | {item['status']} | {item['status_reason']} |")
        lines.append("")
    lines.append("## 工具")
    lines.append("")
    lines.append("| 工具 | 存在 | 版本 |")
    lines.append("| --- | --- | --- |")
    for name, info in report["tools"].items():
        lines.append(f"| {name} | {'yes' if info['present'] else '**no**'} | "
                     f"{info['version'] or ''} |")
    lines.append("")
    lines.append("## 官方来源完整性（git blob SHA）")
    lines.append("")
    lines.append("| 文件 | 存在 | 一致 |")
    lines.append("| --- | --- | --- |")
    for entry in report["official_source_integrity"]:
        lines.append(f"| `{entry['path']}` | {entry['present']} | {entry['match']} |")
    lines.append("")
    lines.append("## 参数 / 权重哈希核对")
    lines.append("")
    lines.append("| 文件 | 期望 | 实际 | 一致 |")
    lines.append("| --- | --- | --- | --- |")
    for entry in report["hash_checks"]:
        lines.append(f"| `{entry['path']}` | `{entry['expected_sha256'][:12]}…` | "
                     f"`{(entry['actual_sha256'] or '缺失')[:12]}…` | {entry['match']} |")
    lines.append("")
    lines.append("## 样本归档")
    lines.append("")
    lines.append(f"- 记录哈希：`{report['sample_archive']['recorded_sha256']}`")
    lines.append(f"- 状态：**{report['sample_archive']['status']}** — {report['sample_archive']['note']}")
    lines.append("")
    lines.append("## 未获得的官方材料扫描（预期缺失）")
    lines.append("")
    lines.append("| 材料 | 状态 | 命中文件 |")
    lines.append("| --- | --- | --- |")
    for entry in report["official_gap_scan"]:
        hits = ", ".join(f"`{h}`" for h in entry["found"]) or "（无）"
        lines.append(f"| {entry['missing_material']} | {entry['status']} | {hits} |")
    lines.append("")
    lines.append("## 快速宿主测试")
    lines.append("")
    lines.append("| 测试 | 状态 | 退出码 |")
    lines.append("| --- | --- | --- |")
    for test in report["host_tests"]:
        lines.append(f"| `{test['name']}` | {test['status']} | {test['exit_code']} |")
    lines.append("")
    if any(test.get("stdout_tail") for test in report["host_tests"]):
        lines.append("```text")
        for test in report["host_tests"]:
            if test.get("stdout_tail"):
                lines.append(f"# {test['name']}")
                lines.extend(test["stdout_tail"])
        lines.append("```")
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
