#!/usr/bin/env python3
"""Queue a second bounded DeepSeek round after the first fleet has capacity."""
import json
import subprocess
import sys
import time
from pathlib import Path

import dispatch as core

FIRST = Path('/home/yang/kwt-worktrees/ds-fleet-20261019/summary.json')
core.STATE = core.ROOT / 'ds-fleet-round2-20261019'
core.BASE = 'codex/kwt-grad-encoder'
core.TASKS = [
    ('r2_scale_contract', '交叉核查上一轮 attn_scale/postnorm_scale 的训练样本、sx、量化边界和 RTL 常数是否能合成一套冻结配置；读第一轮 worktree 只读；只写 graduate/ds_tasks/r2_scale_contract。'),
    ('r2_attn_240', '用已有训练缓存尽可能扩展注意力尺度标定到至少 120 条训练样本，并用独立验证样本评估；对比第一轮 3 条训练尺度，不用验证求尺度；只写 graduate/ds_tasks/r2_attn_240。'),
    ('r2_preprocess', '从 WAV/MFCC 到 99x64 encoder 输入做独立逐元素校对，核对 98 到 99 的补位与文件序列化，对至少 6 条真实样本；只写 graduate/ds_tasks/r2_preprocess。'),
    ('r2_pack_format', '为 12 层参数与 99x64 激活制定可运行的字节打包/解包和 roundtrip 校验，列带宽、端序、对齐；只写 graduate/ds_tasks/r2_pack_format。'),
    ('r2_ffn2_quant', '训练划分冻结 FFN2 的权重/激活尺度，在验证划分测整数 MAC、偏置、溢出和 NRMSE，至少两种精度；只写 graduate/ds_tasks/r2_ffn2_quant。'),
    ('r2_ln_quant', '研究两处 LayerNorm 的定点/查表实现，对验证样本和近零方差边界实测误差与饱和，明确硬件成本；只写 graduate/ds_tasks/r2_ln_quant。'),
    ('r2_qkv_quant', 'QKV 和注意力输出投影分别做训练标定、验证误差/累加范围测试，对比现有冻结格式；只写 graduate/ds_tasks/r2_qkv_quant。'),
    ('r2_softmax', '单头 99x99 Softmax 做两种有界硬件近似，测试行和、概率误差和后续输出误差，记录 ROM/算术开销；只写 graduate/ds_tasks/r2_softmax。'),
    ('r2_gelu_rom', '依据实际训练/验证动态范围设计可综合 GELU ROM/分段方案，测整域最大误差、验证误差和存储比特；只写 graduate/ds_tasks/r2_gelu_rom。'),
    ('r2_layer_switch', '为 12 层不同参数切换建立故障注入测试，证明重复 layer0 参数会失败；能运行则跑 RTL，否则给可运行软件门禁；只写 graduate/ds_tasks/r2_layer_switch。'),
    ('r2_full_numeric', '尝试用第一轮冻结资产构建 12 层整链的多样本数值检验；若配置冲突，实际运行到最早阻塞点并输出可复现失败证据；只写 graduate/ds_tasks/r2_full_numeric。'),
    ('r2_head_accuracy', '测试 FP32 十二层到软件分类头的真实验证样本标签/准确率，若可用则对整数边界加同样测试；不将小样本称官方精度；只写 graduate/ds_tasks/r2_head_accuracy。'),
    ('r2_dma_contract', '把 PC 前处理、RCL 加速器、软件分类头的控制/数据边界落成事务协议和可运行打包 mock，核对传输量；只写 graduate/ds_tasks/r2_dma_contract。'),
    ('r2_e203_driver', '基于实际现有 E203/NICE 信号写最小驱动/事务模拟和超时负例；没有官方 SoC 不宣称集成；只写 graduate/ds_tasks/r2_e203_driver。'),
    ('r2_sram_banking', '为 99x99 注意力和 FFN1/2 张量出具体 SRAM bank、地址、双缓冲排程，用脚本验无地址重叠/端口冲突；只写 graduate/ds_tasks/r2_sram_banking。'),
    ('r2_cycle_bottleneck', '从真实 RTL 日志区分计算、搬运、参数加载、完整 12 层周期，提出一项可验证加速改动并测模拟变化；只写 graduate/ds_tasks/r2_cycle_bottleneck。'),
    ('r2_synth_repro', '在本机开源工具复跑研究生 RTL lint/综合，把单模块与整链、频率假设、不可证明的 PDK 结论分开；只写 graduate/ds_tasks/r2_synth_repro。'),
    ('r2_activity', '从实际仿真 VCD 或事件计数统计主要 RTL 模块切换活动，记录生成命令和文件大小，不冒充功耗；只写 graduate/ds_tasks/r2_activity。'),
    ('r2_gate_audit', '独立复跑上一轮门禁与数据哈希修正，注入坏尺度/坏输出/错误分区，验证退出码能阻止假通过；读第一轮 worktree 只读；只写 graduate/ds_tasks/r2_gate_audit。'),
    ('r2_final_audit', '在第二轮前19项结束后交叉审核每项提交、退出码、样本数和真实证据层级，至少复跑两项；标出冲突与待集成，不合并也不 push；只写 graduate/ds_tasks/r2_final_audit。'),
]


def checkin(state, note):
    subprocess.run([sys.executable, str(core.CHECK), 'update', '--id',
                    'kwt-grad-ds-fleet-round2', '--state', state,
                    '--service', 'kwt-grad-ds-fleet-round2.service',
                    '--result', str(core.STATE / 'summary.json'), '--note', note],
                   cwd=core.PROJECT, check=False, capture_output=True, text=True)


core.checkin = checkin


def main():
    core.STATE.mkdir(parents=True, exist_ok=True)
    core.write_json(core.STATE / 'tasks.json', [
        {'task': task_id, 'scope': scope} for task_id, scope in core.TASKS
    ])
    while True:
        try:
            first = json.loads(FIRST.read_text())
            finished = first.get('finished', 0)
            if first.get('state') == 'complete' or finished >= 20:
                break
        except (OSError, ValueError):
            finished = 0
        core.write_json(core.STATE / 'summary.json', {
            'state': 'queued', 'model': core.MODEL, 'task_count': len(core.TASKS),
            'first_fleet_finished': finished, 'start_condition': 'first_finished>=20',
        })
        checkin('running', f'queued; first fleet {finished}/25 finished')
        time.sleep(60)
    sys.argv = [sys.argv[0], '--parallel', '2']
    return core.main()


if __name__ == '__main__':
    raise SystemExit(main())
