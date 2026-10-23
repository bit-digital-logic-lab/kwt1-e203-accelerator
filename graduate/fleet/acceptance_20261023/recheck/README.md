# 第二次主验收（2026-10-23）

## 已闭合：NRMSE门禁修复

从455f7f2的干净Git归档运行修复代码，独立显式RMS公式检查29例（1/2/64/257元素、零/微小/正常参考、不合法输入），再运行worker的41项unit和13项negative（含8个必须拒绝用例），全部退出0。原反例reference=zeros(2), estimate=1e-7正确返回NRMSE≈10并拒绝0.27%门槛。先前主测试脚本曾以浮点精确等于10判断，触发测试脚本自身断言；已改为1e-14相对容差，并独立重跑通过，未修改worker公式来迁就测试。

完整240×12真实FFN1数据指标仍由前次独立脚本确认最差0.1273378793%，本次只对门禁修复与异常边界做验证，不重复申报一次RTL72,990,720输出重跑。metric_acceptance.json、metric_units.json、metric_negative_tests.json记录本次实际命令与结果。修复分支codex/kwt-ds-r5-metric-floor已通过此局部主验收，未自动合并。

## 已复跑：R5 CACC分bank方案

源提交5812ec231a，干净归档，独立临时工作目录，运行完整run.sh并重新构建Verilator、全矩阵/端口断言/故障/综合代理。退出0，overall_pass=true、failures=[]。详细结果在banks_acceptance.json。

其可移植价值为同步SP/DP bank端口调度与bias-at-export，但源实现仍为16路列复用MAC。123,642周期不能作为目标8×8/8×16阵列性能；宏wrapper仿真和通用综合也不能证明指定宏面积/时序。

## 已复跑：独立软件包

005f656d8c干净归档，仅显式挂载数据集、p027源参数用于来源核对、已有缓存用于交叉比对及Python环境。冻结推理代码/参数包、FP32参考、权重和checkpoint来自归档。完整run_clean.sh退出0：440真实WAV+40合成零静音，FP32与整数模型均470/480，0类别翻转，TensorFlow12类分层复算实际执行；11个unit、10个negative和2个真实故障全部通过。完整结果software_results.json，执行信息software_acceptance.json。该结论仅是软件数值路径。

## 新8×16候选：预验与边界阻塞

复制时worker尚未提交，主智能体按中间results.json所列源码SHA256锁定RTL/TB，重新构建并用Python任意精度点积生成独立golden。结束时分支已提交为534fe5af931cc486e3bf724ae410a9705db9310a；主智能体逐文件核对Git blob，全部与独立重建快照哈希一致，见array8x16_commit_binding.json。该结论仍只覆盖声明的数值范围。

128bit总线、12套冻结FFN1参数的正常样本均逐项一致，观察到II=1、完整1,622,016MAC及13,312个K发射拍。主智能体已验证128bit路径12层、304,128个输出，周期均14,441，MAC有效利用率约87.75%，总线有效拍占约66.63%，见array8x16_partial.json。当前计时对应32bit导出模型子集；64bit路径也已完成12层校对，逐层均19,488周期、0失配；两种总线共608,256输出独立校对通过。128/64bit总线占用约66.63%/99.18%，说明64bit路径已接近带宽饱和。详见array8x16_preview.json。

独立权重翻位测试检出99处差异，比较器退出1。独立全范围用例A=32767/W=2047/bias=0/K=64：正确4292739136，DUT32bit导出为-2228160，25,344值全部不等，overflow_seen=1，比较器退出1。因此**拒绝把该候选作为全范围精确输出方案验收**。即使有溢出报警，也不能用截断输出替代正确原始结果。

已将此阻塞加入后续integration/crosscheck任务：按证明的至少34bit有符号范围选择36/40/48容器，保留所有位，重跑打包/字节数/周期及全范围golden；当前32bit模型子集性能与宽输出方案必须分别报告。

## 未交付任务的处理

R6 resident_w原worker退出0却无提交/results.json，所称后台run_all.py已不存活。已启动专门收尾服务kwt-ds-r6-resident-w-rescue-20261023.service，要求前台或持久服务跑完、写结果、提交后退出；仍待主验收。R6其余未完成/脏工作树结果不算通过。后续按queue_snapshot.json与实际分支HEAD判断，不能只看进程或临时PASS字段。
