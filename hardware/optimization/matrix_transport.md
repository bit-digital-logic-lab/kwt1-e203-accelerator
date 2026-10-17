# FFN1 行主序矩阵的二维搬运契约

输入 `A[99,64]`、权重 `B[64,256]` 与输出 `C[99,256]` 若在共享 SRAM
按普通行主序存放，当前仅支持连续字的 `kwt_dma` 不能把一个非连续列块
当作单条连续 DMA。`matrix_stride_experiment.py` 对首块 `(0,0)` 和尾块
`(96,240)` 做了真实坐标寻址、INT16 点积和 INT32 有效输出写回的独立逐项
核对；它只计算字数与 128-bit 拍数，**不**模拟总线握手或速度。

建议的二维描述符是 `src_base, dst_base, rows, row_words,
src_stride_words, dst_stride_words`，每行内部按连续 burst，行间按两个
stride 跳转。[NVDLA 官方单元说明](https://nvdla.org/hw/v1/ias/unit_description.html#bridge-dma)
也用 line repeat 与行间地址跳转搬运二维数据；这里借鉴的是地址生成方式，
不是其总线宽度或实现。用 8×16 tile 表示：

| 矩阵 | 行数 | 每行 16-bit 字数 | 共享侧 stride | 本地侧 stride | 完整 tile 的 128-bit 拍数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A 读取 | 8 | 64 | 64 | 64 | 64 |
| B 列块读取 | 64 | 16 | 256 | 16 | 128 |
| C 写回 | 8 | 32（16 个 INT32） | 512 | 32 | 32 |

尾块只需加载 A 的 3 个有效行、写回 C 的 3 个有效行；小实验核对了
48 个尾块输出，A=192 字/24 拍、B=1024 字/128 拍、C=96 字/12 拍。
一个二维描述符可以让 DMA 内部迭代这些行 burst，避免 CPU 为 B 的
64 个 K 行重复提交 64 条命令。当前模型中 B 采用 `B[k][c]` 本地顺序，
自然适合一字/拍沿列流向行广播核；A 可保持行主序，在每个 K 开始前
逐行装入行寄存器。现有脉冲阵列所需的 `A[k][r]` 排列则需要本地转置
或预打包，不能把其成本藏在 DMA 之外。

若权重文件允许在装载前预排成列 tile 顺序，B 的共享侧 64 个行 burst
可合成连续块，但预排的存储占用/软件处理应计入数据准备。若下一层可以
直接消费 tiled C，输出也可能少一次行主序重排；这需要模型整体数据布局
约定，当前不假定成立。真实 System Bus 协议和 SRAM 宏端口未提供，
因此这里只定义地址映射与有效范围，暂不实现平台总线 RTL。

```bash
python3 hardware/optimization/run_hook.py matrix_stride --timeout 20 -- \
  python3 hardware/optimization/matrix_stride_experiment.py
```
