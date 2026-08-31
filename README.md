# Mamba 选择性状态空间模型学习项目

本项目用纯 PyTorch 实现 Mamba v1 的核心数据流，并在 TinyStories 连续 token
流上提供一个极小自回归语言模型。目标是看清
`[B,T,D] -> selective scan -> [B,T,D]`，不是复现论文训练规模，也不把教学
循环的速度当作官方 CUDA kernel 性能。

## 当前实现

| 文件 | 作用 |
|---|---|
| `model.py` | Mamba mixer、RMSNorm、残差 block、极小语言模型 |
| `dataset.py` | uint16 memmap 数据加载、随机 batch、label 右移检查 |
| `shape_check.py` | 输出 `Δ/B/C` 等关键 shape，并检查残差与因果性 |
| `train.py` | AdamW、warmup/cosine、FP16、梯度累积、评估、checkpoint 与 TensorBoard |
| `reporting.py` | 统一输出 `| 项目 | 值 |` 表格 |

## 模型与论文参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `d_model` | 128 | token 表示维度 D；为本地实验缩小 |
| `n_layers` | 4 | 同构 Mamba block 数；为本地实验缩小 |
| `d_state` | 16 | 每个内部通道的 SSM 状态维度 N，对齐 Mamba v1 默认值 |
| `d_conv` | 4 | 深度因果卷积宽度，对齐官方默认值 |
| `expand` | 2 | 内部通道 `d_inner=expand*d_model`，对齐官方默认值 |
| `dt_rank` | `ceil(d_model/16)` | 输入依赖步长 Δ 的低秩维度 R |
| `dt_min/max` | 0.001 / 0.1 | 初始 Δ 的对数均匀范围 |
| `bias/conv_bias` | `False/True` | 对齐官方 Mamba v1 默认值 |

核心数据流：

```text
[B,T,D]
  -> in_proj，拆为 x / z: 各 [B,T,2D]
  -> depthwise causal Conv1d + SiLU
  -> x_proj，产生 Δ低秩 [B,T,R]、B/C [B,T,N]
  -> dt_proj + softplus，得到 Δ [B,T,2D]
  -> selective scan，状态 h_t [B,2D,N]
  -> SiLU(z) 门控 -> out_proj -> [B,T,D]
  -> residual
```

`Δ`、`B`、`C` 随当前输入变化，因此状态更新是 time-varying；模型可以按内容
控制信息进入、保留和输出。实现中时间维使用显式 Python 循环，便于对应递推式：

```text
h_t = exp(Δ_t A) * h_(t-1) + Δ_t B_t x_t
y_t = C_t h_t + D x_t
```

## 环境

已验证的本机环境：

| 项目 | 值 |
|---|---|
| Conda env | `mamba` |
| Python | 3.11.16 |
| PyTorch | 2.5.1+cu121 |
| CUDA 可用 | 是 |
| GPU | NVIDIA GeForce RTX 3070 Laptop GPU |

本实现不依赖 `mamba-ssm` 或 `causal-conv1d`。`requirements.txt` 只补充
`numpy` 和 `tensorboard`，不会覆盖远程镜像已有的 CUDA 版 PyTorch。

## 数据

数据层读取两个连续 uint16 token 文件：

```text
data/train.bin
data/val.bin
```

当前已用 `D:\holiday_learning\mini_GPT\data` 中的 TinyStories 数据验证：

| split | token 数 | token id 范围 |
|---|---:|---:|
| train | 224,512,862 | 0..50256 |
| val | 4,765,918 | 0..50256 |

Git 不跟踪数据。本次用于手工上传的目录保留 `data/train.bin`、`data/val.bin`
和已有的 Hugging Face `hf_cache`；训练直接读取两个 `.bin` 文件，`hf_cache`
用于后续重新处理数据。运行时也可用 `--data-dir` 指向远程主机上的其他数据
目录；采样得到的 `x/y` 均为 `[B,T]`，并满足
`y[:, :-1] == x[:, 1:]`。

## 远程主机运行

远程镜像先提供兼容的 CUDA 版 PyTorch，再在项目根目录安装其余依赖：

```bash
pip install -r requirements.txt
```

默认读取项目内 `data/`，checkpoint 写入 `checkpoints/`，TensorBoard 事件写入
`/root/tf-logs/<run-name>/`：

```bash
python train.py --run-name mamba-baseline
```

日志根目录和实验名都可以通过命令行修改：

```bash
python train.py \
  --data-dir /path/to/data \
  --tensorboard-dir /root/tf-logs \
  --run-name mamba-baseline
```

若平台没有自动启动 TensorBoard 服务，可在远程主机执行：

```bash
tensorboard --logdir /root/tf-logs --host 0.0.0.0 --port 6006
```

曲线包含训练 loss、学习率、评估集 train/val loss、perplexity、tokens/sec 和
峰值显存。未传 `--run-name` 时自动使用时间戳子目录，避免不同实验混写。

## 快速开始

以下命令在项目根目录、激活 `mamba` 环境后执行。

检查真实数据：

```powershell
python dataset.py --data-dir D:\holiday_learning\mini_GPT\data
```

检查 block 的关键 shape 和因果性：

```powershell
python shape_check.py
```

先运行一个不写 checkpoint 的两步 smoke test：

```powershell
python train.py `
  --data-dir D:\holiday_learning\mini_GPT\data `
  --max-steps 2 --warmup-steps 1 `
  --batch-size 2 --block-size 32 `
  --d-model 64 --n-layers 2 `
  --eval-interval 1 --eval-iters 1 --no-save
```

再运行默认小实验；checkpoint 写入已忽略的 `checkpoints/`：

```powershell
python train.py --data-dir D:\holiday_learning\mini_GPT\data
```

从 latest 恢复时，模型结构参数必须与 checkpoint 一致：

```powershell
python train.py `
  --data-dir D:\holiday_learning\mini_GPT\data `
  --resume checkpoints\latest.pt --max-steps 200
```

## 已验证边界

2026-08-28 已实际完成：

| 检查 | 结果 |
|---|---|
| Python 语法编译 | 通过 |
| 清理上传目录前的单元测试 | 9/9 通过（含 optimizer 分组与 checkpoint 恢复） |
| CPU 最小前向/反向 | 通过 |
| CUDA shape 与因果检查 | 通过 |
| 真实 TinyStories 数据读取 | 通过 |
| CUDA 两步训练 smoke | 通过，loss 有限、显存统计可输出 |

两步 smoke 只证明训练链路闭合，不能据此判断收敛、生成质量或 Mamba 相对
Transformer 的性能。纯 PyTorch 逐步扫描也不能用于论文吞吐量复现。

## 参考

- Mamba 论文：`Mamba: Linear-Time Sequence Modeling with Selective State Spaces`，
  arXiv:2312.00752
- 官方实现：`state-spaces/mamba` 的 `mamba_ssm/modules/mamba_simple.py`
- 数据与训练壳对照：`D:\holiday_learning\mini_GPT`
