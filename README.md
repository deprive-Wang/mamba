# Mamba 选择性状态空间模型学习项目

本项目用纯 PyTorch 实现 Mamba v1 的核心数据流，并在 TinyStories 连续 token
流上提供一个极小自回归语言模型。目标是看清
`[B,T,D] -> selective scan -> [B,T,D]`，不是复现论文训练规模，也不把教学
循环的速度当作官方 CUDA kernel 性能。

## 当前实现

| 文件 | 作用 |
|---|---|
| `model.py` | Mamba mixer、RMSNorm、残差 block、极小语言模型 |
| `data_splits.py` | 加载物理隔离的 train/validation/test，训练接口不返回 test |
| `dataset.py` | uint16 memmap 随机 batch、label 右移与三段数据检查 |
| `shape_check.py` | 输出 `Δ/B/C` 等关键 shape，并检查残差与因果性 |
| `train.py` | AdamW、warmup/cosine、FP16、梯度累积、评估、checkpoint 与 TensorBoard |
| `evaluate.py` | 加载 `best.pt`，只在物理 `test.bin` 上报告最终指标 |
| `reporting.py` | 统一输出 `| 项目 | 值 |` 表格 |
| `tests/` | 数据隔离和 checkpoint 最终评估的行为测试 |

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

数据层读取三个连续 uint16 token 文件：

```text
data/train.bin
data/val.bin
data/test.bin
```

`train.bin` 全部用于训练，`val.bin` 用于选择最佳 checkpoint，`test.bin` 只由
`evaluate.py` 读取。训练进程不会加载 `test.bin`，因此最终测试不参与训练或
checkpoint 选择。

为便于上传，已直接将本地 `data/val.bin` 固定为原始验证流的前 80%，并把后 20%
写入同目录的 `data/test.bin`。上传 `data/` 内的三个 `.bin` 文件到远程同名目录
即可。

| split | token 数 | 说明 |
|---|---:|---:|
| train | 224,512,862 | 训练 |
| validation | 3,812,734 | 选择最佳 checkpoint |
| test | 953,184 | 最终一次评估 |

Git 不跟踪数据。训练运行只需要 `data/train.bin`、`data/val.bin`、`data/test.bin`；
本地及远程均使用各自的 `data/` 目录。采样得到的 `x/y` 均为 `[B,T]`，并满足
`y[:, :-1] == x[:, 1:]`。

## 远程主机运行

远程镜像先提供兼容的 CUDA 版 PyTorch，再在项目根目录安装其余依赖：

```bash
pip install -r requirements.txt
```

AutoDL 当前 shell 若把 `OMP_NUM_THREADS` 设为无效的 `0`，在每个新终端或
screen 会话中临时覆盖：

```bash
export OMP_NUM_THREADS=1
```

先运行测试、数据检查和 shape 检查：

```bash
python -m unittest discover -s tests -v
python dataset.py --data-dir "$PWD/data"
python shape_check.py --device cuda
```

正式 baseline 从头训练 1000 个 optimizer step：

```bash
python train.py \
  --data-dir "$PWD/data" \
  --output-dir "$PWD/checkpoints" \
  --tensorboard-dir /root/tf-logs \
  --run-name mamba-formal-1000 \
  --max-steps 1000 \
  --warmup-steps 100 \
  --batch-size 4 \
  --block-size 128 \
  --grad-accum 1 \
  --d-model 128 \
  --n-layers 4 \
  --d-state 16 \
  --d-conv 4 \
  --expand 2 \
  --eval-interval 100 \
  --eval-iters 10 \
  --log-interval 10 \
  --checkpoint-interval 100 \
  --device cuda
```

训练结束后，只对 `best.pt` 做一次最终 test 评估：

```bash
python evaluate.py \
  --checkpoint "$PWD/checkpoints/best.pt" \
  --data-dir "$PWD/data" \
  --batch-size 4 \
  --eval-batches 100 \
  --seed 42 \
  --tensorboard-dir /root/tf-logs \
  --run-name mamba-formal-1000 \
  --device cuda
```

若平台没有自动启动 TensorBoard 服务，可在远程主机执行：

```bash
tensorboard --logdir /root/tf-logs --host 0.0.0.0 --port 6006
```

训练曲线包含 loss、学习率、train/validation loss、perplexity、tokens/sec 和
峰值显存。`evaluate.py` 使用与训练相同的 `--tensorboard-dir`、`--run-name` 时，
会把 test loss、test perplexity、评估吞吐和峰值显存追加到同一 run；未传
`--run-name` 时会写入 `evaluation-best` 子目录。脚本在本地默认写入项目根目录
的 `tf-logs/`，远程训练命令通过 `--tensorboard-dir /root/tf-logs` 覆盖该默认值。

## 快速开始

以下命令在项目根目录、激活可用的 PyTorch 环境后执行。

检查真实数据：

```powershell
python dataset.py --data-dir data
```

检查 block 的关键 shape 和因果性：

```powershell
python shape_check.py
```

先运行一个不写 checkpoint 的两步 smoke test：

```powershell
python train.py `
  --data-dir data `
  --max-steps 2 --warmup-steps 1 `
  --batch-size 2 --block-size 32 `
  --d-model 64 --n-layers 2 `
  --eval-interval 1 --eval-iters 1 --no-save
```

再运行默认 1000-step baseline；checkpoint 写入已忽略的 `checkpoints/`：

```powershell
python train.py --data-dir data --run-name mamba-formal-1000
```

从 latest 恢复时，模型结构参数必须与 checkpoint 一致：

```powershell
python train.py `
  --data-dir data `
  --resume checkpoints\latest.pt --max-steps 2000
```

训练完成后的最终测试：

```powershell
python evaluate.py --checkpoint checkpoints\best.pt --data-dir data
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

2026-09-01 本次补充：

| 检查 | 结果 |
|---|---|
| 物理 train/val/test 数据隔离 | 本地行为测试通过 |
| 训练接口不读取 test.bin | 本地行为测试通过 |
| `evaluate.py` CPU 端到端测试 | 本地 `mamba` 环境通过 |
| CUDA 两步 train→best.pt→test 闭环 | 本地 `mamba` 环境通过；仅为 smoke |
| `evaluate.py` CUDA 最终评估 | 本地 RTX 3070：test loss 4.3439，PPL 77.01（100 个固定随机 batch） |

两步 smoke 只证明训练链路闭合，不能据此判断收敛、生成质量或 Mamba 相对
Transformer 的性能。纯 PyTorch 逐步扫描也不能用于论文吞吐量复现。

## 参考

- Mamba 论文：`Mamba: Linear-Time Sequence Modeling with Selective State Spaces`，
  arXiv:2312.00752
- 官方实现：`state-spaces/mamba` 的 `mamba_ssm/modules/mamba_simple.py`
- 数据与训练壳对照：`D:\holiday_learning\mini_GPT`
