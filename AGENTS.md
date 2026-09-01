# Mamba 选择性状态空间模型学习项目

> 维护入口：本文件。后续确定环境、实现路线、实验配置或完成进度时，优先更新这里；不重复维护 `codex.md`。
> 对应计划：`D:\holiday_learning\暑期计划.md` 第九节「ViT 之后：Mamba 优先；AST、DeiT 与 Swin 按需」。

## 项目定位

本项目是 ViT + ResNet-50 之后的架构学习阶段，预计用时 3-5 天。目标不是从头预训练大规模语言模型，也不是复现 Mamba 论文的规模与指标，而是：

1. 理解状态空间模型的基本输入、状态和输出关系。
2. 理解 Mamba 中“选择性”的含义：`B`、`C`、`Δ` 等参数随输入变化。
3. 跟通一个最小 Mamba block，追踪输入 `[B, T, D]`、中间状态和输出 `[B, T, D]`。
4. 用一个极小语言模型或序列分类任务完成小实验，并与 Transformer 做有限、公平的对照。

项目重点是回答“特征怎么提”：Transformer 通过 token-token attention 聚合上下文；Mamba 通过输入依赖的选择性状态更新，让历史信息沿序列递推并选择性保留或遗忘。

## 当前进度

记录日期：2026-09-01

- [x] 创建项目目录 `D:\holiday_learning\mamba`
- [x] 建立项目级 `AGENTS.md`、`.gitignore`、`requirements.txt` 和 `README.md`
- [ ] 完成 ViT 项目收尾后正式切换到 Mamba
- [ ] 阅读 Mamba 论文摘要、图 1、§3 和 §4.2
- [x] 确定独立 Python 环境、执行平台和兼容的 PyTorch/CUDA 组合
- [x] 完成可信纯 PyTorch 教学实现的前向 shape 检查
- [ ] 画出 Transformer block 与 Mamba block 的数据流对照
- [x] 决定进入复用 TinyStories token 流的极小语言模型实验
- [x] 接入 TensorBoard 标量日志，默认根目录为 `/root/tf-logs`
- [x] 将本地 `data/val.bin` 固定划分为 80% validation，并在同目录生成 `test.bin`
- [x] 添加只读取 `best.pt` 与物理 `test.bin` 的独立 `evaluate.py`
- [x] 在本地 `mamba` 环境通过 3 项测试及两步 CUDA train→test 闭环
- [x] 记录实际训练与最终 test 的 loss、perplexity、速度、峰值显存和 TensorBoard 曲线
- [ ] 生成文本样例
- [ ] 在远程运行完整测试、1000-step baseline 与最终 test 评估

当前已使用 `mamba` 环境（Python 3.11.16、PyTorch 2.5.1+cu121）完成纯
PyTorch 教学版 Mamba、数据加载、shape/因果检查与两步 CUDA 训练 smoke。
该 smoke 只验证链路，不构成收敛、生成质量或性能结论。正式 1000-step 小实验已在
AutoDL RTX 3090 完成，最佳 `best.pt` 位于 step 900，validation loss 为 4.341955；
本地 RTX 3070 使用同一 checkpoint 与物理 `test.bin` 完成最终评估，test loss 为
4.3439、perplexity 为 77.01（100 个固定随机 batch）。
训练脚本已记录 loss、perplexity、学习率、tokens/sec 和峰值显存到
TensorBoard；`evaluate.py` 会将最终 test 指标追加到指定 run。日志根目录可通过
`--tensorboard-dir` 修改。
训练 checkpoint 不记录或读取 test 数据；最终评估只读取物理 `test.bin`，该文件
不参与训练和 checkpoint 选择。

## 固定学习顺序

第一轮严格按以下顺序推进：

```text
1. 状态空间模型直觉：输入 x_t、状态 h_t、输出 y_t
2. 离散化与递推：先理解数据流，不先陷入完整数学推导
3. Mamba 的选择机制：B、C、Δ 如何依赖输入
4. Mamba block：投影、局部卷积、选择性扫描、门控、输出投影、残差
5. 最小前向：验证 [B,T,D] -> [B,T,D]
6. 小实验：仅在前五步清楚后启动
7. 与 Transformer 对照：机制、复杂度、速度、显存与适用边界
```

不要把 Mamba 简化成“没有 attention 的 Transformer”。它是以选择性状态更新建模序列的状态空间架构。理论上的线性序列复杂度也不等于它在小数据、短序列或任意硬件上一定更快、更准。

## 第一阶段验收标准

1. 能画出并解释 Transformer attention 与 Mamba 状态递推的数据流差异。
2. 能用自己的话解释选择性：参数随当前输入变化，模型可以控制哪些信息进入、保留和输出。
3. 能跟踪最小 Mamba block 的关键 tensor shape，并验证输入、输出均为 `[B, T, D]`。
4. 能指出残差连接两侧的 shape 约束。
5. 若运行小实验，记录固定配置、实际命令、训练曲线、tokens/sec、峰值显存和至少一组输出或预测。
6. 能说明 Mamba 对长序列的潜在优势，以及小规模实验不能支持哪些泛化结论。

## 环境与依赖原则

- 为本项目建立独立环境，暂定环境名 `mamba`；创建前先确认最终平台和 Python 版本。
- 不向 Anaconda `base`、现有 `ML`、`Mini_GPT`、`lstm_audio`、`vit` 或 Codex 内置 Python 安装本项目依赖。
- 安装前核验官方实现当前支持的平台、Python、PyTorch、CUDA、编译器和 GPU 架构；不要猜版本组合。
- CUDA 版 `torch` 由本机或云镜像按兼容组合提供，不通过通用 `requirements.txt` 强行覆盖。
- `mamba-ssm`、`causal-conv1d` 等带编译或 CUDA 要求的包，待环境核验后再决定版本与安装命令。
- 执行下载或 `pip install` 前，先确认 VPN/代理、网络路径和下载源；不持久修改全局源配置。
- 若官方 CUDA 路径不适合本地 Windows，优先在 AutoDL Linux GPU 环境运行官方最小前向；本地纯 PyTorch 教学实现只能用于理解，不能冒充官方性能结果。

## 实验边界

- 第一优先级是结构理解和最小前向，不直接启动长时间训练。
- 当前任务固定为复用 Mini-GPT TinyStories token 流的极小语言模型实验。
- 正式 baseline 为 1000 step、batch size 4、block size 128、`d_model=128`、4 层。
- 本地与远程 `data/val.bin` 是原验证流前 80%，同目录 `data/test.bin` 是后 20%。
- Transformer 对照必须使用相同输入长度、数据划分、训练预算和评价指标；无法控制的实现差异要明确写出。
- 所有准确率、loss、速度、显存和耗时必须来自实际运行；不得填写预期值冒充结果。
- 数据、权重、缓存、日志和可再生实验产物不提交版本控制。

## 建议目录

以下是按需创建的目标结构，不要一次性生成空壳代码：

```text
mamba/
  AGENTS.md             项目约定、环境、进度与验收标准
  README.md             对外项目说明和最终实验结论
  .gitignore
  requirements.txt      numpy 与 tensorboard；不覆盖 CUDA 版 torch
  notes/                论文与结构笔记；开始阅读时再创建
  model.py              最小 Mamba block 或小模型；方案确定后再创建
  shape_check.py        [B,T,D] 前向与关键 shape 检查
  train.py              只有选择训练实验后才创建
  evaluate.py           best checkpoint 的独立 test loss/perplexity 评估
  data/                 已忽略；数据与缓存
  checkpoints/          已忽略；模型权重
  experiments/          已忽略；日志、曲线和临时实验产物
```

## 代码与记录约定

- Python 使用 4 空格缩进、PEP 8、`pathlib.Path` 和必要的类型注解。
- 代码注释解释状态、离散化、选择性参数、扫描、门控、残差和 tensor shape 等非显然约束，不复述代码。
- 外部输入、文件 IO、checkpoint 和配置读取需要显式校验；内部可信调用链避免重复防御。
- 修改模型数据流或实验协议时，同步更新本文件和 `README.md` 中对应的稳定事实。
- README 只记录实际采用的环境、命令和结果；尚未验证的内容明确标为“待确定”或“待运行”。

## 参考入口

- Mamba 论文：`Mamba: Linear-Time Sequence Modeling with Selective State Spaces`，arXiv:2312.00752。
- 官方实现：`state-spaces/mamba`。
- 已完成的 Transformer 对照项目：`D:\holiday_learning\mini_GPT`。
- 已完成的视觉 Transformer 项目：`D:\holiday_learning\ViT`。

正式使用外部资料或安装包前，应再次核验论文版本、官方仓库说明和当前兼容要求。

## 下一步

上传最新代码及 `data/` 内三个 `.bin` 文件到 AutoDL 后，依次运行测试、
`dataset.py`、`shape_check.py`、1000-step baseline 和 `evaluate.py`；记录
TensorBoard 曲线、最佳 validation 指标与最终 test 指标。之后再整理
Transformer/Mamba 数据流对照和实验结论。
