"""纯 PyTorch 的 Mamba v1 教学实现。

该实现对齐论文与官方 ``mamba_simple.py`` 的核心参数和数据流，但使用 Python
循环完成 selective scan。它适合阅读、shape 跟踪和小实验，不代表官方融合
CUDA kernel 的速度或显存表现。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class MambaConfig:
    """Mamba 语言模型配置。

    ``d_state=16``、``d_conv=4``、``expand=2``、``dt_rank='auto'`` 以及
    ``dt`` 初始化范围均对齐 Mamba v1 官方实现。模型宽度和层数刻意缩小，
    便于在本地跟踪数据流。
    """

    vocab_size: int = 50_257
    block_size: int = 128
    d_model: int = 128
    n_layers: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int | str = "auto"
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    bias: bool = False
    conv_bias: bool = True
    norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        positive_ints = {
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "d_state": self.d_state,
            "d_conv": self.d_conv,
            "expand": self.expand,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} 必须大于 0")
        if self.dt_rank != "auto" and (
            not isinstance(self.dt_rank, int) or self.dt_rank <= 0
        ):
            raise ValueError("dt_rank 必须是 'auto' 或正整数")
        if not 0 < self.dt_min < self.dt_max:
            raise ValueError("dt_min 和 dt_max 必须满足 0 < dt_min < dt_max")
        if self.dt_init not in {"constant", "random"}:
            raise ValueError("dt_init 只支持 'constant' 或 'random'")
        if self.dt_scale <= 0 or self.dt_init_floor <= 0:
            raise ValueError("dt_scale 和 dt_init_floor 必须大于 0")

    @property
    def d_inner(self) -> int:
        """Mamba block 内部通道数 E*D，论文常用扩张倍数 E=2。"""
        return self.expand * self.d_model

    @property
    def resolved_dt_rank(self) -> int:
        """官方 ``auto`` 规则：低秩 Δ 投影维度 R=ceil(D/16)。"""
        if self.dt_rank == "auto":
            return math.ceil(self.d_model / 16)
        return int(self.dt_rank)


class RMSNorm(nn.Module):
    """只按最后一维做 RMS 归一化，不减均值。"""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        values = hidden_states.float()
        rms = values.square().mean(dim=-1, keepdim=True)
        normalized = values * torch.rsqrt(rms + self.eps)
        return (normalized * self.weight.float()).to(input_dtype)


class MambaMixer(nn.Module):
    """Mamba v1 的核心序列变换，输入输出均为 ``[B, T, D]``。"""

    def __init__(self, config: MambaConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_inner = config.d_inner
        self.d_state = config.d_state
        self.dt_rank = config.resolved_dt_rank

        # 一次投影得到 SSM 输入 x 与门控分支 z，各为 [B,T,d_inner]。
        self.in_proj = nn.Linear(
            config.d_model,
            2 * config.d_inner,
            bias=config.bias,
        )
        # groups=d_inner 表示逐通道短卷积，只混合局部时间信息，不混通道。
        self.conv1d = nn.Conv1d(
            config.d_inner,
            config.d_inner,
            kernel_size=config.d_conv,
            groups=config.d_inner,
            padding=config.d_conv - 1,
            bias=config.conv_bias,
        )
        # 当前 token 生成选择性参数：低秩 Δ、B、C。
        self.x_proj = nn.Linear(
            config.d_inner,
            self.dt_rank + 2 * config.d_state,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.dt_rank, config.d_inner, bias=True)

        self._initialize_dt(config)

        # A=-exp(A_log) 保证连续系统稳定；S4D-Real 初始化为 1..N。
        base_a = torch.arange(1, config.d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(base_a.log().repeat(config.d_inner, 1))
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]

        # D 是 SSM 外的逐通道 skip，避免所有信息都必须经过状态递推。
        self.D = nn.Parameter(torch.ones(config.d_inner, dtype=torch.float32))
        self.D._no_weight_decay = True  # type: ignore[attr-defined]
        self.out_proj = nn.Linear(
            config.d_inner,
            config.d_model,
            bias=config.bias,
        )

    def _initialize_dt(self, config: MambaConfig) -> None:
        """让 softplus(dt_bias) 在论文实现的 [dt_min, dt_max] 对数区间。"""
        init_std = self.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, init_std)
        else:
            nn.init.uniform_(self.dt_proj.weight, -init_std, init_std)

        dt = torch.exp(
            torch.rand(self.d_inner)
            * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        # inverse softplus：softplus(inv_dt) == dt，避免初始 Δ 全挤在同一尺度。
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def _selective_scan(
        self,
        x: Tensor,
        delta: Tensor,
        input_b: Tensor,
        input_c: Tensor,
    ) -> Tensor:
        """按时间递推 selective SSM。

        Shapes:
            x, delta: [B,T,d_inner]
            input_b, input_c: [B,T,N]
            state h_t: [B,d_inner,N]

        这里显式保留时间循环，正是为了能逐步看见
        ``h_t = A_bar_t*h_(t-1) + B_bar_t*x_t``。官方实现会把该扫描融合为
        CUDA kernel，因此本函数不能用于官方性能对比。
        """
        input_dtype = x.dtype
        x = x.float()
        delta = delta.float()
        input_b = input_b.float()
        input_c = input_c.float()
        a = -torch.exp(self.A_log.float())  # [d_inner,N]
        skip = self.D.float()

        batch_size, seq_len, _ = x.shape
        state = x.new_zeros((batch_size, self.d_inner, self.d_state))
        outputs: list[Tensor] = []

        for time_index in range(seq_len):
            delta_t = delta[:, time_index]  # [B,d_inner]
            x_t = x[:, time_index]  # [B,d_inner]
            b_t = input_b[:, time_index]  # [B,N]
            c_t = input_c[:, time_index]  # [B,N]

            # ZOH 形式的 A_bar=exp(ΔA)；B_bar*x 使用官方参考扫描的 Δ*B*x。
            a_bar = torch.exp(delta_t.unsqueeze(-1) * a.unsqueeze(0))
            bx = (
                delta_t.unsqueeze(-1)
                * b_t.unsqueeze(1)
                * x_t.unsqueeze(-1)
            )
            state = a_bar * state + bx
            y_t = (state * c_t.unsqueeze(1)).sum(dim=-1) + skip * x_t
            outputs.append(y_t)

        return torch.stack(outputs, dim=1).to(input_dtype)

    def _forward_impl(
        self,
        hidden_states: Tensor,
    ) -> tuple[Tensor, dict[str, tuple[int, ...]]]:
        if hidden_states.ndim != 3:
            raise ValueError("MambaMixer 输入必须是 [B,T,D] 三维张量")
        if hidden_states.size(-1) != self.d_model:
            raise ValueError(
                f"输入最后一维应为 d_model={self.d_model}，"
                f"实际为 {hidden_states.size(-1)}"
            )

        _, seq_len, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)

        # Conv1d 使用 [B,C,T]；截断右侧 padding 后，位置 t 不会看见未来 token。
        x = self.conv1d(x.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        x = F.silu(x)

        selective = self.x_proj(x)
        delta_low_rank, input_b, input_c = torch.split(
            selective,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        delta = F.softplus(self.dt_proj(delta_low_rank))
        y = self._selective_scan(x, delta, input_b, input_c)

        # z 是与 SSM 主分支并行的输入依赖门控，对齐官方 SiLU(z) 门控。
        gated = y * F.silu(z)
        output = self.out_proj(gated)
        shapes = {
            "输入 hidden_states": tuple(hidden_states.shape),
            "输入投影 xz": tuple(xz.shape),
            "卷积后 x": tuple(x.shape),
            "低秩 delta": tuple(delta_low_rank.shape),
            "选择参数 B": tuple(input_b.shape),
            "选择参数 C": tuple(input_c.shape),
            "步长 delta": tuple(delta.shape),
            "扫描输出 y": tuple(y.shape),
            "输出 output": tuple(output.shape),
        }
        return output, shapes

    def forward(self, hidden_states: Tensor) -> Tensor:
        output, _ = self._forward_impl(hidden_states)
        return output

    @torch.no_grad()
    def forward_with_shapes(
        self,
        hidden_states: Tensor,
    ) -> tuple[Tensor, dict[str, tuple[int, ...]]]:
        """执行一次前向并返回关键 tensor shape，供 ``shape_check.py`` 使用。"""
        return self._forward_impl(hidden_states)


class ResidualMambaBlock(nn.Module):
    """Pre-RMSNorm + Mamba mixer + residual，不额外插入 attention/MLP。"""

    def __init__(self, config: MambaConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.mixer = MambaMixer(config)

    def forward(self, hidden_states: Tensor) -> Tensor:
        # 残差相加要求 mixer 输出严格保持 [B,T,D]。
        return hidden_states + self.mixer(self.norm(hidden_states))


class MambaLanguageModel(nn.Module):
    """字符/BPE token 都可用的自回归 Mamba 语言模型。"""

    def __init__(self, config: MambaConfig | None = None) -> None:
        super().__init__()
        self.config = config or MambaConfig()
        self.token_embedding = nn.Embedding(
            self.config.vocab_size,
            self.config.d_model,
        )
        self.layers = nn.ModuleList(
            ResidualMambaBlock(self.config)
            for _ in range(self.config.n_layers)
        )
        self.final_norm = RMSNorm(self.config.d_model, self.config.norm_eps)
        self.lm_head = nn.Linear(
            self.config.d_model,
            self.config.vocab_size,
            bias=False,
        )
        # 输入输出词嵌入共享；Mamba 依靠递推编码顺序，不需要绝对位置嵌入。
        self.lm_head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        token_ids: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids 必须是 [B,T] 二维张量")
        if token_ids.dtype != torch.long:
            raise TypeError("token_ids 必须使用 torch.long")
        if token_ids.numel() == 0:
            raise ValueError("token_ids 不能为空")
        if token_ids.size(1) > self.config.block_size:
            raise ValueError(
                f"序列长度 {token_ids.size(1)} 超过训练配置 block_size="
                f"{self.config.block_size}"
            )
        if token_ids.numel() and (
            int(token_ids.min()) < 0
            or int(token_ids.max()) >= self.config.vocab_size
        ):
            raise ValueError("token id 超出词表范围")
        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError("targets shape 必须与 token_ids 相同")
            if targets.dtype != torch.long:
                raise TypeError("targets 必须使用 torch.long")
            if int(targets.min()) < 0 or int(targets.max()) >= self.config.vocab_size:
                raise ValueError("target token id 超出词表范围")

        hidden_states = self.token_embedding(token_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        logits = self.lm_head(self.final_norm(hidden_states))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss
