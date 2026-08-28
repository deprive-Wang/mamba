"""Mamba block 关键 tensor shape、残差与因果性检查。"""

from __future__ import annotations

import argparse

import torch

from model import MambaConfig, MambaMixer, ResidualMambaBlock
from reporting import format_key_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查最小 Mamba block 数据流")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MambaConfig(block_size=args.seq_len, d_model=args.d_model, n_layers=1)
    device = torch.device(args.device)
    mixer = MambaMixer(config).to(device).eval()
    block = ResidualMambaBlock(config).to(device).eval()
    hidden_states = torch.randn(
        args.batch_size,
        args.seq_len,
        args.d_model,
        device=device,
    )

    output, shapes = mixer.forward_with_shapes(hidden_states)
    if output.shape != hidden_states.shape:
        raise RuntimeError("Mamba mixer 没有保持 [B,T,D]")

    residual_output = block(hidden_states)
    if residual_output.shape != hidden_states.shape:
        raise RuntimeError("残差连接两侧 shape 不一致")

    # 改变最后一个时间步：因果模型前 T-1 个输出必须保持不变。
    changed = hidden_states.clone()
    changed[:, -1] += 1.0
    changed_output = mixer(changed)
    causal_ok = torch.allclose(
        output[:, :-1],
        changed_output[:, :-1],
        atol=1e-5,
        rtol=1e-4,
    )
    if not causal_ok:
        raise RuntimeError("因果性检查失败：未来输入影响了过去输出")

    rows = [
        ("设备", device),
        ("d_state N", config.d_state),
        ("d_conv", config.d_conv),
        ("expand E", config.expand),
        ("dt_rank R", config.resolved_dt_rank),
    ]
    rows.extend(shapes.items())
    rows.extend(
        [
            ("残差 shape", tuple(residual_output.shape)),
            ("因果性", "通过：最后一个输入不影响之前输出"),
            ("实现边界", "教学参考扫描，不代表官方 CUDA kernel 性能"),
        ]
    )
    print(format_key_values(rows))


if __name__ == "__main__":
    main()
