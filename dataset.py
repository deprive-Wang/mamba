"""连续 token 流的数据加载与随机 batch 采样。

兼容 Mini-GPT 已生成的 ``train.bin`` / ``val.bin``：文件是连续的 uint16
token id，不保存样本边界。使用 memmap 可避免把数百 MB 数据一次性读入内存。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from reporting import format_key_values

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_BLOCK_SIZE = 128
DEFAULT_BATCH_SIZE = 8
TOKEN_DTYPE = np.dtype(np.uint16)


def load_token_stream(path: Path) -> np.memmap:
    """只读打开 uint16 token 流，并在 IO 边界检查文件是否合法。"""
    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"找不到 token 文件：{resolved_path}")
    if resolved_path.stat().st_size == 0:
        raise ValueError(f"token 文件为空：{resolved_path}")
    if resolved_path.stat().st_size % TOKEN_DTYPE.itemsize != 0:
        raise ValueError(
            f"文件字节数不是 uint16 的整数倍，可能已损坏：{resolved_path}"
        )
    return np.memmap(resolved_path, dtype=TOKEN_DTYPE, mode="r")


def load_splits(data_dir: Path) -> dict[str, np.memmap]:
    """加载 ``data_dir/train.bin`` 与 ``data_dir/val.bin``。"""
    resolved_dir = data_dir.expanduser().resolve()
    if not resolved_dir.is_dir():
        raise FileNotFoundError(
            f"数据目录不存在：{resolved_dir}。可用 --data-dir 指向 Mini-GPT/data。"
        )
    return {
        split: load_token_stream(resolved_dir / f"{split}.bin")
        for split in ("train", "val")
    }


def get_batch(
    tokens: np.memmap | np.ndarray,
    batch_size: int = DEFAULT_BATCH_SIZE,
    block_size: int = DEFAULT_BLOCK_SIZE,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """随机采样自回归输入 x 和右移一格的标签 y，均为 ``[B,T]``。"""
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if block_size <= 0:
        raise ValueError("block_size 必须大于 0")
    if tokens.ndim != 1:
        raise ValueError("token 流必须是一维数组")

    # 每个样本需要 T+1 个 token，最后一个合法起点是 len(tokens)-T-1。
    max_start = len(tokens) - block_size - 1
    if max_start < 0:
        raise ValueError(
            f"token 数量 {len(tokens):,} 不足以采样 block_size={block_size}"
        )
    starts = torch.randint(
        low=0,
        high=max_start + 1,
        size=(batch_size,),
        generator=generator,
    )

    windows = np.stack(
        [
            np.asarray(tokens[int(start):int(start) + block_size + 1], dtype=np.int64)
            for start in starts
        ]
    )
    batch = torch.from_numpy(windows)
    x = batch[:, :-1].contiguous()
    y = batch[:, 1:].contiguous()

    target_device = torch.device(device)
    if target_device.type != "cpu":
        x = x.to(target_device, non_blocking=True)
        y = y.to(target_device, non_blocking=True)
    return x, y


def validate_token_range(tokens: np.memmap, vocab_size: int) -> tuple[int, int]:
    """检查 token id 是否能安全进入指定词表的 Embedding。"""
    if vocab_size <= 0:
        raise ValueError("vocab_size 必须大于 0")
    min_token = int(tokens.min())
    max_token = int(tokens.max())
    if min_token < 0 or max_token >= vocab_size:
        raise ValueError(
            f"token 范围 [{min_token}, {max_token}] 超出词表 [0, {vocab_size - 1}]"
        )
    return min_token, max_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 Mamba 训练数据")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--vocab-size", type=int, default=50_257)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = load_splits(args.data_dir)
    rows: list[tuple[str, object]] = []
    for name, tokens in splits.items():
        min_token, max_token = validate_token_range(tokens, args.vocab_size)
        rows.extend(
            [
                (f"{name} tokens", f"{len(tokens):,}"),
                (f"{name} token 范围", f"[{min_token}, {max_token}]"),
            ]
        )

    generator = torch.Generator().manual_seed(42)
    x, y = get_batch(
        splits["train"],
        args.batch_size,
        args.block_size,
        generator=generator,
    )
    if not torch.equal(y[:, :-1], x[:, 1:]):
        raise RuntimeError("label 右移校验失败")
    rows.extend(
        [
            ("x shape", tuple(x.shape)),
            ("y shape", tuple(y.shape)),
            ("dtype", x.dtype),
            ("label 右移", "通过：y[:, :-1] == x[:, 1:]"),
        ]
    )
    print(format_key_values(rows))


if __name__ == "__main__":
    main()
