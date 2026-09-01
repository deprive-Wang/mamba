"""连续 token 流的训练、验证与测试划分。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

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


def _validate_data_dir(data_dir: Path) -> Path:
    resolved_dir = data_dir.expanduser().resolve()
    if not resolved_dir.is_dir():
        raise FileNotFoundError(f"数据目录不存在：{resolved_dir}")
    return resolved_dir


def load_training_splits(data_dir: Path) -> dict[str, np.memmap]:
    """只返回训练可见的 train.bin 与 val.bin。"""
    resolved_dir = _validate_data_dir(data_dir)
    return {
        "train": load_token_stream(resolved_dir / "train.bin"),
        "val": load_token_stream(resolved_dir / "val.bin"),
    }


def load_test_split(data_dir: Path) -> np.memmap:
    """只返回物理隔离的 test.bin。"""
    resolved_dir = _validate_data_dir(data_dir)
    return load_token_stream(resolved_dir / "test.bin")
