"""使用训练期间未见的 test token 评估最佳 checkpoint。"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from data_splits import load_test_split
from dataset import DEFAULT_DATA_DIR, get_batch
from model import MambaConfig, MambaLanguageModel
from reporting import format_key_values

DEFAULT_CHECKPOINT = Path("checkpoints/best.pt")
DEFAULT_EVAL_BATCHES = 100
DEFAULT_BATCH_SIZE = 4
DEFAULT_TENSORBOARD_DIR = Path(__file__).resolve().parent / "tf-logs"


def _load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[MambaLanguageModel, dict[str, object]]:
    resolved_path = checkpoint_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"找不到 checkpoint：{resolved_path}")

    checkpoint = torch.load(
        resolved_path,
        map_location=device,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint 顶层必须是字典")

    config_values = checkpoint.get("config")
    if not isinstance(config_values, dict):
        raise ValueError("checkpoint 缺少有效的模型 config")
    try:
        config = MambaConfig(**config_values)
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint 的模型 config 无效") from error

    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict):
        raise ValueError("checkpoint 缺少有效的模型权重")

    model = MambaLanguageModel(config).to(device)
    model.load_state_dict(model_state)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: Path,
    data_dir: Path,
    batch_size: int,
    eval_batches: int,
    seed: int,
    device: torch.device,
) -> dict[str, float | int | str]:
    """对物理隔离的 test.bin 做可复现的随机 batch 评估。"""
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if eval_batches <= 0:
        raise ValueError("eval_batches 必须大于 0")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但 torch.cuda.is_available() 为 False")

    model, checkpoint = _load_model(checkpoint_path, device)
    checkpoint_step = checkpoint.get("step")
    if (
        isinstance(checkpoint_step, bool)
        or not isinstance(checkpoint_step, int)
        or checkpoint_step < 0
    ):
        raise ValueError("checkpoint 缺少有效的 step")
    best_val_loss = checkpoint.get("best_val_loss")
    if (
        isinstance(best_val_loss, bool)
        or not isinstance(best_val_loss, (int, float))
        or not math.isfinite(float(best_val_loss))
    ):
        raise ValueError("checkpoint 缺少有效的 best_val_loss")

    test_tokens = load_test_split(data_dir)
    if len(test_tokens) <= model.config.block_size:
        raise ValueError(
            "test token 数量不足以评估 checkpoint 的 block_size="
            f"{model.config.block_size}"
        )

    generator = torch.Generator().manual_seed(seed)
    use_amp = device.type == "cuda"
    if use_amp:
        torch.cuda.reset_peak_memory_stats(device)

    total_loss = 0.0
    evaluated_tokens = 0
    started_at = time.perf_counter()
    for _ in range(eval_batches):
        inputs, targets = get_batch(
            test_tokens,
            batch_size,
            model.config.block_size,
            device,
            generator,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            _, loss = model(inputs, targets)
        if loss is None:
            raise RuntimeError("测试前向没有返回 loss")
        total_loss += float(loss.item())
        evaluated_tokens += targets.numel()

    elapsed_seconds = max(time.perf_counter() - started_at, 1e-9)
    test_loss = total_loss / eval_batches
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if use_amp
        else 0.0
    )
    return {
        "checkpoint_step": checkpoint_step,
        "best_val_loss": float(best_val_loss),
        "test_tokens_total": len(test_tokens),
        "eval_batches": eval_batches,
        "evaluated_tokens": evaluated_tokens,
        "test_loss": test_loss,
        "test_perplexity": math.exp(min(test_loss, 20)),
        "elapsed_seconds": elapsed_seconds,
        "tokens_per_second": evaluated_tokens / elapsed_seconds,
        "peak_memory_mb": peak_memory_mb,
        "evaluation_mode": f"固定 seed={seed} 的随机 batch",
    }


def build_evaluation_log_dir(
    tensorboard_dir: Path,
    run_name: str | None,
    checkpoint_path: Path,
) -> Path:
    """解析最终测试指标的 TensorBoard 目录。"""
    name = run_name or f"evaluation-{checkpoint_path.stem}"
    if not name.strip() or name in {".", ".."} or Path(name).name != name:
        raise ValueError("--run-name 必须是单个非空目录名，不能包含路径分隔符")
    return tensorboard_dir.expanduser() / name


def write_evaluation_summary(
    result: dict[str, float | int | str],
    log_dir: Path,
) -> None:
    """将最终 test 指标追加到指定的 TensorBoard run。"""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise RuntimeError(
            "缺少 TensorBoard，请先执行 pip install -r requirements.txt"
        ) from error

    checkpoint_step = int(result["checkpoint_step"])
    writer = SummaryWriter(log_dir=str(log_dir))
    try:
        writer.add_scalar("test/loss", float(result["test_loss"]), checkpoint_step)
        writer.add_scalar(
            "test/perplexity",
            float(result["test_perplexity"]),
            checkpoint_step,
        )
        writer.add_scalar(
            "test/tokens_per_second",
            float(result["tokens_per_second"]),
            checkpoint_step,
        )
        writer.add_scalar(
            "test/peak_memory_mb",
            float(result["peak_memory_mb"]),
            checkpoint_step,
        )
        writer.add_scalar(
            "test/evaluated_tokens",
            int(result["evaluated_tokens"]),
            checkpoint_step,
        )
        writer.flush()
    finally:
        writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 Mamba 最佳 checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--eval-batches", type=int, default=DEFAULT_EVAL_BATCHES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=DEFAULT_TENSORBOARD_DIR,
        help="TensorBoard 日志根目录，默认项目根目录下的 tf-logs",
    )
    parser.add_argument(
        "--run-name",
        help="与训练相同则把 test 指标追加到同一 TensorBoard run",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    result = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        seed=args.seed,
        device=device,
    )
    tensorboard_log_dir = build_evaluation_log_dir(
        args.tensorboard_dir,
        args.run_name,
        args.checkpoint,
    )
    write_evaluation_summary(result, tensorboard_log_dir)
    rows = [
        ("设备", device),
        ("checkpoint", args.checkpoint.expanduser().resolve()),
        ("checkpoint step", result["checkpoint_step"]),
        ("TensorBoard", tensorboard_log_dir.expanduser().resolve()),
        ("best val loss", f"{result['best_val_loss']:.4f}"),
        ("test 数据", "物理 test.bin"),
        ("test tokens", f"{result['test_tokens_total']:,}"),
        ("评估方式", result["evaluation_mode"]),
        ("评估 token", f"{result['evaluated_tokens']:,}"),
        ("test loss", f"{result['test_loss']:.4f}"),
        ("test perplexity", f"{result['test_perplexity']:.2f}"),
        ("耗时", f"{result['elapsed_seconds']:.2f} s"),
        ("tokens/sec", f"{result['tokens_per_second']:.0f}"),
        ("峰值显存", f"{result['peak_memory_mb']:.0f} MB"),
    ]
    print(format_key_values(rows))


if __name__ == "__main__":
    main()
