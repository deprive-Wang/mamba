"""在连续 token 流上训练极小 Mamba 语言模型。"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dataset import DEFAULT_DATA_DIR, get_batch, load_splits
from model import MambaConfig, MambaLanguageModel
from reporting import format_key_values

PEAK_LR = 3e-4
MIN_LR = 3e-5
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0


def configure_optimizer(
    model: nn.Module,
    learning_rate: float,
    device: torch.device,
) -> torch.optim.AdamW:
    """A_log、D、bias、norm 不衰减，其余矩阵参数使用 AdamW 衰减。"""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        should_skip = parameter.ndim < 2 or bool(
            getattr(parameter, "_no_weight_decay", False)
        )
        (no_decay if should_skip else decay).append(parameter)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=BETAS,
        fused=device.type == "cuda",
    )


def get_learning_rate(step: int, warmup_steps: int, max_steps: int) -> float:
    """linear warmup 后 cosine 衰减。step 从 0 开始。"""
    if warmup_steps and step < warmup_steps:
        return PEAK_LR * (step + 1) / warmup_steps
    if step >= max_steps:
        return MIN_LR
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR + cosine * (PEAK_LR - MIN_LR)


@torch.no_grad()
def estimate_loss(
    model: MambaLanguageModel,
    splits: dict[str, np.memmap],
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    """在 train/val 上各随机抽若干 batch，返回平均 token 交叉熵。"""
    model.eval()
    losses: dict[str, float] = {}
    for split_name, tokens in splits.items():
        total_loss = 0.0
        for _ in range(eval_iters):
            x, y = get_batch(tokens, batch_size, block_size, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("评估前向没有返回 loss")
            total_loss += float(loss.item())
        losses[split_name] = total_loss / eval_iters
    model.train()
    return losses


def save_checkpoint(
    path: Path,
    model: MambaLanguageModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "best_val_loss": best_val_loss,
            "config": asdict(model.config),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: MambaLanguageModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, float]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到 checkpoint：{path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("config") != asdict(model.config):
        raise ValueError("checkpoint 的模型配置与本次命令不一致")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint["step"]), float(checkpoint["best_val_loss"])


def format_metrics_header() -> str:
    return (
        f"| {'step':>9} | {'train loss':>10} | {'val loss':>8} | "
        f"{'val ppl':>8} | {'lr':>9} | {'tok/s':>9} | {'peak mem':>9} |"
    )


def format_metrics_row(
    step: int,
    max_steps: int,
    train_loss: float | None,
    val_loss: float | None,
    learning_rate: float,
    tokens_per_second: float | None,
    peak_memory_mb: float,
) -> str:
    train_text = "-" if train_loss is None else f"{train_loss:.4f}"
    val_text = "-" if val_loss is None else f"{val_loss:.4f}"
    perplexity = "-" if val_loss is None else f"{math.exp(min(val_loss, 20)):.2f}"
    speed = "-" if tokens_per_second is None else f"{tokens_per_second:.0f}"
    return (
        f"| {f'{step}/{max_steps}':>9} | {train_text:>10} | {val_text:>8} | "
        f"{perplexity:>8} | {learning_rate:>9.2e} | {speed:>9} | "
        f"{f'{peak_memory_mb:.0f} MB':>9} |"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练纯 PyTorch 教学版 Mamba LM")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-iters", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--vocab-size", type=int, default=50_257)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="只跑链路，不写 checkpoint",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> torch.device:
    positive_names = (
        "max_steps",
        "batch_size",
        "block_size",
        "grad_accum",
        "eval_interval",
        "eval_iters",
        "log_interval",
        "vocab_size",
        "d_model",
        "n_layers",
        "d_state",
        "d_conv",
        "expand",
    )
    for name in positive_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} 必须大于 0")
    if args.warmup_steps < 0 or args.warmup_steps > args.max_steps:
        raise ValueError("--warmup-steps 必须位于 [0, max-steps]")
    if args.checkpoint_interval < 0:
        raise ValueError("--checkpoint-interval 不能小于 0")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但 torch.cuda.is_available() 为 False")
    return device


def main() -> None:
    args = parse_args()
    device = validate_args(args)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    config = MambaConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
    )
    model = MambaLanguageModel(config).to(device)
    optimizer = configure_optimizer(model, PEAK_LR, device)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    splits = load_splits(args.data_dir)

    start_step = 0
    best_val_loss = float("inf")
    if args.resume is not None:
        start_step, best_val_loss = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scaler,
            device,
        )
        if start_step >= args.max_steps:
            raise ValueError("checkpoint step 已达到或超过 --max-steps")

    print(
        format_key_values(
            [
                ("设备", device),
                ("PyTorch", torch.__version__),
                ("参数量", f"{model.num_parameters():,}"),
                ("数据目录", args.data_dir.expanduser().resolve()),
                ("batch / grad_accum", f"{args.batch_size} / {args.grad_accum}"),
                ("序列长度 T", args.block_size),
                ("d_model / layers", f"{args.d_model} / {args.n_layers}"),
                (
                    "d_state / d_conv / expand",
                    f"{args.d_state} / {args.d_conv} / {args.expand}",
                ),
                ("实现", "纯 PyTorch 逐步扫描；仅用于学习和小实验"),
            ]
        )
    )
    print(format_metrics_header())

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    latest_train_loss: float | None = None
    last_eval_loss: float | None = None
    interval_tokens = 0
    interval_start = time.perf_counter()

    for step in range(start_step, args.max_steps):
        learning_rate = get_learning_rate(step, args.warmup_steps, args.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        accumulated_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(
                splits["train"],
                args.batch_size,
                args.block_size,
                device,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                _, loss = model(x, y)
                if loss is None:
                    raise RuntimeError("训练前向没有返回 loss")
                scaled_loss = loss / args.grad_accum
            scaler.scale(scaled_loss).backward()
            accumulated_loss += float(loss.item())
            interval_tokens += x.numel()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        completed_step = step + 1
        latest_train_loss = accumulated_loss / args.grad_accum

        should_eval = completed_step == 1 or completed_step % args.eval_interval == 0
        if should_eval:
            losses = estimate_loss(
                model,
                splits,
                args.batch_size,
                args.block_size,
                args.eval_iters,
                device,
                use_amp,
            )
            last_eval_loss = losses["val"]
            if last_eval_loss < best_val_loss:
                best_val_loss = last_eval_loss
                if not args.no_save:
                    save_checkpoint(
                        args.output_dir / "best.pt",
                        model,
                        optimizer,
                        scaler,
                        completed_step,
                        best_val_loss,
                    )

        should_log = (
            completed_step == 1
            or completed_step % args.log_interval == 0
            or completed_step == args.max_steps
            or should_eval
        )
        if should_log:
            elapsed = max(time.perf_counter() - interval_start, 1e-9)
            tokens_per_second = interval_tokens / elapsed
            peak_memory_mb = (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else 0.0
            )
            print(
                format_metrics_row(
                    completed_step,
                    args.max_steps,
                    latest_train_loss,
                    last_eval_loss if should_eval else None,
                    learning_rate,
                    tokens_per_second,
                    peak_memory_mb,
                )
            )
            interval_tokens = 0
            interval_start = time.perf_counter()

        should_checkpoint = (
            not args.no_save
            and args.checkpoint_interval > 0
            and completed_step % args.checkpoint_interval == 0
        )
        if should_checkpoint or (not args.no_save and completed_step == args.max_steps):
            save_checkpoint(
                args.output_dir / "latest.pt",
                model,
                optimizer,
                scaler,
                completed_step,
                best_val_loss,
            )


if __name__ == "__main__":
    main()
