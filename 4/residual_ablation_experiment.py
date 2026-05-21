"""Residual-connection ablation experiment for a mini Transformer.

The script reuses the same compact Transformer encoder-decoder setting as the
reproduction code, but compares two variants on the same reverse-copy task:

    1. transformer: standard LayerNorm(x + Sublayer(x))
    2. no_residual: ablated LayerNorm(Sublayer(x))

    source tokens:  [a, b, c, d, ...]
    target tokens:  [..., d, c, b, a, EOS]

Run:
    python residual_ablation_experiment.py --quick
    python residual_ablation_experiment.py --steps 400 --seq-len 5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F


PAD = 0
BOS = 1
EOS = 2


@dataclass
class ExperimentConfig:
    seed: int = 7
    vocab_size: int = 32
    seq_len: int = 10
    d_model: int = 64
    heads: int = 4
    d_ff: int = 128
    layers: int = 3
    dropout: float = 0.1
    batch_size: int = 64
    steps: int = 300
    eval_batches: int = 20
    lr: float = 0.001
    log_every: int = 50
    device: str = "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def make_batch(cfg: ExperimentConfig, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create one batch of random reverse-copy examples.

    Returns:
        src: [B, S]
        tgt_in: [B, S + 1], starts with BOS
        tgt_out: [B, S + 1], reversed source followed by EOS
    """
    src = torch.randint(3, cfg.vocab_size, (batch_size, cfg.seq_len), device=cfg.device)
    reversed_src = torch.flip(src, dims=[1])
    bos = torch.full((batch_size, 1), BOS, dtype=torch.long, device=cfg.device)
    eos = torch.full((batch_size, 1), EOS, dtype=torch.long, device=cfg.device)
    tgt_in = torch.cat([bos, reversed_src], dim=1)
    tgt_out = torch.cat([reversed_src, eos], dim=1)
    return src, tgt_in, tgt_out


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by heads")
        self.d_model = d_model
        self.heads = heads
        self.d_head = d_model // heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.heads, self.d_head).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._split(self.q_proj(query))
        k = self._split(self.k_proj(key))
        v = self._split(self.v_proj(value))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(query.size(0), query.size(1), self.d_model)
        return self.out_proj(context)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, cfg: ExperimentConfig, residual: bool) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.heads, cfg.dropout)
        self.ff = PositionWiseFeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.residual = residual

    def _join(self, x: torch.Tensor, sublayer_out: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        if self.residual:
            return norm(x + self.dropout(sublayer_out))
        return norm(self.dropout(sublayer_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._join(x, self.self_attn(x, x, x), self.norm1)
        x = self._join(x, self.ff(x), self.norm2)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ExperimentConfig, residual: bool) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.heads, cfg.dropout)
        self.cross_attn = MultiHeadAttention(cfg.d_model, cfg.heads, cfg.dropout)
        self.ff = PositionWiseFeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.norm3 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.residual = residual

    def _join(self, x: torch.Tensor, sublayer_out: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        if self.residual:
            return norm(x + self.dropout(sublayer_out))
        return norm(self.dropout(sublayer_out))

    def forward(self, x: torch.Tensor, memory: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        x = self._join(x, self.self_attn(x, x, x, causal_mask), self.norm1)
        x = self._join(x, self.cross_attn(x, memory, memory), self.norm2)
        x = self._join(x, self.ff(x), self.norm3)
        return x


class MiniTransformer(nn.Module):
    def __init__(self, cfg: ExperimentConfig, residual: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.src_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.tgt_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = SinusoidalPositionalEncoding(cfg.d_model, max_len=cfg.seq_len + 8)
        self.encoder = nn.ModuleList([EncoderLayer(cfg, residual) for _ in range(cfg.layers)])
        self.decoder = nn.ModuleList([DecoderLayer(cfg, residual) for _ in range(cfg.layers)])
        self.out = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.scale = math.sqrt(cfg.d_model)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        src_x = self.pos(self.src_embed(src) * self.scale)
        tgt_x = self.pos(self.tgt_embed(tgt_in) * self.scale)
        for layer in self.encoder:
            src_x = layer(src_x)
        causal_mask = torch.triu(
            torch.ones(tgt_in.size(1), tgt_in.size(1), dtype=torch.bool, device=tgt_in.device),
            diagonal=1,
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        for layer in self.decoder:
            tgt_x = layer(tgt_x, src_x, causal_mask)
        return self.out(tgt_x)


class SimpleAdam:
    """Small Adam optimizer to avoid optional torch.optim imports in lean envs."""

    def __init__(
        self,
        params,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.98),
        eps: float = 1e-9,
    ) -> None:
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_count = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self) -> None:
        self.step_count += 1
        bias_correction1 = 1.0 - self.beta1**self.step_count
        bias_correction2 = 1.0 - self.beta2**self.step_count
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            grad = p.grad
            m.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
            v.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
            step_size = self.lr * math.sqrt(bias_correction2) / bias_correction1
            p.addcdiv_(m, v.sqrt().add_(self.eps), value=-step_size)


@torch.no_grad()
def clip_grad_norm(parameters, max_norm: float) -> float:
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0
    total_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.detach()) for p in params]))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in params:
            p.grad.mul_(clip_coef)
    return float(total_norm)


@torch.no_grad()
def evaluate(model: MiniTransformer, cfg: ExperimentConfig, batches: int) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    correct_sequences = 0
    total_sequences = 0
    for _ in range(batches):
        src, tgt_in, tgt_out = make_batch(cfg, cfg.batch_size)
        logits = model(src, tgt_in)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), tgt_out.reshape(-1))
        pred = logits.argmax(dim=-1)
        matches = pred.eq(tgt_out)
        total_loss += loss.item()
        correct_tokens += matches.sum().item()
        total_tokens += tgt_out.numel()
        correct_sequences += matches.all(dim=1).sum().item()
        total_sequences += tgt_out.size(0)
    return {
        "loss": total_loss / batches,
        "token_accuracy": correct_tokens / total_tokens,
        "sequence_accuracy": correct_sequences / total_sequences,
    }


def train_one(name: str, residual: bool, cfg: ExperimentConfig) -> tuple[dict[str, float], list[dict[str, float]]]:
    set_seed(cfg.seed)
    model = MiniTransformer(cfg, residual=residual).to(cfg.device)
    optimizer = SimpleAdam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9)
    history: list[dict[str, float]] = []
    started = time.time()
    for step in range(1, cfg.steps + 1):
        model.train()
        src, tgt_in, tgt_out = make_batch(cfg, cfg.batch_size)
        logits = model(src, tgt_in)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), tgt_out.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % cfg.log_every == 0 or step == cfg.steps:
            metrics = evaluate(model, cfg, max(2, cfg.eval_batches // 4))
            row = {"model": name, "step": step, "train_loss": loss.item(), **metrics}
            history.append(row)
            print(
                f"{name:14s} step={step:4d} train_loss={loss.item():.3f} "
                f"eval_loss={metrics['loss']:.3f} token_acc={metrics['token_accuracy']:.3f} "
                f"seq_acc={metrics['sequence_accuracy']:.3f}",
                flush=True,
            )
    final = evaluate(model, cfg, cfg.eval_batches)
    final["seconds"] = time.time() - started
    final["parameters"] = sum(p.numel() for p in model.parameters())
    return final, history


def write_results(out_dir: Path, cfg: ExperimentConfig, results: dict[str, dict[str, float]], history: list[dict[str, float]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    if history:
        keys = list(history[0].keys())
        with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(history)


def plot_ablation_history(history: list[dict[str, float]], out_dir: Path) -> Path:
    """Save token/sequence accuracy curves for both ablation variants."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "residual_ablation_curves.png"
    width, height = 1120, 650
    margin_l, margin_r, margin_t, margin_b = 100, 50, 80, 90
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("arial.ttf", 30)
        font = ImageFont.truetype("arial.ttf", 22)
        small = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        title_font = font = small = ImageFont.load_default()

    x0, y0 = margin_l, height - margin_b
    x1, y1 = width - margin_r, margin_t
    draw.text((margin_l, 24), "Residual Ablation on Reverse-Copy Task", fill=(20, 20, 20), font=title_font)
    draw.line((x0, y0, x1, y0), fill=(40, 40, 40), width=2)
    draw.line((x0, y0, x0, y1), fill=(40, 40, 40), width=2)
    for i in range(6):
        y = y0 - i * (y0 - y1) / 5
        draw.line((x0, y, x1, y), fill=(225, 225, 225), width=1)
        draw.text((35, y - 10), f"{i / 5:.1f}", fill=(70, 70, 70), font=small)

    steps = sorted({int(row["step"]) for row in history})
    min_step, max_step = min(steps), max(steps)
    span = max(1, max_step - min_step)
    series = [
        ("Transformer token acc.", "transformer", "token_accuracy", (35, 110, 180)),
        ("Transformer sequence acc.", "transformer", "sequence_accuracy", (45, 150, 80)),
        ("No residual token acc.", "no_residual", "token_accuracy", (185, 70, 60)),
        ("No residual sequence acc.", "no_residual", "sequence_accuracy", (120, 120, 120)),
    ]
    for label, model, key, color in series:
        points = []
        for row in history:
            if row["model"] != model:
                continue
            x = x0 + (int(row["step"]) - min_step) / span * (x1 - x0)
            y = y0 - float(row[key]) * (y0 - y1)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=4)
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)

    lx, ly = 600, 95
    draw.rectangle((lx - 15, ly - 12, lx + 425, ly + 130), fill="white", outline=(230, 230, 230))
    for i, (label, _, _, color) in enumerate(series):
        yy = ly + i * 32
        draw.line((lx, yy + 10, lx + 45, yy + 10), fill=color, width=5)
        draw.text((lx + 58, yy), label, fill=(45, 45, 45), font=small)
    draw.text((width / 2 - 65, height - 42), "Training step", fill=(70, 70, 70), font=font)
    draw.text((18, 48), "Accuracy", fill=(70, 70, 70), font=font)
    image.save(path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use a shorter CPU-friendly run.")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=Path("results_residual_ablation"))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(seed=args.seed, device=args.device)
    if args.quick:
        cfg.steps = 160
        cfg.layers = 3
        cfg.seq_len = 8
        cfg.eval_batches = 10
    if args.steps is not None:
        cfg.steps = args.steps
    if args.layers is not None:
        cfg.layers = args.layers
    if args.seq_len is not None:
        cfg.seq_len = args.seq_len
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    print("Configuration:")
    print(json.dumps(asdict(cfg), indent=2))
    all_history: list[dict[str, float]] = []
    results: dict[str, dict[str, float]] = {}
    for name, residual in [("transformer", True), ("no_residual", False)]:
        final, history = train_one(name, residual, cfg)
        results[name] = final
        all_history.extend(history)

    write_results(args.out_dir, cfg, results, all_history)
    figure_path = plot_ablation_history(all_history, args.out_dir)
    print("\nFinal results:")
    print(json.dumps(results, indent=2))
    print(f"Saved results to {args.out_dir.resolve()}")
    print(f"Saved figure to {figure_path.resolve()}")


if __name__ == "__main__":
    main()
