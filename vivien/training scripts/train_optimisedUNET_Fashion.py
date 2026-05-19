"""
Train a compact U-Net DDPM on FashionMNIST.

This version is designed for Snellius-style GPU jobs:
- self-contained U-Net denoiser with timestep conditioning
- cosine or linear beta schedules
- DDPM posterior-variance sampling
- AMP mixed precision on CUDA
- EMA model for cleaner samples
- resumable checkpoints, loss curves, and sample grids
"""

import os
import math
import copy
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from torchvision.datasets import FashionMNIST
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW


# -----------------------------------------------------------------------------
# Arguments


def parse_args():
    p = argparse.ArgumentParser(description="Train a U-Net DDPM on FashionMNIST")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--n_timesteps", type=int, default=1000)
    p.add_argument("--beta_schedule", type=str, default="cosine", choices=["cosine", "linear"])
    p.add_argument("--beta_min", type=float, default=1e-4)
    p.add_argument("--beta_max", type=float, default=2e-2)
    p.add_argument("--model_channels", type=int, default=64)
    p.add_argument("--channel_mults", type=str, default="1,2,4")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--dataset_path", type=str, default="~/datasets")
    p.add_argument("--save_dir", type=str, default="models/unet_fashion_mnist_v2")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--sample_every", type=int, default=10)
    p.add_argument("--sample_count", type=int, default=64)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--no_amp", action="store_true", help="Disable CUDA mixed precision")
    return p.parse_args()


def parse_channel_mults(text):
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


# -----------------------------------------------------------------------------
# U-Net components


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.float()
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=x.device) * -scale)
        emb = x[:, None] * freqs[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


def group_norm(channels):
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, dropout=0.0):
        super().__init__()
        self.norm1 = group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.norm2 = group_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-2)
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = group_norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        head_dim = c // self.heads
        y = self.norm(x)
        qkv = self.qkv(y).reshape(b, 3, self.heads, head_dim, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        scale = head_dim ** -0.5
        attn = torch.einsum("bhdn,bhdm->bhnm", q * scale, k).softmax(dim=-1)
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UNetDenoiser(nn.Module):
    def __init__(self, channels=1, base_channels=64, channel_mults=(1, 2, 4), dropout=0.1):
        super().__init__()
        self.channels = channels
        model_channels = [base_channels * m for m in channel_mults]
        time_dim = base_channels * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = nn.Conv2d(channels, model_channels[0], kernel_size=3, padding=1)

        self.downs = nn.ModuleList()
        for idx in range(len(model_channels) - 1):
            ch = model_channels[idx]
            next_ch = model_channels[idx + 1]
            self.downs.append(nn.ModuleDict({
                "block1": ResidualBlock(ch, ch, time_dim, dropout),
                "block2": ResidualBlock(ch, ch, time_dim, dropout),
                "down": Downsample(ch, next_ch),
            }))

        mid_ch = model_channels[-1]
        self.mid1 = ResidualBlock(mid_ch, mid_ch, time_dim, dropout)
        self.mid_attn = AttentionBlock(mid_ch)
        self.mid2 = ResidualBlock(mid_ch, mid_ch, time_dim, dropout)

        self.ups = nn.ModuleList()
        for idx in reversed(range(len(model_channels) - 1)):
            skip_ch = model_channels[idx]
            current_ch = model_channels[idx + 1]
            self.ups.append(nn.ModuleDict({
                "up": Upsample(current_ch, skip_ch),
                "block1": ResidualBlock(skip_ch * 2, skip_ch, time_dim, dropout),
                "block2": ResidualBlock(skip_ch, skip_ch, time_dim, dropout),
            }))

        final_ch = model_channels[0]
        self.final_norm = group_norm(final_ch)
        self.final_conv = nn.Conv2d(final_ch, channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h = self.init_conv(x)

        skips = []
        for layer in self.downs:
            h = layer["block1"](h, t_emb)
            h = layer["block2"](h, t_emb)
            skips.append(h)
            h = layer["down"](h)

        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        for layer in self.ups:
            h = layer["up"](h)
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = layer["block1"](h, t_emb)
            h = layer["block2"](h, t_emb)

        return self.final_conv(F.silu(self.final_norm(h)))


# -----------------------------------------------------------------------------
# Diffusion process


class Diffusion(nn.Module):
    def __init__(self, model, image_resolution, n_times=1000, beta_schedule="cosine",
                 beta_min=1e-4, beta_max=2e-2):
        super().__init__()
        self.n_times = n_times
        self.img_H, self.img_W, self.img_C = image_resolution
        self.model = model

        if beta_schedule == "linear":
            betas = torch.linspace(beta_min, beta_max, steps=n_times)
        elif beta_schedule == "cosine":
            betas = self._cosine_beta_schedule(n_times)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_betas", torch.sqrt(betas))
        self.register_buffer("alphas", alphas)
        self.register_buffer("sqrt_alphas", torch.sqrt(alphas))
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))

    @staticmethod
    def _cosine_beta_schedule(timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alpha_bars = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_bars = alpha_bars / alpha_bars[0]
        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        return betas.clamp(1e-4, 0.999)

    def extract(self, a, t, x_shape):
        b = t.shape[0]
        out = a.gather(0, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    @staticmethod
    def scale_to_minus_one_to_one(x):
        return x * 2 - 1

    @staticmethod
    def reverse_scale_to_zero_to_one(x):
        return (x + 1) * 0.5

    def q_sample(self, x_zeros, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_zeros)
        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, t, x_zeros.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, t, x_zeros.shape)
        return x_zeros * sqrt_alpha_bar + noise * sqrt_one_minus_alpha_bar

    def forward(self, x_zeros):
        x_zeros = self.scale_to_minus_one_to_one(x_zeros)
        b = x_zeros.shape[0]
        t = torch.randint(low=0, high=self.n_times, size=(b,), device=x_zeros.device).long()
        epsilon = torch.randn_like(x_zeros)
        perturbed_images = self.q_sample(x_zeros, t, epsilon)
        pred_epsilon = self.model(perturbed_images, t)
        return perturbed_images, epsilon, pred_epsilon

    @torch.no_grad()
    def denoise_at_t(self, x_t, timestep, t):
        z = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        epsilon_pred = self.model(x_t, timestep)
        alpha = self.extract(self.alphas, timestep, x_t.shape)
        beta = self.extract(self.betas, timestep, x_t.shape)
        sqrt_alpha = self.extract(self.sqrt_alphas, timestep, x_t.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep, x_t.shape)
        posterior_variance = self.extract(self.posterior_variance, timestep, x_t.shape)
        mean = (x_t - beta / sqrt_one_minus_alpha_bar * epsilon_pred) / sqrt_alpha
        x_t_minus_1 = mean + torch.sqrt(posterior_variance) * z
        return x_t_minus_1.clamp(-1.0, 1.0)

    @torch.no_grad()
    def sample(self, n):
        device = next(self.model.parameters()).device
        x_t = torch.randn((n, self.img_C, self.img_H, self.img_W), device=device)
        for t in range(self.n_times - 1, -1, -1):
            timestep = torch.full((n,), t, device=device, dtype=torch.long)
            x_t = self.denoise_at_t(x_t, timestep, t)
        return self.reverse_scale_to_zero_to_one(x_t).clamp(0.0, 1.0)


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ema_params = dict(self.model.named_parameters())
        model_params = dict(model.named_parameters())
        for name, param in model_params.items():
            ema_params[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
        ema_buffers = dict(self.model.named_buffers())
        model_buffers = dict(model.named_buffers())
        for name, buffer in model_buffers.items():
            ema_buffers[name].copy_(buffer)


# -----------------------------------------------------------------------------
# Helpers


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_loss_curve(train_losses, path):
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(train_losses) + 1), train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE noise-prediction loss")
    plt.title("Training loss")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


@torch.no_grad()
def save_forward_process_grid(diffusion, images, path):
    device = images.device
    x0 = diffusion.scale_to_minus_one_to_one(images[:8])
    timesteps = torch.linspace(0, diffusion.n_times - 1, steps=8, device=device).long()
    rows = []
    fixed_noise = torch.randn_like(x0)
    for t in timesteps:
        t_batch = t.repeat(x0.shape[0])
        xt = diffusion.q_sample(x0, t_batch, fixed_noise)
        rows.append(diffusion.reverse_scale_to_zero_to_one(xt).clamp(0.0, 1.0))
    grid = torch.cat(rows, dim=0)
    save_image(grid, path, nrow=x0.shape[0])
    print(f"Saved: {path}")


@torch.no_grad()
def save_generated_samples(diffusion, path, sample_count):
    diffusion.model.eval()
    samples = diffusion.sample(sample_count)
    nrow = int(math.sqrt(sample_count))
    save_image(samples, path, nrow=max(1, nrow))
    print(f"Saved: {path}")


def save_checkpoint(path, epoch, model, ema, optimizer, scaler, train_losses, args):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "train_losses": train_losses,
        "config": vars(args),
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


# -----------------------------------------------------------------------------
# Main


def main():
    args = parse_args()
    args.channel_mults = parse_channel_mults(args.channel_mults)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"Device: {device}")
    print(f"AMP enabled: {use_amp}")
    print(f"Config: {vars(args)}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = FashionMNIST(os.path.expanduser(args.dataset_path), transform=transform, train=True, download=True)
    test_dataset = FashionMNIST(os.path.expanduser(args.dataset_path), transform=transform, train=False, download=True)

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, **loader_kwargs)

    img_size = (28, 28, 1)
    model = UNetDenoiser(
        channels=1,
        base_channels=args.model_channels,
        channel_mults=args.channel_mults,
        dropout=args.dropout,
    ).to(device)

    diffusion = Diffusion(
        model,
        image_resolution=img_size,
        n_times=args.n_timesteps,
        beta_schedule=args.beta_schedule,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
    ).to(device)

    ema = EMA(model, decay=args.ema_decay)
    ema.model.to(device)
    ema_diffusion = Diffusion(
        ema.model,
        image_resolution=img_size,
        n_times=args.n_timesteps,
        beta_schedule=args.beta_schedule,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    denoising_loss = nn.MSELoss()

    start_epoch = 0
    train_losses = []

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        ema.model.load_state_dict(checkpoint["ema_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        train_losses = checkpoint.get("train_losses", [])
        start_epoch = checkpoint["epoch"]
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    print(f"Parameters: {count_parameters(model):,}")
    print("Start training U-Net DDPM on FashionMNIST...")

    fixed_test_batch = next(iter(test_loader))[0].to(device)
    save_forward_process_grid(diffusion, fixed_test_batch, save_dir / "forward_process.png")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        seen = 0

        progress = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{args.epochs}")
        for x, _ in progress:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                _, epsilon, pred_epsilon = diffusion(x)
                loss = denoising_loss(pred_epsilon, epsilon)

            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            seen += batch_size
            progress.set_postfix(loss=total_loss / seen)

        avg_loss = total_loss / seen
        train_losses.append(avg_loss)
        print(f"\tEpoch {epoch + 1} complete!\tDenoising Loss: {avg_loss:.6f}")

        torch.save({"train": train_losses}, save_dir / "losses.pt")
        save_loss_curve(train_losses, save_dir / "loss_curve.png")

        should_save = (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs
        should_sample = args.sample_every > 0 and ((epoch + 1) % args.sample_every == 0 or (epoch + 1) == args.epochs)

        if should_save:
            save_checkpoint(
                save_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt",
                epoch + 1,
                model,
                ema,
                optimizer,
                scaler,
                train_losses,
                args,
            )

        if should_sample:
            save_generated_samples(
                ema_diffusion,
                save_dir / f"generated_epoch_{epoch + 1:03d}.png",
                args.sample_count,
            )

    save_checkpoint(save_dir / "trained.pt", args.epochs, model, ema, optimizer, scaler, train_losses, args)
    save_generated_samples(ema_diffusion, save_dir / "generated_samples.png", args.sample_count)

    ground_truth = fixed_test_batch[:args.sample_count].detach().cpu()
    save_image(ground_truth, save_dir / "ground_truth_samples.png", nrow=int(math.sqrt(args.sample_count)))
    print("Done!")


if __name__ == "__main__":
    main()
