"""
Trains a U-Net DDPM on FashionMNIST using lucidrains' GaussianDiffusion with the
modern recipe: v-prediction + cosine schedule + Min-SNR loss weighting + EMA.

Differences vs. train_02..train_05 and where each change comes from:

  1. GaussianDiffusion from lucidrains instead of the hand-written Diffusion class.
     The previous scripts wrapped lucidrains' Unet with a vanilla Ho et al. 2020
     DDPM. GaussianDiffusion bundles the same forward/reverse process plus the
     loss weighting / parameterization / sampler that the field standardized on
     after 2022.  https://github.com/lucidrains/denoising-diffusion-pytorch

  2. v-prediction objective. Better-conditioned target than eps at the high-noise
     end of the schedule, which is the regime that dominated MSE in train_02..05.
     Salimans & Ho 2022 (Progressive Distillation), Appendix D.
     https://arxiv.org/abs/2202.00512

  3. Cosine beta-schedule. Same idea as train_04 but plugged into GaussianDiffusion
     so it composes with the other changes.
     Nichol & Dhariwal 2021, Improved DDPM.  https://arxiv.org/abs/2102.09672

  4. Min-SNR-gamma loss weighting (gamma=5). Reweights per-timestep loss so the
     low-noise / high-SNR regime gets enough gradient. In train_02..05 a single
     unweighted MSE meant fine-detail steps were drowned out by the trivial
     high-noise steps -- the most likely reason none of those ablations produced
     class-identifiable items.  Hang et al. 2023.
     https://arxiv.org/abs/2303.09556

  5. dim=64 (was 32). Lucidrains' Trainer defaults for this dataset size.

  6. Pad 28 -> 32 with transforms.Pad(2). Powers of two play nicer with the
     U-Net's 3 downsampling levels; 28/8 = 3.5 forces awkward rounding.

  7. EMA on the whole GaussianDiffusion module, sampled from at the end and in
     the periodic mid-training grids. Same pattern as train_03 but applied to
     the wrapped module so the buffers (alpha_bars etc.) come along.

  8. Periodic sample grids (every --sample_every epochs). Train_02..05 only
     emitted one grid at epoch 200, so plateau vs. ongoing progress was
     indistinguishable from the artifacts alone.

Saves to: models/vivien_unet_fashion_v_min_snr/
"""

import argparse
import copy
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.utils import make_grid
from tqdm import tqdm

from denoising_diffusion_pytorch import GaussianDiffusion, Unet

np.Inf = np.inf 


def parse_args():
    p = argparse.ArgumentParser(description="Train v-prediction DDPM on FashionMNIST")
    p.add_argument('--epochs',          type=int,   default=200)
    p.add_argument('--batch_size',      type=int,   default=128)
    p.add_argument('--lr',              type=float, default=1e-4)
    p.add_argument('--n_timesteps',     type=int,   default=1000)
    p.add_argument('--dim',             type=int,   default=64)
    p.add_argument('--ema_decay',       type=float, default=0.995)
    p.add_argument('--ema_update_every', type=int,  default=10)
    p.add_argument('--min_snr_gamma',   type=float, default=5.0)
    p.add_argument('--sample_every',    type=int,   default=10,
                   help='Save a sample grid (EMA) every N epochs.')
    p.add_argument('--ckpt_every',      type=int,   default=50)
    p.add_argument('--dataset_path',    type=str,   default='~/datasets')
    p.add_argument('--save_dir',        type=str,   default='models/vivien_unet_fashion_v_min_snr')
    p.add_argument('--seed',            type=int,   default=1234)
    return p.parse_args()


# Simple EMA over an nn.Module; same shape as train_03_ema.py but wraps the
# whole GaussianDiffusion (which carries the U-Net + schedule buffers).
class EMA:
    def __init__(self, module, decay=0.995, update_every=10):
        self.decay = decay
        self.update_every = update_every
        self.step = 0
        self.ema_module = copy.deepcopy(module)
        self.ema_module.eval()
        for p in self.ema_module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, module):
        self.step += 1
        if self.step % self.update_every != 0:
            return
        for ep, p in zip(self.ema_module.parameters(), module.parameters()):
            ep.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)
        for eb, b in zip(self.ema_module.buffers(), module.buffers()):
            eb.data.copy_(b.data)


def save_sample_grid(images, path, title):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(title)
    plt.imshow(np.transpose(make_grid(images.detach().cpu().clamp(0, 1), padding=2), (1, 2, 0)))
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Config: {vars(args)}")

    os.makedirs(args.save_dir, exist_ok=True)
    samples_dir = os.path.join(args.save_dir, 'samples_during_training')
    os.makedirs(samples_dir, exist_ok=True)

    # Pad 28 -> 32 so the U-Net depth math is clean
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(2),
    ])
    train_dataset = FashionMNIST(args.dataset_path, transform=transform, train=True, download=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)

    model = Unet(
        dim=args.dim,
        dim_mults=(1, 2, 4),
        channels=1,
        flash_attn=False,
    )

    # GaussianDiffusion: ties together the U-Net + schedule + parameterization +
    # loss weighting. Handles its own [0,1] <-> [-1,1] scaling internally
    diffusion = GaussianDiffusion(
        model,
        image_size=32,
        timesteps=args.n_timesteps,
        objective='pred_v',
        beta_schedule='cosine',
        min_snr_loss_weight=True,
        min_snr_gamma=args.min_snr_gamma,
        auto_normalize=True,
    ).to(device)

    ema = EMA(diffusion, decay=args.ema_decay, update_every=args.ema_update_every)
    ema.ema_module.to(device)

    optimizer = AdamW(diffusion.parameters(), lr=args.lr)

    print(f"Parameters: {count_parameters(diffusion):,}")
    print("Start training (v-prediction + cosine + Min-SNR + EMA)...")

    train_losses = []

    for epoch in range(args.epochs):
        diffusion.train()
        running_loss = 0.0
        n_batches = 0

        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad()
            # GaussianDiffusion.forward returns the (already-weighted) scalar loss
            loss = diffusion(x)
            loss.backward()
            optimizer.step()
            ema.update(diffusion)

            running_loss += loss.item()
            n_batches += 1

        avg = running_loss / max(n_batches, 1)
        train_losses.append(avg)
        print(f"\tEpoch {epoch + 1}: loss={avg:.6f}")

        # Mid-training sample grids from the EMA model
        if (epoch + 1) % args.sample_every == 0 or epoch == 0:
            ema.ema_module.eval()
            with torch.no_grad():
                samples = ema.ema_module.sample(batch_size=64)
            save_sample_grid(
                samples,
                os.path.join(samples_dir, f'samples_epoch_{epoch + 1:03d}.png'),
                f"EMA samples - epoch {epoch + 1}",
            )

        if (epoch + 1) % args.ckpt_every == 0:
            torch.save(diffusion.state_dict(),
                       os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1}.pt'))
            torch.save(ema.ema_module.state_dict(),
                       os.path.join(args.save_dir, f'checkpoint_ema_epoch_{epoch + 1}.pt'))
            torch.save({'train': train_losses},
                       os.path.join(args.save_dir, 'losses.pt'))

    # Final artifacts
    torch.save(diffusion.state_dict(),         os.path.join(args.save_dir, 'trained.pt'))
    torch.save(ema.ema_module.state_dict(),    os.path.join(args.save_dir, 'trained_ema.pt'))
    torch.save({'train': train_losses},        os.path.join(args.save_dir, 'losses.pt'))

    # Final grids: EMA and raw, side-by-side for sanity check. The EMA samples should be better, but the raw ones should still be class-identifiable and not pure noise.
    print("Generating final samples...")
    ema.ema_module.eval()
    diffusion.eval()
    with torch.no_grad():
        ema_samples = ema.ema_module.sample(batch_size=64)
        raw_samples = diffusion.sample(batch_size=64)
    save_sample_grid(ema_samples, os.path.join(args.save_dir, 'generated_samples_ema.png'),
                     "Generated Images (v-pred + cosine + Min-SNR, EMA)")
    save_sample_grid(raw_samples, os.path.join(args.save_dir, 'generated_samples_raw.png'),
                     "Generated Images (v-pred + cosine + Min-SNR, raw)")

    # Loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses)
    plt.xlabel('epoch'); plt.ylabel('loss (weighted)'); plt.title('Training loss')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'loss_curve.png'))
    plt.close()

    print("Done!")


if __name__ == '__main__':
    main()
