"""
Trains a U-Net DDPM on FashionMNIST using lucidrains' GaussianDiffusion with an
intermediate recipe: U-Net + GaussianDiffusion + cosine schedule, but without
EMA, v-prediction, or Min-SNR loss weighting.

Purpose of this file:

  This is the "Model B" ablation between the original hand-written DDPM and the
  strongest train_06_v_min_snr.py recipe. It keeps the parts that make the
  implementation cleaner and more stable than the original code, while removing
  the three modern improvements that likely gave train_06 its largest quality
  jump.

Differences vs. train_06_v_min_snr.py:

  1. No v-prediction.
     Uses objective='pred_noise', i.e. the classic Ho et al. 2020 epsilon/noise
     prediction target.

  2. No Min-SNR-gamma loss weighting.
     Uses the default unweighted diffusion loss. This should make the experiment
     visibly weaker than train_06 while still benefiting from the lucidrains
     implementation and U-Net architecture.

  3. No EMA.
     The model sampled during training and at the end is the raw training model,
     not an exponential moving average of the weights.

Kept from train_06_v_min_snr.py:

  1. GaussianDiffusion from lucidrains instead of the hand-written Diffusion class.
     This keeps the forward/reverse process and sampler consistent.

  2. Cosine beta-schedule.
     This is kept because it is a clean intermediate improvement over the earlier
     linear schedule and helps FashionMNIST without being as strong as the full
     v-prediction + Min-SNR + EMA recipe.

  3. dim=64 U-Net.
     The architecture is kept comparable so this ablation isolates the training
     objective / loss-weighting / EMA changes.

  4. Pad 28 -> 32 with transforms.Pad(2).
     Powers of two play nicer with the U-Net's 3 downsampling levels; 28/8 = 3.5
     forces awkward rounding.

  5. Periodic sample grids every --sample_every epochs.
     This makes it easy to compare training progress against train_06.

Saves to: models/vivien_unet_fashion_model_b_no_ema_no_v_no_min_snr/
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.utils import make_grid
from tqdm import tqdm

from denoising_diffusion_pytorch import GaussianDiffusion, Unet

np.Inf = np.inf  # compat shim for older numpy-using deps


def parse_args():
    p = argparse.ArgumentParser(description="Train intermediate epsilon-prediction DDPM on FashionMNIST")
    p.add_argument('--epochs',          type=int,   default=200)
    p.add_argument('--batch_size',      type=int,   default=128)
    p.add_argument('--lr',              type=float, default=1e-4)
    p.add_argument('--n_timesteps',     type=int,   default=1000)
    p.add_argument('--dim',             type=int,   default=64)
    p.add_argument('--sample_every',    type=int,   default=10,
                   help='Save a raw-model sample grid every N epochs.')
    p.add_argument('--ckpt_every',      type=int,   default=50)
    p.add_argument('--dataset_path',    type=str,   default='~/datasets')
    p.add_argument('--save_dir',        type=str,   default='models/vivien_unet_fashion_model_b_no_ema_no_v_no_min_snr')
    p.add_argument('--seed',            type=int,   default=1234)
    return p.parse_args()


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

    # Pad 28 -> 32 so the U-Net depth math is clean.
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

    # Model B: lucidrains diffusion wrapper + U-Net + cosine schedule, but classic
    # epsilon prediction, no Min-SNR loss weighting, and no EMA.
    diffusion = GaussianDiffusion(
        model,
        image_size=32,
        timesteps=args.n_timesteps,
        objective='pred_noise',
        beta_schedule='cosine',
        min_snr_loss_weight=False,
        auto_normalize=True,
    ).to(device)

    optimizer = AdamW(diffusion.parameters(), lr=args.lr)

    print(f"Parameters: {count_parameters(diffusion):,}")
    print("Start training Model B (epsilon prediction + cosine, no Min-SNR, no EMA)...")

    train_losses = []

    for epoch in range(args.epochs):
        diffusion.train()
        running_loss = 0.0
        n_batches = 0

        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad()
            # GaussianDiffusion.forward returns the scalar diffusion loss.
            loss = diffusion(x)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg = running_loss / max(n_batches, 1)
        train_losses.append(avg)
        print(f"\tEpoch {epoch + 1}: loss={avg:.6f}")

        # Mid-training sample grids from the raw model.
        if (epoch + 1) % args.sample_every == 0 or epoch == 0:
            diffusion.eval()
            with torch.no_grad():
                samples = diffusion.sample(batch_size=64)
            save_sample_grid(
                samples,
                os.path.join(samples_dir, f'samples_epoch_{epoch + 1:03d}.png'),
                f"Raw samples - epoch {epoch + 1}",
            )

        if (epoch + 1) % args.ckpt_every == 0:
            torch.save(diffusion.state_dict(),
                       os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1}.pt'))
            torch.save({'train': train_losses},
                       os.path.join(args.save_dir, 'losses.pt'))

    # Final artifacts
    torch.save(diffusion.state_dict(),  os.path.join(args.save_dir, 'trained.pt'))
    torch.save({'train': train_losses}, os.path.join(args.save_dir, 'losses.pt'))

    print("Generating final samples...")
    diffusion.eval()
    with torch.no_grad():
        raw_samples = diffusion.sample(batch_size=64)
    save_sample_grid(raw_samples, os.path.join(args.save_dir, 'generated_samples_raw.png'),
                     "Generated Images (epsilon + cosine, no Min-SNR, no EMA)")

    # Loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses)
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.title('Training loss')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'loss_curve.png'))
    plt.close()

    print("Done!")


if __name__ == '__main__':
    main()
