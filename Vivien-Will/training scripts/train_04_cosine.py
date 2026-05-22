"""
Trains a U-Net DDPM on FashionMNIST with a cosine beta-schedule
(Nichol & Dhariwal, "Improved DDPM", 2021).

Ablation #2 over train_02:
  - Cosine beta schedule instead of linear [1e-4, 2e-2].
  - All other hyperparameters identical to train_02 (no EMA, lr=5e-5).
  - Checkpoints saved every 50 epochs.

Saves model weights, loss curves, and generated sample images to:
    models/vivien_lucidrains_unet_fashion_mnist_cosine/
"""

import os
import math
import argparse

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from torchvision.datasets import FashionMNIST
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam

from denoising_diffusion_pytorch import Unet as LucidUnet

np.Inf = np.inf


# Argument parsing

def parse_args():
    p = argparse.ArgumentParser(description="Train U-Net DDPM on FashionMNIST with cosine beta-schedule")
    p.add_argument('--epochs',             type=int,   default=200)
    p.add_argument('--batch_size',         type=int,   default=128)
    p.add_argument('--lr',                 type=float, default=5e-5)
    p.add_argument('--n_timesteps',        type=int,   default=1000)
    p.add_argument('--timestep_emb_dim',   type=int,   default=256)
    p.add_argument('--beta_schedule',      type=str,   default='cosine', choices=['linear', 'cosine'])
    p.add_argument('--cosine_s',           type=float, default=0.008)
    p.add_argument('--dataset_path',       type=str,   default='~/datasets')
    p.add_argument('--save_dir',           type=str,   default='models/vivien_lucidrains_unet_fashion_mnist_cosine')
    p.add_argument('--seed',               type=int,   default=1234)
    return p.parse_args()


# Beta schedules

def cosine_beta_schedule(n_times, s=0.008):
    # Linear schedule destroys signal too fast at 28x28; cosine keeps SNR
    # higher in mid-range t where detail is learned. Improved DDPM, eq. 17:
    # https://arxiv.org/abs/2102.09672
    """Cosine schedule from Nichol & Dhariwal 2021 (eq. 17)."""
    steps = n_times + 1
    t = torch.linspace(0, n_times, steps) / n_times
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-4, 0.999)


def linear_beta_schedule(n_times, beta_1=1e-4, beta_T=2e-2):
    return torch.linspace(start=beta_1, end=beta_T, steps=n_times)


# Diffusion process

class Diffusion(nn.Module):
    def __init__(self, model, image_resolution, n_times=1000,
                 beta_schedule='cosine', beta_minmax=[1e-4, 2e-2],
                 cosine_s=0.008, device='cpu'):
        super(Diffusion, self).__init__()

        self.n_times = n_times
        self.img_H, self.img_W, self.img_C = image_resolution
        self.model = model

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_times, *beta_minmax).to(device)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_times, s=cosine_s).to(device)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        self.sqrt_betas = torch.sqrt(betas)

        self.alphas = 1 - betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)

        self.device = device

    def extract(self, a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def scale_to_minus_one_to_one(self, x):
        return x * 2 - 1

    def reverse_scale_to_zero_to_one(self, x):
        return (x + 1) * 0.5

    def make_noisy(self, x_zeros, t):
        epsilon = torch.randn_like(x_zeros).to(self.device)
        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, t, x_zeros.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, t, x_zeros.shape)
        noisy_sample = x_zeros * sqrt_alpha_bar + epsilon * sqrt_one_minus_alpha_bar
        return noisy_sample.detach(), epsilon

    def forward(self, x_zeros):
        x_zeros = self.scale_to_minus_one_to_one(x_zeros)
        B, _, _, _ = x_zeros.shape
        t = torch.randint(low=0, high=self.n_times, size=(B,)).long().to(self.device)
        perturbed_images, epsilon = self.make_noisy(x_zeros, t)
        pred_epsilon = self.model(perturbed_images, t)
        return perturbed_images, epsilon, pred_epsilon

    def denoise_at_t(self, x_t, timestep, t):
        B, _, _, _ = x_t.shape
        z = torch.randn_like(x_t).to(self.device) if t > 1 else torch.zeros_like(x_t).to(self.device)
        epsilon_pred = self.model(x_t, timestep)
        alpha = self.extract(self.alphas, timestep, x_t.shape)
        sqrt_alpha = self.extract(self.sqrt_alphas, timestep, x_t.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep, x_t.shape)
        sqrt_beta = self.extract(self.sqrt_betas, timestep, x_t.shape)
        x_t_minus_1 = 1 / sqrt_alpha * (x_t - (1 - alpha) / sqrt_one_minus_alpha_bar * epsilon_pred) + sqrt_beta * z
        return x_t_minus_1.clamp(-1., 1.)

    def sample(self, N):
        x_t = torch.randn((N, self.img_C, self.img_H, self.img_W)).to(self.device)
        for t in range(self.n_times - 1, -1, -1):
            timestep = torch.tensor([t]).repeat_interleave(N, dim=0).long().to(self.device)
            x_t = self.denoise_at_t(x_t, timestep, t)
        return self.reverse_scale_to_zero_to_one(x_t)


# Helpers

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_sample_grid(images, path, title):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(title)
    plt.imshow(np.transpose(make_grid(images.detach().cpu(), padding=2, normalize=True), (1, 2, 0)))
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# Main

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")
    print(f"Config: {vars(args)}")

    os.makedirs(args.save_dir, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = FashionMNIST(args.dataset_path, transform=transform, train=True,  download=True)
    test_dataset  = FashionMNIST(args.dataset_path, transform=transform, train=False, download=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False,
                              num_workers=4, pin_memory=True)

    img_size = (28, 28, 1)

    model = LucidUnet(
        dim=32,
        dim_mults=(1, 2, 4),
        channels=1,
        flash_attn=False,
    ).to(DEVICE)

    diffusion = Diffusion(
        model, image_resolution=img_size, n_times=args.n_timesteps,
        beta_schedule=args.beta_schedule, beta_minmax=[1e-4, 2e-2],
        cosine_s=args.cosine_s, device=DEVICE,
    ).to(DEVICE)

    optimizer = Adam(diffusion.parameters(), lr=args.lr)
    denoising_loss = nn.MSELoss()

    print(f"Parameters: {count_parameters(diffusion):,}")
    print(f"Beta schedule: {args.beta_schedule} (cosine_s={args.cosine_s})")

    print("Start training U-Net DDPM (+ cosine schedule) on FashionMNIST...")
    train_losses = []

    for epoch in range(args.epochs):
        model.train()
        noise_prediction_loss = 0.0

        for batch_idx, (x, _) in tqdm(enumerate(train_loader), total=len(train_loader),
                                       desc=f"Epoch {epoch + 1}/{args.epochs}"):
            optimizer.zero_grad()
            x = x.to(DEVICE)
            _, epsilon, pred_epsilon = diffusion(x)
            loss = denoising_loss(pred_epsilon, epsilon)
            noise_prediction_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = noise_prediction_loss / (batch_idx + 1)
        train_losses.append(avg_loss)
        print(f"\tEpoch {epoch + 1} complete!\tDenoising Loss: {avg_loss:.6f}")

        if (epoch + 1) % 50 == 0:
            ckpt_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    torch.save(model.state_dict(), os.path.join(args.save_dir, 'trained.pt'))
    torch.save({'train': train_losses}, os.path.join(args.save_dir, 'losses.pt'))
    print(f"Model saved to: {args.save_dir}/trained.pt")

    print("Generating samples...")
    model.eval()
    with torch.no_grad():
        generated_images = diffusion.sample(N=64)

    save_sample_grid(generated_images, os.path.join(args.save_dir, 'generated_samples.png'),
                     "Generated Images (U-Net DDPM + cosine - FashionMNIST)")

    model.eval()
    for x, _ in test_loader:
        x = x.to(DEVICE)
        perturbed_images, _, _ = diffusion(x)
        perturbed_images = diffusion.reverse_scale_to_zero_to_one(perturbed_images)
        break

    save_sample_grid(perturbed_images[:64], os.path.join(args.save_dir, 'perturbed_samples.png'),
                     "Perturbed Images (forward process - cosine)")
    save_sample_grid(x[:64], os.path.join(args.save_dir, 'ground_truth_samples.png'),
                     "Ground-truth FashionMNIST Images")

    print("Done!")


if __name__ == '__main__':
    main()
