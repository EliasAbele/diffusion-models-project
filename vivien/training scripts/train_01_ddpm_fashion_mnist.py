"""
Trains a stacked-dilated-convolution DDPM on Fashion-MNIST.
Saves model weights, loss curves, and generated sample images to:
    models/vivien_ddpm_fashion_mnist/
"""

import os
import math
import argparse

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for cluster
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from torchvision.datasets import FashionMNIST
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam

np.Inf = np.inf  # compatibility fix


# Argument parsing (mirrors cluster_train.py style)

def parse_args():
    p = argparse.ArgumentParser(description="Train DDPM (stacked conv) on Fashion-MNIST")
    p.add_argument('--epochs',             type=int,   default=200)
    p.add_argument('--batch_size',         type=int,   default=128)
    p.add_argument('--lr',                 type=float, default=5e-5)
    p.add_argument('--n_timesteps',        type=int,   default=1000)
    p.add_argument('--hidden_dim',         type=int,   default=256)
    p.add_argument('--n_layers',           type=int,   default=8)
    p.add_argument('--timestep_emb_dim',   type=int,   default=256)
    p.add_argument('--dataset_path',       type=str,   default='~/datasets')
    p.add_argument('--save_dir',           type=str,   default='models/vivien_ddpm_fashion_mnist')
    p.add_argument('--seed',               type=int,   default=1234)
    return p.parse_args()


# Model components

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ConvBlock(nn.Conv2d):
    """
    Conv2D Block
        Args:
            x: (N, C_in, H, W)
        Returns:
            y: (N, C_out, H, W)
    """

    def __init__(self, in_channels, out_channels, kernel_size, activation_fn=None,
                 drop_rate=0., stride=1, padding='same', dilation=1, groups=1,
                 bias=True, gn=False, gn_groups=8):

        if padding == 'same':
            padding = kernel_size // 2 * dilation

        super(ConvBlock, self).__init__(in_channels, out_channels, kernel_size,
                                        stride=stride, padding=padding, dilation=dilation,
                                        groups=groups, bias=bias)

        self.activation_fn = nn.SiLU() if activation_fn else None
        self.group_norm = nn.GroupNorm(gn_groups, out_channels) if gn else None

    def forward(self, x, time_embedding=None, residual=False):

        if residual:
            # in the paper, diffusion timestep embedding was only applied to residual blocks of U-Net
            x = x + time_embedding
            y = x
            x = super(ConvBlock, self).forward(x)
            y = y + x
        else:
            y = super(ConvBlock, self).forward(x)
        y = self.group_norm(y) if self.group_norm is not None else y
        y = self.activation_fn(y) if self.activation_fn is not None else y

        return y


class Denoiser(nn.Module):

    def __init__(self, image_resolution, hidden_dims=[256, 256],
                 diffusion_time_embedding_dim=256, n_times=1000):
        super(Denoiser, self).__init__()

        _, _, img_C = image_resolution

        self.time_embedding = SinusoidalPosEmb(diffusion_time_embedding_dim)

        self.in_project = ConvBlock(img_C, hidden_dims[0], kernel_size=7)

        self.time_project = nn.Sequential(
            ConvBlock(diffusion_time_embedding_dim, hidden_dims[0],
                      kernel_size=1, activation_fn=True),
            ConvBlock(hidden_dims[0], hidden_dims[0], kernel_size=1)
        )

        self.convs = nn.ModuleList(
            [ConvBlock(in_channels=hidden_dims[0], out_channels=hidden_dims[0], kernel_size=3)]
        )
        for idx in range(1, len(hidden_dims)):
            self.convs.append(
                ConvBlock(hidden_dims[idx - 1], hidden_dims[idx], kernel_size=3,
                          dilation=3 ** ((idx - 1) // 2),
                          activation_fn=True, gn=True, gn_groups=8)
            )

        self.out_project = ConvBlock(hidden_dims[-1], out_channels=img_C, kernel_size=3)

    def forward(self, perturbed_x, diffusion_timestep):
        y = perturbed_x

        diffusion_embedding = self.time_embedding(diffusion_timestep)
        diffusion_embedding = self.time_project(
            diffusion_embedding.unsqueeze(-1).unsqueeze(-2)
        )

        y = self.in_project(y)

        for i in range(len(self.convs)):
            y = self.convs[i](y, diffusion_embedding, residual=True)

        y = self.out_project(y)
        return y


# Diffusion process

class Diffusion(nn.Module):
    def __init__(self, model, image_resolution, n_times=1000,
                 beta_minmax=[1e-4, 2e-2], device='cpu'):
        super(Diffusion, self).__init__()

        self.n_times = n_times
        self.img_H, self.img_W, self.img_C = image_resolution
        self.model = model

        beta_1, beta_T = beta_minmax
        betas = torch.linspace(start=beta_1, end=beta_T, steps=n_times).to(device)
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
    hidden_dims = [args.hidden_dim] * args.n_layers

    model = Denoiser(
        image_resolution=img_size,
        hidden_dims=hidden_dims,
        diffusion_time_embedding_dim=args.timestep_emb_dim,
        n_times=args.n_timesteps,
    ).to(DEVICE)

    diffusion = Diffusion(
        model, image_resolution=img_size,
        n_times=args.n_timesteps, beta_minmax=[1e-4, 2e-2], device=DEVICE
    ).to(DEVICE)

    optimizer = Adam(diffusion.parameters(), lr=args.lr)
    denoising_loss = nn.MSELoss()

    print(f"Parameters: {count_parameters(diffusion):,}")

    print("Start training DDPM (stacked conv) on Fashion-MNIST...")
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

        # Save checkpoint every 50 epochs (notebook-compatible format)
        if (epoch + 1) % 50 == 0:
            ckpt_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    final_path = os.path.join(args.save_dir, 'trained.pt')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_losses[-1],
    }, final_path)
    torch.save({'train': train_losses}, os.path.join(args.save_dir, 'losses.pt'))
    print(f"Model saved to: {final_path}")

    print("Generating samples...")
    model.eval()
    with torch.no_grad():
        generated_images = diffusion.sample(N=64)

    save_sample_grid(generated_images, os.path.join(args.save_dir, 'generated_samples.png'),
                     "Generated Images (DDPM - stacked conv, Fashion-MNIST)")

    # Forward process visualization (from test set)
    model.eval()
    for x, _ in test_loader:
        x = x.to(DEVICE)
        perturbed_images, _, _ = diffusion(x)
        perturbed_images = diffusion.reverse_scale_to_zero_to_one(perturbed_images)
        break

    save_sample_grid(perturbed_images[:64], os.path.join(args.save_dir, 'perturbed_samples.png'),
                     "Perturbed Images (forward process, Fashion-MNIST)")
    save_sample_grid(x[:64], os.path.join(args.save_dir, 'ground_truth_samples.png'),
                     "Ground-truth Images (Fashion-MNIST)")

    print("Done!")


if __name__ == '__main__':
    main()
