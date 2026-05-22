"""
cluster_generate.py
-------------------
Entry point for generating images on the cluster after training.
Loads a trained model from a job directory and saves generated images to disk.
Do not call directly — use cluster.py's train_and_generate_on_cluster().
"""

import os
import argparse
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ddpm import NoiseScheduler, UNet, generate_image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--job_dir',       type=str, required=True,
                   help='Path to job directory containing config.pt and trained.pkl')
    p.add_argument('--n_images',      type=int,   default=8)
    p.add_argument('--stochasticity', type=float, default=1.0)
    p.add_argument('--T',             type=int,   default=1000)
    p.add_argument('--ncol',          type=int,   default=4)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    config     = torch.load(os.path.join(args.job_dir, 'config.pt'))
    state_dict = torch.load(os.path.join(args.job_dir, 'trained.pkl'), weights_only=True)

    model = UNet(**config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded model with config: {config}")

    scheduler = NoiseScheduler(T=args.T)
    x = generate_image(model, scheduler, stochasticity=args.stochasticity, n_images=args.n_images)

    # save raw tensor so it can be loaded in the notebook
    tensor_path = os.path.join(args.job_dir, 'generated.pt')
    torch.save(x.cpu(), tensor_path)
    print(f"Saved generated tensor to: {tensor_path}")

    # also save a png for quick visual inspection without loading the notebook
    imgs = x.detach().cpu()
    if imgs.shape[1] in [1, 3]:
        imgs = imgs.permute(0, 2, 3, 1)
    if imgs.shape[-1] == 1:
        imgs = imgs.squeeze(-1)

    import math
    nrow = math.ceil(args.n_images / args.ncol)
    fig, axes = plt.subplots(nrow, args.ncol, figsize=(args.ncol * 3, nrow * 3))
    axes = axes.flatten() if args.n_images > 1 else [axes]

    for i in range(args.n_images):
        img = imgs[i]
        if img.min() < 0:
            img = (img + 1) / 2
        axes[i].imshow(img.clamp(0, 1), cmap='gray' if imgs.ndim == 3 else None)
        axes[i].axis('off')
    for j in range(args.n_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    png_path = os.path.join(args.job_dir, 'generated.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Saved preview image to: {png_path}")


if __name__ == '__main__':
    main()
