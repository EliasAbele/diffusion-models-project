import math
import torch
import matplotlib.pyplot as plt

from .generate import generate_image
from .scheduler import NoiseScheduler


def plot_image(image):
    plt.imshow(image.cpu().detach().squeeze(), cmap="gray")
    plt.colorbar()
    plt.show()


def plot_generated(x: torch.Tensor, ncol=4):
    """Plot a batch of generated images. x shape: (n, 1, H, W)."""
    imgs = x.detach().cpu()
    n_images = imgs.shape[0]

    if imgs.shape[1] in [1, 3]:
        imgs = imgs.permute(0, 2, 3, 1)
    if imgs.shape[-1] == 1:
        imgs = imgs.squeeze(-1)

    nrow = math.ceil(n_images / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3))
    axes = axes.flatten() if n_images > 1 else [axes]

    for i in range(n_images):
        display_img = imgs[i]
        if display_img.min() < 0:
            display_img = (display_img + 1) / 2
        axes[i].imshow(display_img.clamp(0, 1), cmap='gray' if imgs.ndim == 3 else None)
        axes[i].axis('off')

    for j in range(n_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


# def plot_stochasticities(model, scheduler: NoiseScheduler,
#                          stochasticities=[0, 0.33, 0.67, 1.0], ncols=2):
#     nrows = math.ceil(len(stochasticities) / ncols)
#     fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))

#     for ax, s in zip(axes.flat, stochasticities):
#         image = generate_image(model, scheduler, stochasticity=s, n_images=1)
#         ax.imshow(image.cpu().detach().squeeze(), cmap='gray')
#         ax.set_title(f's={s}')
#         ax.axis('off')

#     plt.tight_layout()
#     plt.show()


# def plot_model_comparison(images_list, model_names, ncol=None):
#     for imgs, name in zip(images_list, model_names):
#         imgs = imgs.detach().cpu()
#         n_images = imgs.shape[0]
#         current_ncol = ncol if ncol is not None else n_images
#         nrow = math.ceil(n_images / current_ncol)

#         if imgs.shape[1] in [1, 3]:
#             imgs = imgs.permute(0, 2, 3, 1)
#         if imgs.shape[-1] == 1:
#             imgs = imgs.squeeze(-1)

#         fig, axes = plt.subplots(nrow, current_ncol, figsize=(current_ncol * 2.5, nrow * 2.5))
#         fig.suptitle(f"Model: {name}", fontsize=16, fontweight='bold', y=1.02)
#         axes = axes.flatten() if n_images > 1 else [axes]

#         for i in range(n_images):
#             display_img = imgs[i]
#             if display_img.min() < 0:
#                 display_img = (display_img + 1) / 2
#             axes[i].imshow(display_img.clamp(0, 1), cmap='gray' if imgs.ndim == 3 else None)
#             axes[i].axis('off')

#         for j in range(n_images, len(axes)):
#             axes[j].axis('off')

#         plt.tight_layout()
#         plt.show()
