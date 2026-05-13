import torch
from .scheduler import NoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def noisy_image(n=1, img_shape=(28, 28)):
    """Generates a batch of n pure-noise images. Output shape: (n, 1, H, W)."""
    return torch.randn((n, 1, *img_shape))


def generate_image(unet, scheduler: NoiseScheduler, stochasticity=1.0, n_images=1,
                   return_intermediates=False):
    """
    Runs the full reverse diffusion chain and returns the denoised image tensor.

    Args:
        unet:               trained UNet model
        scheduler:          NoiseScheduler instance
        stochasticity:      scale factor on sigma_t (0 = DDIM-like, 1 = full DDPM)
        n_images:           number of images to generate in parallel
        return_intermediates: if True, also returns list of intermediate x tensors

    Returns:
        x (Tensor):         final denoised images, shape (n_images, 1, H, W)
        intermediates (list, optional): list of cpu tensors at each step
    """
    intermediates = []
    unet.eval()

    x = noisy_image(n_images).to(device)
    if return_intermediates:
        intermediates.append(x.cpu())

    with torch.no_grad():
        for t in range(scheduler.T - 1, 0, -1):
            z = noisy_image(n_images).to(device) if t > 1 else torch.zeros_like(x)
            alpha     = scheduler.alpha(t)
            alpha_bar = scheduler.alpha_bar(t)
            sigma_t   = stochasticity * torch.sqrt(scheduler.beta(t))

            noise_pred = unet(x)
            x = (1 / torch.sqrt(alpha)) * (
                x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * noise_pred
            ) + sigma_t * z

            if return_intermediates:
                intermediates.append(x.cpu())

    if return_intermediates:
        return x, intermediates
    return x
