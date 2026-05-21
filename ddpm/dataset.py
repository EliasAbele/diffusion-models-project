import torch
from torch.utils.data import DataLoader,Subset
from torchvision import datasets, transforms
import torch.nn.functional as F

from .scheduler import NoiseScheduler

from torch.utils.data import Dataset
 
class GalaxyDataset(Dataset):
    """
    Wrapper for Galaxy10 (or any numpy array dataset) to make it compatible 
    with PyTorch Dataset interface.
    
    Args:
        images: numpy array of images with shape (N, H, W, C) or (N, H, W)
        labels: numpy array of labels with shape (N,)
        transform: optional transform to apply to images
    """
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels if labels is not None else np.zeros(len(images))
        self.transform = transform
 
    def __len__(self):
        return len(self.images)
 
    def __getitem__(self, idx):
        # Get image and label
        x = self.images[idx]
        y = self.labels[idx]
 
        # Convert numpy → torch tensor
        x = torch.tensor(x, dtype=torch.float32)
        
        # Normalize to [0, 1] if needed (Galaxy10 is [0, 255])
        if x.max() > 1.0:
            x = x / 255.0
 
        # Convert HWC → CHW format for PyTorch
        if x.ndim == 3 and x.shape[-1] in [1, 3]:  # Check if last dim is channels
            x = x.permute(2, 0, 1)

            x = F.interpolate(
                x.unsqueeze(0),
                size=(64, 64),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Apply optional transform
        if self.transform:
            x = self.transform(x)
 
        return x, y
    
class NoisyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, scheduler: NoiseScheduler):
        self.dataset = dataset
        self.scheduler = scheduler

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        scheduler_device = self.scheduler.betas.device
        x, _ = self.dataset[idx]
        t = torch.randint(0, self.scheduler.T, (1,))
        x_noisy, noise = self.scheduler.add_noise(
            x.unsqueeze(0).to(scheduler_device),
            t.to(scheduler_device)
        )
        return x_noisy.squeeze(0), noise.squeeze(0), t.squeeze(0)
    
NoisyMNIST = NoisyDataset

def load_mnist(transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    train_set = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
    test_set  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train_set, test_set


def get_noisy_loaders(train_set, test_set, scheduler: NoiseScheduler, batch_size=32):
    train_noisy = NoisyDataset(train_set, scheduler)
    test_noisy  = NoisyDataset(test_set,  scheduler)
    train_loader = DataLoader(train_noisy, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_noisy,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def zeros_only(dataset):
    indices = (dataset.targets == 0).nonzero(as_tuple=True)[0]
    return Subset(dataset, indices)

def get_noisy_loaders_filtered(train_set, test_set, scheduler, filter_fn, batch_size=32):
    train_noisy = NoisyDataset(filter_fn(train_set), scheduler)
    test_noisy  = NoisyDataset(filter_fn(test_set),  scheduler)
    train_loader = DataLoader(train_noisy, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_noisy,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader