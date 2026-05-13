import torch
from torch.utils.data import DataLoader,Subset
from torchvision import datasets, transforms

from .scheduler import NoiseScheduler


class NoisyMNIST(torch.utils.data.Dataset):
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
        return x_noisy.squeeze(0), noise.squeeze(0)


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
    train_noisy = NoisyMNIST(train_set, scheduler)
    test_noisy  = NoisyMNIST(test_set,  scheduler)
    train_loader = DataLoader(train_noisy, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_noisy,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def zeros_only(dataset):
    indices = (dataset.targets == 0).nonzero(as_tuple=True)[0]
    return Subset(dataset, indices)

def get_noisy_loaders_filtered(train_set, test_set, scheduler, filter_fn, batch_size=32):
    train_noisy = NoisyMNIST(filter_fn(train_set), scheduler)
    test_noisy  = NoisyMNIST(filter_fn(test_set),  scheduler)
    train_loader = DataLoader(train_noisy, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_noisy,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
