"""
cluster_train.py
----------------
Fully general SLURM entry point. Loads a model and datasets from disk,
wraps datasets in DataLoaders, trains, and saves the trained state dict.
No assumptions about architecture or data — do not call directly.
"""

import argparse
import torch
import matplotlib
import os
matplotlib.use('Agg')

from torch.utils.data import DataLoader
from ddpm import find_lr, train


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path',        type=str,   required=True)
    p.add_argument('--train_dataset_path',type=str,   required=True)
    p.add_argument('--test_dataset_path', type=str,   required=True)
    p.add_argument('--epochs',            type=int,   default=50)
    p.add_argument('--batch_size',        type=int,   default=32)
    p.add_argument('--weight_decay',      type=float, default=1e-4)
    p.add_argument('--lr',                type=float, default=None)
    p.add_argument('--save_path',         type=str,   default='model_trained.pkl')
    p.add_argument('--early_stopping',    type=int,   default=10)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Config: {vars(args)}")

    model = torch.load(args.model_path, map_location=device,weights_only=False)
    model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_dataset = torch.load(args.train_dataset_path,weights_only=False)
    test_dataset  = torch.load(args.test_dataset_path,weights_only=False)
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    if args.lr is None:
        suggested_lr = find_lr(model, train_loader)
        lr = suggested_lr * 0.5
        print(f"LR finder: {suggested_lr:.2e} → using {lr:.2e}")
    else:
        lr = args.lr

    train_losses, test_losses = train(
        model, train_loader, test_loader,
        epochs=args.epochs,
        lr=lr,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping,
        save_path=args.save_path,
    )
    job_dir = os.path.dirname(args.save_path)

    torch.save(model.config,        os.path.join(job_dir, 'config.pt'))
    torch.save(model.state_dict(),  args.save_path)
    torch.save({'train': train_losses, 'test': test_losses},
                                    os.path.join(job_dir, 'losses.pt'))

    print(f"Best test loss: {min(test_losses):.4f}")
    print(f"Trained model saved to: {args.save_path}")

    # clean up init files
    os.remove(args.model_path)
    os.remove(args.train_dataset_path)
    os.remove(args.test_dataset_path)
    print("Cleaned up init files.")


if __name__ == '__main__':
    main()
