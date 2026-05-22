#!/usr/bin/env python
"""
DDPM Training Script for Galaxy10 Dataset
Train a diffusion model on galaxy images from the Galaxy10 dataset
With Weights & Biases logging
"""

import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import wandb
import os

# Import DDPM components
from ddpm import NoiseScheduler, UNet, generate_image
from ddpm.dataset import NoisyDataset
from ddpm.dataset import GalaxyDataset
from ddpm.utils import channel_list, model_name
from ddpm.viz import plot_generated

# =============================================================================
# CONFIGURATION - MODIFY THESE PARAMETERS
# =============================================================================

# Weights & Biases configuration
WANDB_PROJECT = "galaxy-ddpm"  # Change this to your project name
WANDB_ENTITY = None  # Set to your wandb username/team or leave as None
WANDB_RUN_NAME = "galaxy_C0_128_convs_2_50_epochs"  # Descriptive name for this run

# Model hyperparameters
CHANNEL0 = 128              # Base number of channels
CONVS_PER_LEVEL = 2        # Convolutions per level in UNet
INPUT_CHANNELS = 3         # RGB images from Galaxy10

# Training hyperparameters
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 1e-4       # Can be adjusted based on find_lr results
WEIGHT_DECAY = 1e-6
EARLY_STOPPING_PATIENCE = 10

# Noise scheduler parameters
T = 1000
BETA_START = 1e-4
BETA_END = 0.02

# Data split
TEST_SIZE = 0.1
RANDOM_SEED = 42

# Output paths
SAVE_DIR = 'models'
RESULTS_DIR = 'results'
SAVE_PATH = f'{SAVE_DIR}/galaxy_C0_{CHANNEL0}_convs_{CONVS_PER_LEVEL}.pkl'
LOSS_PATH = f'{RESULTS_DIR}/galaxy_losses.pkl'

# wandb logging frequency
LOG_INTERVAL = 10  # Log every N batches
IMAGE_LOG_INTERVAL = 5  # Generate and log images every N epochs

# =============================================================================
# SETUP
# =============================================================================

# Create output directories
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# =============================================================================
# INITIALIZE WEIGHTS & BIASES
# =============================================================================
print("\n" + "="*80)
print("INITIALIZING WEIGHTS & BIASES")
print("="*80)

# Initialize wandb
wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    name=WANDB_RUN_NAME,
    config={
        "channel0": CHANNEL0,
        "convs_per_level": CONVS_PER_LEVEL,
        "input_channels": INPUT_CHANNELS,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "noise_T": T,
        "beta_start": BETA_START,
        "beta_end": BETA_END,
        "test_size": TEST_SIZE,
        "random_seed": RANDOM_SEED,
    }
)

print(f"wandb run: {wandb.run.name}")
print(f"wandb url: {wandb.run.url}")

# =============================================================================
# 1. LOAD GALAXY10 DATASET
# =============================================================================
print("\n" + "="*80)
print("LOADING GALAXY10 DATASET")
print("="*80)

from astroNN.datasets import load_galaxy10

# Load images and labels (downloads automatically on first run to ~/.astroNN/datasets/)
images, labels = load_galaxy10()

# Convert to float32
labels = labels.astype(np.float32)
images = images.astype(np.float32)

print(f"Dataset loaded successfully")
print(f"  Total samples: {len(images)}")
print(f"  Image shape: {images[0].shape}")
print(f"  Value range: [{images.min():.2f}, {images.max():.2f}]")

# Log dataset info to wandb
wandb.config.update({
    "dataset_size": len(images),
    "image_shape": images[0].shape,
    "value_range": [float(images.min()), float(images.max())]
})

# Save a sample image for verification
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i].astype(int))
    ax.set_title(f"Galaxy {i}")
    ax.axis('off')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/sample_galaxies.png', dpi=150, bbox_inches='tight')
plt.close()

# Log sample images to wandb
wandb.log({"sample_galaxies": wandb.Image(f'{RESULTS_DIR}/sample_galaxies.png')})
print(f"  Saved samples to: {RESULTS_DIR}/sample_galaxies.png")

# =============================================================================
# 2. TRAIN/TEST SPLIT
# =============================================================================
print("\n" + "="*80)
print("CREATING TRAIN/TEST SPLIT")
print("="*80)

train_idx, test_idx = train_test_split(
    np.arange(labels.shape[0]), 
    test_size=TEST_SIZE, 
    random_state=RANDOM_SEED
)

train_images = images[train_idx]
train_labels = labels[train_idx]
test_images = images[test_idx]
test_labels = labels[test_idx]

print(f"Train set: {len(train_images)} samples")
print(f"Test set:  {len(test_images)} samples")

wandb.config.update({
    "train_size_samples": len(train_images),
    "test_size_samples": len(test_images)
})

# =============================================================================
# 3. CREATE NOISY DATASETS
# =============================================================================
print("\n" + "="*80)
print("CREATING NOISY DATASETS")
print("="*80)

scheduler = NoiseScheduler(T=T, beta_start=BETA_START, beta_end=BETA_END)
train_set = GalaxyDataset(train_images, train_labels)
test_set = GalaxyDataset(test_images, test_labels)
train_set_noisy = NoisyDataset(train_images, scheduler)
test_set_noisy = NoisyDataset(test_images, scheduler)

print(f"Noise scheduler configured:")
print(f"  T: {T}")
print(f"  Beta start: {BETA_START}")
print(f"  Beta end: {BETA_END}")

# =============================================================================
# 4. BUILD UNET MODEL
# =============================================================================
print("\n" + "="*80)
print("BUILDING UNET MODEL")
print("="*80)

channels = channel_list(CHANNEL0)
unet = UNet(
    channels=channels, 
    convs_per_level=CONVS_PER_LEVEL, 
    input_channels=INPUT_CHANNELS
).to(device)

model_name_str = model_name(CHANNEL0, CONVS_PER_LEVEL)
num_params = sum(p.numel() for p in unet.parameters())

print(f"Model: {model_name_str}")
print(f"  Total parameters: {num_params:,}")
print(f"  Input channels: {INPUT_CHANNELS}")
print(f"  Base channels: {CHANNEL0}")
print(f"  Convolutions per level: {CONVS_PER_LEVEL}")

wandb.config.update({"num_parameters": num_params})
wandb.watch(unet, log="all", log_freq=100)  # Watch model gradients

# =============================================================================
# 5. CREATE DATA LOADERS
# =============================================================================
print("\n" + "="*80)
print("CREATING DATA LOADERS")
print("="*80)

from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_set_noisy,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_set_noisy,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print(f"Batch size: {BATCH_SIZE}")
print(f"Train batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")

# =============================================================================
# 6. TRAINING SETUP
# =============================================================================
print("\n" + "="*80)
print("SETTING UP TRAINING")
print("="*80)

optimizer = torch.optim.Adam(unet.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = torch.nn.MSELoss()

print(f"Training configuration:")
print(f"  Epochs: {EPOCHS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Weight decay: {WEIGHT_DECAY}")
print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
print(f"  Save path: {SAVE_PATH}")

# =============================================================================
# 7. TRAINING LOOP WITH WANDB LOGGING
# =============================================================================
print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)

train_losses = []
test_losses = []
best_test_loss = float('inf')
patience_counter = 0
global_step = 0
scaler = torch.cuda.amp.GradScaler()

for epoch in range(EPOCHS):
    # Training phase
    unet.train()
    epoch_train_loss = 0.0
    
    for batch_idx, (x_noisy, noise, t) in enumerate(train_loader):
        x_noisy = x_noisy.to(device)
        noise = noise.to(device)
        t = t.to(device)
        
        # Forward pass
        with torch.cuda.amp.autocast():
            noise_pred = unet(x_noisy, t)
            loss = criterion(noise_pred, noise)
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        epoch_train_loss += loss.item()
        global_step += 1
        
        # Log to wandb every LOG_INTERVAL batches
        if batch_idx % LOG_INTERVAL == 0:
            wandb.log({
                "batch_train_loss": loss.item(),
                "epoch": epoch,
                "global_step": global_step
            })
        
        if batch_idx % 50 == 0:
            print(f"  Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}")
    
    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation phase
    unet.eval()
    epoch_test_loss = 0.0
    
    with torch.no_grad():
        for x_noisy, noise, t in test_loader:
            x_noisy = x_noisy.to(device)
            noise = noise.to(device)
            t = t.to(device)
            
            noise_pred = unet(x_noisy, t)
            loss = criterion(noise_pred, noise)
            epoch_test_loss += loss.item()
    
    avg_test_loss = epoch_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    
    # Log epoch metrics to wandb
    wandb.log({
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "test_loss": avg_test_loss,
        "learning_rate": LEARNING_RATE
    })
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}")
    
    # Generate and log sample images periodically
    if (epoch + 1) % IMAGE_LOG_INTERVAL == 0:
        print(f"  Generating sample images...")
        unet.eval()
        with torch.no_grad():
            generated = generate_image(unet, scheduler, stochasticity=1.0, n_images=8)
        
        # Plot generated images
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i, ax in enumerate(axes.flat):
            img = generated[i].cpu().numpy()
            if img.shape[0] == 3:  # CHW format
                img = np.transpose(img, (1, 2, 0))
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_title(f"Generated {i+1}")
            ax.axis('off')
        plt.tight_layout()
        
        img_path = f'{RESULTS_DIR}/generated_epoch_{epoch+1}.png'
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        wandb.log({
            f"generated_images_epoch_{epoch+1}": wandb.Image(img_path),
            "epoch": epoch
        })
        print(f"  Saved generated images to: {img_path}")
    
    # Early stopping check
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        patience_counter = 0
        # Save best model
        torch.save(unet.state_dict(), SAVE_PATH)
        print(f"  ✓ New best model saved (test loss: {best_test_loss:.6f})")
        
        # Save best model to wandb
        wandb.save(SAVE_PATH)
    else:
        patience_counter += 1
        print(f"  No improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})")
        
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

print("\n" + "="*80)
print("TRAINING COMPLETED")
print("="*80)
print(f"Best test loss: {best_test_loss:.6f}")
print(f"Total epochs: {len(train_losses)}")

# =============================================================================
# 8. SAVE LOSSES
# =============================================================================
print("\n" + "="*80)
print("SAVING TRAINING RESULTS")
print("="*80)

torch.save({
    'train': train_losses,
    'test': test_losses,
    'config': {
        'channel0': CHANNEL0,
        'convs_per_level': CONVS_PER_LEVEL,
        'input_channels': INPUT_CHANNELS,
        'batch_size': BATCH_SIZE,
        'epochs': len(train_losses),
        'lr': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'best_test_loss': best_test_loss,
    }
}, LOSS_PATH)

print(f"Losses saved to: {LOSS_PATH}")
wandb.save(LOSS_PATH)

# =============================================================================
# 9. PLOT LOSS CURVES
# =============================================================================
print("\n" + "="*80)
print("PLOTTING LOSS CURVES")
print("="*80)

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(test_losses, label='Test Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MSE Loss', fontsize=12)
plt.title('DDPM Training Loss - Galaxy10 Dataset', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

loss_curve_path = f'{RESULTS_DIR}/loss_curve_galaxy.png'
plt.savefig(loss_curve_path, dpi=150, bbox_inches='tight')
plt.close()

wandb.log({"loss_curve": wandb.Image(loss_curve_path)})
print(f"Loss curve saved to: {loss_curve_path}")

# =============================================================================
# 10. GENERATE FINAL SAMPLE IMAGES
# =============================================================================
print("\n" + "="*80)
print("GENERATING FINAL SAMPLE IMAGES")
print("="*80)

# Load best model
unet.load_state_dict(torch.load(SAVE_PATH))
unet.eval()

with torch.no_grad():
    # Generate multiple samples
    generated = generate_image(unet, scheduler, stochasticity=1.0, n_images=16)

# Plot grid of generated images
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    img = generated[i].cpu().numpy()
    if img.shape[0] == 3:  # CHW format
        img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.axis('off')
plt.tight_layout()

final_samples_path = f'{RESULTS_DIR}/final_generated_galaxies.png'
plt.savefig(final_samples_path, dpi=150, bbox_inches='tight')
plt.close()

wandb.log({"final_generated_samples": wandb.Image(final_samples_path)})
print(f"Final samples saved to: {final_samples_path}")

# Generate denoising process visualization
print("Generating denoising process visualization...")
with torch.no_grad():
    x_final, intermediates = generate_image(
        unet, scheduler, stochasticity=1.0, n_images=1, return_intermediates=True
    )

fig, axes = plt.subplots(1, 11, figsize=(22, 2))
steps_to_show = list(range(0, 1000, 100)) + [999]
for ax, idx in zip(axes, steps_to_show):
    img = intermediates[idx].squeeze().cpu().numpy()
    if img.shape[0] == 3:  # CHW format
        img = np.transpose(img, (1, 2, 0))
    if img.min() < 0:
        img = (img + 1) / 2
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.set_title(f't={1000 - idx}', fontsize=9)
    ax.axis('off')
plt.tight_layout()

denoising_path = f'{RESULTS_DIR}/denoising_process_galaxy.png'
plt.savefig(denoising_path, dpi=150, bbox_inches='tight')
plt.close()

wandb.log({"denoising_process": wandb.Image(denoising_path)})
print(f"Denoising process saved to: {denoising_path}")

# =============================================================================
# 11. FINISH WANDB RUN
# =============================================================================
print("\n" + "="*80)
print("FINALIZING WANDB")
print("="*80)

# Log final summary statistics
wandb.run.summary["final_train_loss"] = train_losses[-1]
wandb.run.summary["final_test_loss"] = test_losses[-1]
wandb.run.summary["best_test_loss"] = best_test_loss
wandb.run.summary["total_epochs"] = len(train_losses)
wandb.run.summary["num_parameters"] = num_params

print(f"wandb run completed: {wandb.run.url}")
wandb.finish()

print("\n" + "="*80)
print("ALL DONE!")
print("="*80)
print(f"Model saved to: {SAVE_PATH}")
print(f"Results saved to: {RESULTS_DIR}/")
print(f"View results at: {wandb.run.url}")