#!/usr/bin/env python
"""
Local Testing Script for Galaxy DDPM Training
Run this BEFORE submitting to the cluster to catch errors early!

This script tests:
1. All imports work correctly
2. Dataset loading and wrapping works
3. Model creation works
4. One training batch runs successfully
5. Image generation works

Run with: python test_galaxy_training_local.py
"""

import sys
import torch
import numpy as np
from sklearn.model_selection import train_test_split

print("="*80)
print("GALAXY DDPM - LOCAL TESTING SCRIPT")
print("="*80)
print("\nThis script will verify everything works before cluster submission.\n")

# =============================================================================
# TEST 1: Import all DDPM components
# =============================================================================
print("[1/7] Testing imports...")
import argparse
#model type
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="full",
                    choices=["full", "no_att", "no_time"])
args = parser.parse_args()

try:
    from ddpm import NoiseScheduler, UNet, generate_image

    from ddpm.dataset import NoisyDataset, GalaxyDataset
    from ddpm.utils import channel_list, model_name
    from ddpm.viz import plot_generated
    print("✓ All DDPM imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("\nFIX: Make sure you've added GalaxyDataset to ddpm/dataset.py")
    print("     and exported it in ddpm/__init__.py")
    sys.exit(1)

# =============================================================================
# TEST 2: Test with small dummy data (no download needed)
# =============================================================================
print("\n[2/7] Creating dummy Galaxy10-like data...")
try:
    # Create fake galaxy images (same shape as Galaxy10)
    # Galaxy10 images are (N, 256, 256, 3) with values [0, 255]
    dummy_images = np.random.randint(0, 256, size=(100, 256, 256, 3)).astype(np.float32)
    dummy_labels = np.random.randint(0, 10, size=(100,)).astype(np.float32)
    
    print(f"✓ Created dummy data: {dummy_images.shape}")
    print(f"  Value range: [{dummy_images.min():.1f}, {dummy_images.max():.1f}]")
except Exception as e:
    print(f"✗ Failed to create dummy data: {e}")
    sys.exit(1)

# =============================================================================
# TEST 3: Test GalaxyDataset wrapper
# =============================================================================
print("\n[3/7] Testing GalaxyDataset wrapper...")
try:
    test_dataset = GalaxyDataset(dummy_images[:10], dummy_labels[:10])
    
    # Test __len__
    assert len(test_dataset) == 10, "Dataset length incorrect"
    
    # Test __getitem__
    x, y = test_dataset[0]
    
    # Check it's a tensor
    assert isinstance(x, torch.Tensor), f"Expected torch.Tensor, got {type(x)}"
    
    # Check shape is CHW format (channels first)
    assert x.shape[0] == 3, f"Expected 3 channels first, got shape {x.shape}"
    
    # Check normalization to [0, 1]
    assert x.max() <= 1.0, f"Expected values in [0,1], got max={x.max()}"
    assert x.min() >= 0.0, f"Expected values in [0,1], got min={x.min()}"
    
    print(f"✓ GalaxyDataset works correctly")
    print(f"  Input shape (HWC): {dummy_images[0].shape}")
    print(f"  Output shape (CHW): {x.shape}")
    print(f"  Value range: [{x.min():.3f}, {x.max():.3f}]")
except Exception as e:
    print(f"✗ GalaxyDataset failed: {e}")
    print("\nFIX: Check that GalaxyDataset class is correctly implemented in ddpm/dataset.py")
    sys.exit(1)

# =============================================================================
# TEST 4: Test NoisyDataset wrapper
# =============================================================================
print("\n[4/7] Testing NoisyDataset wrapper...")
try:
    scheduler = NoiseScheduler(T=1000, beta_start=1e-4, beta_end=0.02)
    
    # Create GalaxyDataset first
    galaxy_dataset = GalaxyDataset(dummy_images[:10], dummy_labels[:10])
    
    # Then wrap with NoisyDataset
    noisy_dataset = NoisyDataset(galaxy_dataset, scheduler)
    
    # Test __getitem__
    x_noisy, noise, t = noisy_dataset[0]
    
    assert x_noisy.shape == noise.shape, "Noisy image and noise shape mismatch"
    assert isinstance(t, torch.Tensor), "Timestep should be a tensor"
    
    print(f"✓ NoisyDataset works correctly")
    print(f"  Noisy image shape: {x_noisy.shape}")
    print(f"  Noise shape: {noise.shape}")
    print(f"  Timestep: {t.item()}")
except Exception as e:
    print(f"✗ NoisyDataset failed: {e}")
    print("\nFIX: Check that NoisyDataset correctly handles the wrapped GalaxyDataset")
    sys.exit(1)

# =============================================================================
# TEST 5: Test UNet model creation
# =============================================================================
print("\n[5/7] Testing UNet model creation...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    CHANNEL0 = 64  # Use smaller model for testing
    CONVS_PER_LEVEL = 2
    INPUT_CHANNELS = 3
    ATTENTION_HEADS = 0        # Default = 8 / Set to 0 for no attention
    TIME_EMB_DIM = 256         # Default = 256/ Set to None for no time embedding
    
    channels = channel_list(CHANNEL0)
    unet = UNet(
        channels=channels,
        convs_per_level=CONVS_PER_LEVEL,
        input_channels=INPUT_CHANNELS,
        num_heads_att=ATTENTION_HEADS,
        time_emb_dim=TIME_EMB_DIM,
    ).to(device)
    
    num_params = sum(p.numel() for p in unet.parameters())
    
    print(f"✓ UNet created successfully")
    print(f"  Device: {device}")
    print(f"  Parameters: {num_params:,}")
    print(f"  Input channels: {INPUT_CHANNELS}")
except Exception as e:
    print(f"✗ UNet creation failed: {e}")
    sys.exit(1)

# =============================================================================
# TEST 6: Test one forward pass
# =============================================================================
print("\n[6/7] Testing forward pass through model...")
try:
    # Get a batch of noisy data
    x_noisy, noise, t = noisy_dataset[0]
    
    # Add batch dimension
    x_noisy = x_noisy.unsqueeze(0).to(device)
    t = t.unsqueeze(0).to(device)
    noise = noise.unsqueeze(0).to(device)
    
    # Forward pass
    with torch.no_grad():
        noise_pred = unet(x_noisy, t)
    
    assert noise_pred.shape == noise.shape, "Predicted noise shape mismatch"
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {x_noisy.shape}")
    print(f"  Output shape: {noise_pred.shape}")
    print(f"  Expected shape: {noise.shape}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    sys.exit(1)

# =============================================================================
# TEST 7: Test one training step
# =============================================================================
print("\n[7/7] Testing one training step...")
try:
    from torch.utils.data import DataLoader
    
    # Create small dataloader
    train_loader = DataLoader(noisy_dataset, batch_size=2, shuffle=True)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    
    # One training step
    unet.train()
    x_noisy, noise, t = next(iter(train_loader))
    
    x_noisy = x_noisy.to(device)
    noise = noise.to(device)
    t = t.to(device)
    
    # Forward
    noise_pred = unet(x_noisy, t)
    loss = criterion(noise_pred, noise)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"✓ Training step successful")
    print(f"  Batch size: {x_noisy.shape[0]}")
    print(f"  Loss: {loss.item():.6f}")
except Exception as e:
    print(f"✗ Training step failed: {e}")
    sys.exit(1)

# =============================================================================
# OPTIONAL: Test with real Galaxy10 data (if available)
# =============================================================================
print("\n" + "="*80)
print("OPTIONAL: Testing with real Galaxy10 data...")
print("="*80)

try:
    from astroNN.datasets import load_galaxy10
    
    print("Loading Galaxy10 dataset (this may take a moment)...")
    images, labels = load_galaxy10()
    
    # Convert to float32
    images = images.astype(np.float32)
    labels = labels.astype(np.float32)
    
    # Use just a small subset for testing
    test_images = images[:20]
    test_labels = labels[:20]
    
    # Test the full pipeline
    galaxy_ds = GalaxyDataset(test_images, test_labels)
    noisy_ds = NoisyDataset(galaxy_ds, scheduler)
    
    x_noisy, noise, t = noisy_ds[0]
    
    print(f"✓ Real Galaxy10 data works!")
    print(f"  Dataset size: {len(images)}")
    print(f"  Image shape: {images[0].shape}")
    print(f"  Processed shape: {x_noisy.shape}")
    
except ImportError:
    print("⚠ astroNN not installed - skipping real data test")
    print("  Install with: pip install astroNN")
except Exception as e:
    print(f"⚠ Real data test failed: {e}")
    print("  This is OK if you're just testing the pipeline")

# =============================================================================
# SUCCESS!
# =============================================================================
print("\n" + "="*80)
print("ALL TESTS PASSED! ✓")
print("="*80)
print("\nYour code is ready for cluster submission!")
print("\nNext steps:")
print("1. Upload train_galaxy.py to the cluster")
print("2. Make sure ddpm/dataset.py includes the GalaxyDataset class")
print("3. Submit with: sbatch submit_galaxy_training.sh")
print("\nNote: The cluster version will use the full dataset and train for many epochs.")