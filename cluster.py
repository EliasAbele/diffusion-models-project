"""
cluster.py
----------
Provides train_on_cluster(), which serializes any nn.Module and any
torch Datasets, writes a SLURM job script, and submits it.
The cluster reconstructs everything from disk, trains, and saves weights.

Example
-------
from cluster import train_on_cluster, job_status
from ddpm import UNet, NoiseScheduler
from ddpm.dataset import load_mnist, NoisyMNIST
from ddpm.utils import channel_list

scheduler = NoiseScheduler()
train_set, test_set = load_mnist()
train_dataset = NoisyMNIST(train_set, scheduler)
test_dataset  = NoisyMNIST(test_set,  scheduler)

unet = UNet(channel_list(128), convs_per_level=2)

job_id, save_path = train_on_cluster(
    model=unet,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    epochs=100,
    weight_decay=1e-6,
    job_name='unet_128_long',
)

# check status while waiting
job_status(job_id)

# once done, load trained weights back
import torch
unet.load_state_dict(torch.load(save_path))
"""

import os
import subprocess
import textwrap
import torch


PROJECT_DIR = '/home/scur0036/diffusion-models-project'
CONDA_ENV   = 'diffusion'
LOGS_DIR    = os.path.join(PROJECT_DIR, 'logs')
MODELS_DIR  = os.path.join(PROJECT_DIR, 'models')


def train_on_cluster(
    model,
    train_dataset,
    test_dataset,
    epochs          = 50,
    batch_size      = 32,
    weight_decay    = 1e-4,
    lr              = None,       # None = LR finder on the node
    early_stopping  = 10,
    time            = '04:00:00',
    mem             = '32G',
    cpus            = 4,
    job_name        = 'ddpm_job',
    save_path       = None,
):
    """
    Serialize model and datasets, write a SLURM job script, and submit.
    Returns (job_id, save_path).

    Any nn.Module and any torch Dataset that is pickle-serializable works.
    Trained state dict is saved to save_path. Load back with:
        model.load_state_dict(torch.load(save_path))
    """
    os.makedirs(LOGS_DIR,   exist_ok=True)
    job_dir         = os.path.join(MODELS_DIR, job_name)
    os.makedirs(job_dir, exist_ok=True)

    model_path      = os.path.join(job_dir, 'init.pt')
    train_data_path = os.path.join(job_dir, 'train_dataset.pt')
    test_data_path  = os.path.join(job_dir, 'test_dataset.pt')
    save_path       = os.path.join(job_dir, 'trained.pkl')  # if not overridden

    torch.save(model,         model_path)
    torch.save(train_dataset, train_data_path)
    torch.save(test_dataset,  test_data_path)
    print(f"Serialized model      → {model_path}")
    print(f"Serialized train data → {train_data_path}")
    print(f"Serialized test data  → {test_data_path}")

    if save_path is None:
        save_path = os.path.join(MODELS_DIR, f'{job_name}_trained.pkl')

    cmd = (
        f'python {PROJECT_DIR}/cluster_train.py'
        f' --model_path {model_path}'
        f' --train_dataset_path {train_data_path}'
        f' --test_dataset_path {test_data_path}'
        f' --epochs {epochs}'
        f' --batch_size {batch_size}'
        f' --weight_decay {weight_decay}'
        f' --save_path {save_path}'
        f' --early_stopping {early_stopping}'
    )
    if lr is not None:
        cmd += f' --lr {lr}'

    job_script = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --partition=gpu_a100
        #SBATCH --nodes=1
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task={cpus}
        #SBATCH --gpus=1
        #SBATCH --time={time}
        #SBATCH --mem={mem}
        #SBATCH --output={LOGS_DIR}/{job_name}_%j.out
        #SBATCH --error={LOGS_DIR}/{job_name}_%j.err

        echo "Job ID: $SLURM_JOB_ID"
        echo "Node: $SLURMD_NODENAME"
        echo "Start: $(date)"

        source ~/.bashrc
        conda activate {CONDA_ENV}

        cd {PROJECT_DIR}

        {cmd}

        echo "Done: $(date)"
    """)

    job_script_path = os.path.join(PROJECT_DIR, f'{job_name}.job')
    with open(job_script_path, 'w') as f:
        f.write(job_script)

    result = subprocess.run(
        ['sbatch', job_script_path],
        capture_output=True, text=True, check=True
    )

    job_id = result.stdout.strip().split()[-1]
    print(result.stdout.strip())
    print(f"Logs:    {LOGS_DIR}/{job_name}_{job_id}.out")
    print(f"Trained model will be saved to: {save_path}")

    return job_id, save_path


def job_status(job_id):
    """Check the status of a submitted job."""
    result = subprocess.run(
        ['squeue', '--job', str(job_id), '--format=%.10i %.12j %.8T %.10M %.6D %R'],
        capture_output=True, text=True
    )
    print(result.stdout if result.stdout.strip()
          else f"Job {job_id} no longer in queue (finished or failed).")


def get_job_dir(job_name):
    return os.path.join(MODELS_DIR, job_name)

def get_save_path(job_name):
    return os.path.join(get_job_dir(job_name), 'trained.pkl')

def get_losses_path(job_name):
    return os.path.join(get_job_dir(job_name), 'losses.pt')

def get_config_path(job_name):
    return os.path.join(get_job_dir(job_name), 'config.pt')