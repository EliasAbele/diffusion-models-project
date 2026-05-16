"""
cluster.py
----------
Submit training and generation jobs to SLURM. Supports venv, conda,
and module-only environments. Reads project config from .cluster_config
written by setup.py, but all settings can be overridden manually.

Quickstart
----------
Run setup.py once in the terminal, then in your notebook:

    import cluster

    # override if needed (setup.py sets these automatically)
    # cluster.PROJECT_DIR = '/your/project/path'
    # cluster.VENV_DIR    = '/your/project/path/venv'

    job_id = cluster.train_and_generate_on_cluster(
        model=unet,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        epochs=100,
        n_images=8,
        job_name='unet_128',
    )
    cluster.job_status(job_id)

    # load results
    import torch
    x = torch.load(cluster.get_generated_path('unet_128'))
"""

import os
import subprocess
import textwrap
import torch


# ---------------------------------------------------------------------------
# Config — set automatically by setup.py, override manually if needed
# ---------------------------------------------------------------------------

PROJECT_DIR = None
VENV_DIR    = None
LOGS_DIR    = None
MODELS_DIR  = None
ON_SNELLIUS = False

# environment type: 'venv', 'conda', or 'module'
ENV_TYPE    = 'conda'
CONDA_ENV   = 'diffusion'   # only used if ENV_TYPE == 'conda'

# Snellius module to load before activating the environment
PYTHON_MODULE = 'Python/3.11.3-GCCcore-12.3.0'


def _load_config():
    """Load config written by setup.py if it exists."""
    global PROJECT_DIR, VENV_DIR, LOGS_DIR, MODELS_DIR, ON_SNELLIUS,CONDA_ENV

    # look for .cluster_config next to cluster.py
    here        = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(here, '.cluster_config')

    if os.path.exists(config_path):
        with open(config_path) as f:
            for line in f:
                line = line.strip()
                if '=' not in line:
                    continue
                key, val = line.split('=', 1)
                if key == 'PROJECT_DIR':
                    PROJECT_DIR = val
                elif key == 'VENV_DIR':
                    VENV_DIR = val
                elif key == 'ON_SNELLIUS':
                    ON_SNELLIUS = val == 'True'
                elif key == 'CONDA_ENV':
                    CONDA_ENV = val

    if PROJECT_DIR:
        LOGS_DIR   = os.path.join(PROJECT_DIR, 'logs')
        MODELS_DIR = os.path.join(PROJECT_DIR, 'models')


_load_config()


def _check_config():
    if PROJECT_DIR is None:
        raise RuntimeError(
            "PROJECT_DIR is not set. Either run setup.py first, "
            "or set cluster.PROJECT_DIR manually."
        )
    if ENV_TYPE == 'venv' and VENV_DIR is None:
        raise RuntimeError(
            "VENV_DIR is not set. Either run setup.py first, "
            "or set cluster.VENV_DIR manually."
        )
    if ENV_TYPE == 'conda' and CONDA_ENV is None:
        raise RuntimeError(
            "CONDA_ENV is not set. Set cluster.CONDA_ENV = 'your_env_name'."
        )


def _activation_block():
    """Return the shell lines that activate the right environment."""
    if ENV_TYPE == 'venv':
        lines = []
        if ON_SNELLIUS:
            lines += [
                'module load 2025',
                f'module load {PYTHON_MODULE}',
            ]
        lines.append(f'source {VENV_DIR}/bin/activate')
        return '\n'.join(lines)

    elif ENV_TYPE == 'conda':
        return textwrap.dedent(f"""\
            source ~/.bashrc
            conda activate {CONDA_ENV}""")

    elif ENV_TYPE == 'module':
        return textwrap.dedent(f"""\
            module load 2025
            module load {PYTHON_MODULE}""")

    else:
        raise ValueError(f"Unknown ENV_TYPE: {ENV_TYPE!r}. Choose 'venv', 'conda', or 'module'.")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_job_dir(job_name):
    return os.path.join(MODELS_DIR, job_name)

def get_save_path(job_name):
    return os.path.join(get_job_dir(job_name), 'trained.pkl')

def get_losses_path(job_name):
    return os.path.join(get_job_dir(job_name), 'losses.pt')

def get_config_path(job_name):
    return os.path.join(get_job_dir(job_name), 'config.pt')

def get_generated_path(job_name):
    return os.path.join(get_job_dir(job_name), 'generated.pt')


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _serialize(model, train_dataset, test_dataset, job_name):
    job_dir = get_job_dir(job_name)
    os.makedirs(job_dir,  exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    model_path      = os.path.join(job_dir, 'init.pt')
    train_data_path = os.path.join(job_dir, 'train_dataset.pt')
    test_data_path  = os.path.join(job_dir, 'test_dataset.pt')

    torch.save(model,         model_path)
    torch.save(train_dataset, train_data_path)
    torch.save(test_dataset,  test_data_path)
    print(f"Serialized model      → {model_path}")
    print(f"Serialized train data → {train_data_path}")
    print(f"Serialized test data  → {test_data_path}")

    return model_path, train_data_path, test_data_path


def _train_cmd(model_path, train_data_path, test_data_path, job_name,
               epochs, batch_size, weight_decay, lr, early_stopping):
    save_path = get_save_path(job_name)
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
    return cmd


def _generate_cmd(job_name, n_images, stochasticity, T, ncol):
    job_dir = get_job_dir(job_name)
    return (
        f'python {PROJECT_DIR}/cluster_generate.py'
        f' --job_dir {job_dir}'
        f' --n_images {n_images}'
        f' --stochasticity {stochasticity}'
        f' --T {T}'
        f' --ncol {ncol}'
    )


def _submit(job_name, commands, time, mem, cpus):
    activation = _activation_block()
    commands_str = '\n'.join(commands)

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

        {activation}

        cd {PROJECT_DIR}

        {commands_str}

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
    print(f"Logs: {LOGS_DIR}/{job_name}_{job_id}.out")
    return job_id


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_on_cluster(
    model,
    train_dataset,
    test_dataset,
    epochs          = 50,
    batch_size      = 32,
    weight_decay    = 1e-4,
    lr              = None,
    early_stopping  = 10,
    time            = '04:00:00',
    mem             = '32G',
    cpus            = 4,
    job_name        = 'ddpm_job',
):
    """Serialize model + datasets and submit a training job. Returns job_id."""
    _check_config()
    model_path, train_data_path, test_data_path = _serialize(
        model, train_dataset, test_dataset, job_name
    )
    cmd = _train_cmd(
        model_path, train_data_path, test_data_path, job_name,
        epochs, batch_size, weight_decay, lr, early_stopping
    )
    return _submit(job_name, [cmd], time, mem, cpus)


def train_and_generate_on_cluster(
    model,
    train_dataset,
    test_dataset,
    epochs          = 50,
    batch_size      = 32,
    weight_decay    = 1e-4,
    lr              = None,
    early_stopping  = 10,
    n_images        = 8,
    stochasticity   = 1.0,
    T               = 1000,
    ncol            = 4,
    time            = '04:00:00',
    mem             = '32G',
    cpus            = 4,
    job_name        = 'ddpm_job',
):
    """
    Serialize model + datasets and submit a job that trains then generates.
    Generation only runs if training completes successfully.
    Returns job_id.
    """
    _check_config()
    model_path, train_data_path, test_data_path = _serialize(
        model, train_dataset, test_dataset, job_name
    )
    train_cmd = _train_cmd(
        model_path, train_data_path, test_data_path, job_name,
        epochs, batch_size, weight_decay, lr, early_stopping
    )
    gen_cmd = _generate_cmd(job_name, n_images, stochasticity, T, ncol)
    return _submit(job_name, [train_cmd, gen_cmd], time, mem, cpus)

def generate_on_cluster(
    job_name,
    n_images      = 8,
    stochasticity = 1.0,
    T             = 1000,
    ncol          = 4,
    time          = '01:00:00',
    mem           = '16G',
    cpus          = 4,
):
    """
    Submit a generation-only job for an already trained model.
    The job directory must already contain config.pt and trained.pkl,
    i.e. training must have completed already.
    Returns job_id.
    """
    _check_config()
    cmd = _generate_cmd(job_name, n_images, stochasticity, T, ncol)
    return _submit(job_name + '_gen', [cmd], time, mem, cpus)


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------

def job_status(job_id):
    """Check the status of a submitted job."""
    result = subprocess.run(
        ['squeue', '--job', str(job_id)],
        capture_output=True, text=True
    )
    if result.returncode != 0 or not result.stdout.strip():
        result = subprocess.run(
            ['sacct', '-j', str(job_id),
             '--format=JobID,JobName,State,Elapsed,ExitCode'],
            capture_output=True, text=True
        )
    print(result.stdout if result.stdout.strip()
          else f"Job {job_id} not found.")


def cancel_job(job_id):
    """Cancel a queued or running job."""
    result = subprocess.run(
        ['scancel', str(job_id)],
        capture_output=True, text=True
    )
    print(f"Cancelled job {job_id}" if result.returncode == 0 else result.stderr)
