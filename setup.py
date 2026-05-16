"""
setup.py
--------
Run this once from the terminal after cloning the repo:

    python setup.py

This will:
  1. Detect whether you are on Snellius or a local machine
  2. Create a virtual environment in <project_root>/venv
  3. Install all requirements into it

After this, job scripts submitted via cluster.py will automatically
activate this venv. You never need to run this again unless you delete
the venv or add new dependencies to requirements.txt.

Usage on Snellius:
    module load 2025
    module load Python/3.11.3-GCCcore-12.3.0
    python setup.py

Usage on a local machine:
    python setup.py
"""

import os
import sys
import subprocess
import platform

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR    = os.path.join(PROJECT_DIR, 'venv')
REQS_FILE   = os.path.join(PROJECT_DIR, 'requirements.txt')

# Snellius is detected by hostname
ON_SNELLIUS = 'snellius' in platform.node().lower()


def run(cmd, **kwargs):
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def create_venv():
    print(f"\n[1/2] Creating virtual environment at {VENV_DIR} ...")
    run([sys.executable, '-m', 'venv', VENV_DIR])


def install_requirements():
    pip = os.path.join(VENV_DIR, 'bin', 'pip')
    print(f"\n[2/2] Installing requirements from {REQS_FILE} ...")
    run([pip, 'install', '--upgrade', 'pip'])
    run([pip, 'install', '-r', REQS_FILE])


def write_config():
    """Write project config so cluster.py knows where the venv is."""
    config_path = os.path.join(PROJECT_DIR, '.cluster_config')
    with open(config_path, 'w') as f:
        f.write(f"PROJECT_DIR={PROJECT_DIR}\n")
        f.write(f"VENV_DIR={VENV_DIR}\n")
        f.write(f"ON_SNELLIUS={ON_SNELLIUS}\n")
    print(f"\nConfig written to {config_path}")


def main():
    print("=" * 60)
    print("DDPM project setup")
    print(f"  Platform : {'Snellius' if ON_SNELLIUS else platform.node()}")
    print(f"  Python   : {sys.executable}")
    print(f"  Project  : {PROJECT_DIR}")
    print("=" * 60)

    if ON_SNELLIUS:
        print("\nNote: on Snellius, make sure you loaded the right Python module")
        print("before running this script:")
        print("  module load 2025")
        print("  module load Python/3.11.3-GCCcore-12.3.0")
        print()

    if os.path.exists(VENV_DIR):
        print(f"Venv already exists at {VENV_DIR}.")
        answer = input("Reinstall? [y/N] ").strip().lower()
        if answer != 'y':
            print("Skipping venv creation.")
            write_config()
            return

    if not os.path.exists(REQS_FILE):
        print(f"ERROR: {REQS_FILE} not found. Cannot install requirements.")
        sys.exit(1)

    create_venv()
    install_requirements()
    write_config()

    print("\n✓ Setup complete. You can now use cluster.py from your notebook.")


if __name__ == '__main__':
    main()
