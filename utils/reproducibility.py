"""Seeding, determinism, RNG-state capture/restore and run metadata."""

import os
import platform
import random
import subprocess

import numpy as np
import torch

def set_seed(seed: int, deterministic: bool = True, deterministic_algorithms: bool = True):
    """Seeds random, numpy and torch (CPU and CUDA). deterministic=True gives
    bit-reproducible runs (cudnn.deterministic, no autotuning, deterministic
    algorithms); deterministic=False enables cudnn.benchmark autotuning, which
    is several times faster (PRUSR: 4.6x) but only reproducible up to
    floating-point reordering."""
    # cuBLAS is deterministic only with a fixed workspace, read when its handle
    # is created (the first CUDA matmul in the process); it also restricts
    # cuBLAS, so it is set only when bit-reproducibility is asked for.
    if deterministic and deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic and deterministic_algorithms, warn_only=True)


def seed_worker(worker_id: int):
    # torch.initial_seed() inside a worker is already base_seed + worker_id,
    # and base_seed comes from the DataLoader's seeded generator.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_rng_state(loader_generator: torch.Generator) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader_generator": loader_generator.get_state(),
    }


def set_rng_state(state: dict, loader_generator: torch.Generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    loader_generator.set_state(state["loader_generator"])


def git_info(repo_dir: str = ".") -> dict:
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=repo_dir, stderr=subprocess.DEVNULL).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    # Only tracked files count as "dirty": the repo always has untracked
    # outputs lying around, which don't affect what code actually ran.
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def library_versions() -> dict:
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    from importlib.metadata import version, PackageNotFoundError
    for package in ("lpips", "torchvision", "pyyaml"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions
