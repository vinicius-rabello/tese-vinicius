"""Shared data-gathering helpers for the multistage evaluation report
(generate_multistage_report.py) and its companion validation notebook
(validate_multistage_report.ipynb). Kept in one place so both always read
the exact same numbers -- the notebook is a genuine check on the PDF, not a
separate reimplementation that could quietly drift from it.

Everything here is driven by training-log "Final MSE" (the same metric
EarlyStopping/checkpoint-selection already use) -- zero GPU cost. Only
load_multistage() and its callers touch the GPU.
"""

import os
import re
import glob
import sys
from collections import OrderedDict
from datetime import datetime as dt

import torch

sys.path.append("..")
from models.model_factory import create_model
from models.multistage import MultistageModel
from train_multistage import load_stage1

BASE_MODEL_DIR = os.path.abspath("../models")
MODELS = ["SRCNN", "DSCMS", "PRUSR", "SoFoNO", "SRNO"]

STRATEGY_ORDER = ["mse", "mse_lpips", "mse_lpips_ke", "mse_wke"]
STRATEGY_LABELS = {
    "mse": "MSE only",
    "mse_lpips": "0.6 MSE + 0.4 LPIPS",
    "mse_lpips_ke": "0.6 MSE + 0.2 LPIPS + 0.2 KE",
    "mse_wke": "0.6 MSE + 0.4 Weighted KE",
}

TRAIN_SET_SIZE = 6924
TEST_FILES = [
    "../hycom_numpy/hycom_2022.npy",
    "../hycom_numpy/hycom_2023.npy",
    "../hycom_numpy/hycom_2024.npy",
]
device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- helpers --
def categorize_strategy(experiment_name):
    """"wke" is checked before "ke" since "wke" also contains the substring "ke"."""
    name = experiment_name.lower()
    if "wke" in name:
        return "mse_wke"
    if "ke" in name:
        return "mse_lpips_ke"
    if "lpips" in name:
        return "mse_lpips"
    if "mse" in name:
        return "mse"
    return "unknown"


def find_multistage_checkpoints(model_name):
    """Every stage-2 best checkpoint under models/{model_name}/*multistage*/,
    deduplicated by (strategy, loss_norm) keeping the most recent run."""
    model_dir = os.path.join(BASE_MODEL_DIR, model_name)
    filename = f"{model_name.upper()}_best.pth"
    pattern = os.path.join(model_dir, "*multistage*", "*", "output", "weights", filename)

    found = {}
    for weight_path in sorted(glob.glob(pattern)):
        relative = os.path.relpath(weight_path, model_dir)
        experiment_folder, run_timestamp = relative.split(os.sep)[0], relative.split(os.sep)[1]
        loss_norm = "l1" if experiment_folder.endswith("_l1") else "mse"
        base_folder = experiment_folder[:-3] if loss_norm == "l1" else experiment_folder
        base_folder = base_folder.replace("_multistage", "")
        strategy = categorize_strategy(base_folder)
        try:
            run_date = dt.strptime(run_timestamp, "%d_%m_%Y_%H%M%S")
        except ValueError:
            run_date = dt.min
        key = (strategy, loss_norm)
        if key not in found or run_date > found[key][0]:
            found[key] = (run_date, weight_path, experiment_folder)

    return [
        {"strategy": s, "loss_norm": ln, "weight_path": wp, "experiment_folder": ef}
        for (s, ln), (_, wp, ef) in sorted(found.items())
    ]


def find_stage1_log(model_name, strategy):
    """Most-recently-modified stage-1 (non-multistage) log matching this strategy."""
    model_dir = os.path.join(BASE_MODEL_DIR, model_name)
    pattern = os.path.join(model_dir, "**", "output", "logs", f"{model_name.upper()}_loss.txt")
    candidates = []
    for path in glob.glob(pattern, recursive=True):
        if "multistage" in path:
            continue
        top_folder = os.path.relpath(path, model_dir).split(os.sep)[0]
        if categorize_strategy(top_folder) == strategy:
            candidates.append(path)
    return max(candidates, key=os.path.getmtime) if candidates else None


def parse_stage1_log(path):
    """Stage-1 logs report the composite loss's MSE component separately
    (LOSS_MODE>=1 always includes it), so this is directly comparable to
    multistage's "Final MSE" regardless of stage-1 LOSS_MODE."""
    text = open(path).read()
    epochs = [int(x) for x in re.findall(r"Epoch \[(\d+)/", text)]
    val_mse = [float(x) for x in re.findall(r"Val\s+-> Total: [0-9.eE+-]+ \| MSE: ([0-9.eE+-]+)", text)]
    return epochs, val_mse


def parse_multistage_log(path):
    text = open(path).read()
    epochs = [int(x) for x in re.findall(r"Epoch \[(\d+)/", text)]
    val_final_mse = [float(x) for x in re.findall(
        r"Val\s+-> Residual loss: [0-9.eE+-]+ \| Final MSE: ([0-9.eE+-]+)", text)]
    return epochs, val_final_mse


def multistage_log_path(weight_path, model_name):
    """Weights live at .../<run>/output/weights/{MODEL}_best.pth; the matching
    log sits at .../<run>/output/logs/{MODEL}_multistage_loss.txt."""
    return (weight_path
            .replace(os.sep + "weights" + os.sep, os.sep + "logs" + os.sep)
            .replace(f"{model_name.upper()}_best.pth", f"{model_name.upper()}_multistage_loss.txt"))


def resolve_from_repo_root(path):
    """STAGE1_CHECKPOINT is stored relative to the repo root (train_multistage.py's
    cwd), but callers here run with cwd=notebooks/."""
    return path if os.path.isabs(path) else os.path.join("..", path)


def load_multistage(model_name, weight_path):
    """Loads stage1 (via the checkpoint path stored inside this checkpoint) and
    stage2, returning (multistage_model, stage1, stage2, epsilon)."""
    checkpoint = torch.load(weight_path, map_location=device)
    stage1 = load_stage1(model_name.lower(), resolve_from_repo_root(checkpoint["stage1_checkpoint"]), device)
    stage2 = create_model(model_name.lower(), in_channels=2, out_channels=2).to(device)
    cleaned = OrderedDict(
        (k[len("module."):] if k.startswith("module.") else k, v)
        for k, v in checkpoint["model_state_dict"].items()
    )
    stage2.load_state_dict(cleaned)
    stage2.eval()
    epsilon = checkpoint["epsilon"]
    multistage = MultistageModel(stage1, stage2, epsilon).to(device)
    multistage.eval()
    return multistage, stage1, stage2, epsilon


# ---------------------------------------------------------- data gathering --
def gather_model_rows(model_name):
    """One row per completed (strategy, loss_norm) checkpoint: stage-1 alone
    Final MSE, +stage-2 Final MSE, % improvement, epoch counts, epsilon.
    This is the exact data the PDF report's tables/bar-chart/plots are built
    from -- call it directly to check any number in the report by hand."""
    rows = []
    for entry in find_multistage_checkpoints(model_name):
        strategy, loss_norm, weight_path = entry["strategy"], entry["loss_norm"], entry["weight_path"]

        stage1_log = find_stage1_log(model_name, strategy)
        if stage1_log is None:
            continue
        s1_epochs, s1_val = parse_stage1_log(stage1_log)
        if not s1_val:
            continue

        ms_log = multistage_log_path(weight_path, model_name)
        if not os.path.exists(ms_log):
            continue
        ms_epochs, ms_val = parse_multistage_log(ms_log)
        if not ms_val:
            continue

        checkpoint = torch.load(weight_path, map_location="cpu")
        rows.append({
            "strategy": strategy, "loss_norm": loss_norm,
            "stage1_final_mse": s1_val[-1], "stage1_epochs": s1_epochs[-1],
            "multistage_final_mse": ms_val[-1], "multistage_epochs": ms_epochs[-1],
            "epsilon": checkpoint["epsilon"], "weight_path": weight_path,
            "improvement_pct": 100 * (s1_val[-1] - ms_val[-1]) / s1_val[-1],
        })
    rows.sort(key=lambda r: (
        STRATEGY_ORDER.index(r["strategy"]) if r["strategy"] in STRATEGY_ORDER else 99,
        r["loss_norm"],
    ))
    return rows
