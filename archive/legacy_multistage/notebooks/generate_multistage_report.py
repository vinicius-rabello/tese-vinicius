"""Compact multistage evaluation report: a headline cross-model bar chart,
five 8-row tables (one per model, one row per strategy x loss-norm), five
more 8-row tables with full test-set metrics (if evaluation_results_
multistage_summary.csv exists), and one paper-style plot page per model
showing its single best-performing config.

The first five tables are driven directly by training-log "Final MSE" (the
same metric EarlyStopping/checkpoint-selection already use, and the same one
parse_stage1_log()/parse_multistage_log() treat as comparable across every
stage-1 LOSS_MODE) -- zero GPU cost, safe to run alongside an active training
queue. The second five read the real full-test-set sweep (all 7 per-channel
metrics + 2 combined ones) from evaluate_multistage_models.ipynb's aggregate
CSV. Only the five headline plot pages do real forward passes (15 samples
each, one per model), so the GPU footprint stays light.

A model with no completed multistage configs yet (e.g. SRNO before its queue
slot runs) is skipped with a note instead of erroring -- re-run this script
once more configs finish to fill in the rest.

Run from the notebooks/ directory:

    python generate_multistage_report.py
"""

import os
import sys
import textwrap
import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append("..")
from datasets.super_res_dataset import SuperResDataset
from metrics.kinetic_energy_spectrum import compute_energy_spectrum
from multistage_report_lib import (
    MODELS, STRATEGY_ORDER, STRATEGY_LABELS, TEST_FILES, device,
    find_stage1_log, parse_stage1_log, parse_multistage_log, multistage_log_path,
    load_multistage, gather_model_rows,
)

OUT_PATH = "evaluation_report_multistage.pdf"
FULL_METRICS_CSV = "evaluation_results_multistage_summary.csv"

PER_CHANNEL_METRICS = ["mse", "ssim", "psnr", "lpips", "fsim", "epi", "tke"]
COMBINED_METRICS = ["ke", "wke"]
LOWER_IS_BETTER = {"mse", "lpips", "tke", "ke", "wke"}

all_rows = {model: gather_model_rows(model) for model in MODELS}
for model, rows in all_rows.items():
    print(f"{model}: {len(rows)}/8 configs available")

full_metrics_df = pd.read_csv(FULL_METRICS_CSV) if os.path.exists(FULL_METRICS_CSV) else None
if full_metrics_df is None:
    print(f"No {FULL_METRICS_CSV} found -- skipping the full-test-set metric tables "
          "(run the evaluate_multistage_models.ipynb sweep to produce it).")


# ------------------------------------------------------------- the report --
with PdfPages(OUT_PATH) as pdf:
    # ---- title / methodology ----
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.9, "Multistage Residual Model Evaluation Report",
              ha="center", fontsize=20, fontweight="bold")
    fig.text(0.5, 0.86, datetime.date.today().isoformat(), ha="center", fontsize=10, color="gray")

    def wrapped(text):
        return textwrap.fill(text, width=80) + "\n"

    methodology = (
        wrapped(
            "Multistage residual model: final(LR) = stage1(LR) + epsilon * stage2(LR). "
            "Stage 1 is a frozen, already-trained model (any of the 4 loss strategies). "
            "Stage 2 is a freshly-initialized copy of the same architecture, trained "
            "with plain MSE or L1 to predict stage 1's normalized residual "
            "(HR - stage1(LR)) / epsilon, where epsilon = RMS(residual) over the "
            "training set."
        ) + "\n"
        + wrapped(
            "All figures below use 'Final MSE': the validation MSE tracked during "
            "training itself (same metric EarlyStopping/checkpoint-selection use), "
            "not a separate full-test-set sweep -- directly comparable across every "
            "stage-1 loss strategy since the MSE component is logged separately "
            "regardless of LOSS_MODE."
        ) + "\n"
        + wrapped(
            "Caveat: early stopping (patience=15, min_delta=1e-4) can trigger "
            "mechanically at exactly epoch 16 when a strategy's Final MSE range is "
            "smaller than min_delta itself (true for mse_lpips/mse_lpips_ke stage-1 "
            "models) -- this looks like fast convergence but may just be a metric-scale "
            "artifact cutting stage 2 off early, understating its true benefit for "
            "those two strategies."
        ) + "\n"
        + wrapped(
            "Caveat: for mse_lpips_ke/mse_wke, stage 1 already directly optimizes "
            "KE/WKE, so its baseline KE/WKE values are small -- stage 2 (plain "
            "MSE/L1 on the residual, never KE/WKE-aware) consistently pushes them "
            "back up 2-19x across every architecture, even while MSE/SSIM/PSNR "
            "improve. Multistage trades spectral accuracy for pixel accuracy "
            "whenever stage 1 already nailed the spectrum."
        )
    )
    fig.text(0.08, 0.74, methodology, ha="left", va="top", fontsize=10, family="monospace")
    pdf.savefig(fig)
    plt.close(fig)

    # ---- cross-model summary bar chart ----
    fig, ax = plt.subplots(figsize=(11, 6.5))
    bar_w = 0.2
    x = np.arange(len(STRATEGY_ORDER))
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODELS)))

    for i, model in enumerate(MODELS):
        rows = all_rows[model]
        best_per_strategy = {}
        for r in rows:
            s = r["strategy"]
            if s not in best_per_strategy or r["improvement_pct"] > best_per_strategy[s]:
                best_per_strategy[s] = r["improvement_pct"]
        values = [best_per_strategy.get(s, np.nan) for s in STRATEGY_ORDER]
        ax.bar(x + (i - len(MODELS) / 2 + 0.5) * bar_w, values, width=bar_w,
               label=model, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGY_ORDER], fontsize=9)
    ax.set_ylabel("Multistage MSE improvement over Stage 1 alone (%)")
    ax.set_title("Multistage improvement by strategy, across all 5 architectures\n"
                  "(best of MSE/L1-trained stage 2 shown per bar)", fontsize=12)
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    ax.legend(fontsize=9, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ---- 5 per-model tables (8 rows each) ----
    for model in MODELS:
        rows = all_rows[model]
        fig = plt.figure(figsize=(11, 5.5))
        fig.text(0.5, 0.93, f"{model} -- all completed multistage configs", ha="center",
                  fontsize=15, fontweight="bold")

        if not rows:
            fig.text(0.5, 0.5, "No completed multistage configs yet.", ha="center",
                      fontsize=12, color="gray")
            pdf.savefig(fig)
            plt.close(fig)
            continue

        col_labels = ["Strategy", "Loss Norm", "Epsilon", "Stage-1\nFinal MSE",
                      "+Stage-2\nFinal MSE", "Stage-2\nEpochs", "Improvement"]
        cell_text = []
        for r in rows:
            cell_text.append([
                STRATEGY_LABELS[r["strategy"]],
                r["loss_norm"].upper(),
                f"{r['epsilon']:.4g}",
                f"{r['stage1_final_mse']:.4g}",
                f"{r['multistage_final_mse']:.4g}",
                str(r["multistage_epochs"]),
                f"{r['improvement_pct']:+.1f}%",
            ])
        n_missing = 8 - len(rows)
        for _ in range(n_missing):
            cell_text.append(["(not yet run)", "-", "-", "-", "-", "-", "-"])

        # Winning row per column (bolded below): lowest Stage-1 Final MSE (col 3),
        # lowest +Stage-2 Final MSE (col 4), highest Improvement % (col 6) --
        # Epsilon/Epochs (cols 2, 5) have no "better" direction, so no winner.
        winner_row = {
            3: min(range(len(rows)), key=lambda i: rows[i]["stage1_final_mse"]),
            4: min(range(len(rows)), key=lambda i: rows[i]["multistage_final_mse"]),
            6: max(range(len(rows)), key=lambda i: rows[i]["improvement_pct"]),
        }

        ax = fig.add_axes([0.03, 0.08, 0.94, 0.8])
        ax.axis("off")
        col_widths = [0.20, 0.10, 0.11, 0.14, 0.14, 0.11, 0.13]
        table = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center",
                          colWidths=col_widths, bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        table.scale(1, 1.8)
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor("#dbe5f1")
                cell.set_text_props(fontweight="bold")
            elif r - 1 < len(rows):
                improvement = rows[r - 1]["improvement_pct"]
                if improvement >= 50:
                    cell.set_facecolor("#eafbea")
                elif improvement < 25:
                    cell.set_facecolor("#fdeeea")
                if winner_row.get(c) == r - 1:
                    cell.set_text_props(fontweight="bold")
            else:
                cell.set_facecolor("#f2f2f2")
        pdf.savefig(fig)
        plt.close(fig)

    # ---- 5 more per-model tables: full test-set metrics (from evaluate_model()) ----
    # The tables above use "Final MSE" from training logs (val set, zero GPU
    # cost). These instead use the real full-test-set sweep (2022-2024, all 7
    # per-channel metrics + 2 combined ones) from
    # evaluate_multistage_models.ipynb's aggregate CSV -- directly answers
    # "does multistage help on metrics besides MSE too?" Shown as the
    # multistage model's absolute value, with the % improvement over stage-1
    # alone in parentheses (direction-aware: lower-is-better for
    # MSE/LPIPS/TKE/KE/WKE, higher-is-better for the rest).
    if full_metrics_df is not None:
        for model in MODELS:
            model_df = full_metrics_df[full_metrics_df["Model"] == model].copy()
            model_df["_order"] = model_df["Strategy"].apply(
                lambda s: STRATEGY_ORDER.index(s) if s in STRATEGY_ORDER else 99)
            model_df = model_df.sort_values(["_order", "Loss_Norm"])

            fig = plt.figure(figsize=(17, 6))
            fig.text(0.5, 0.93, f"{model} -- full test-set metrics (2022-2024): "
                      "multistage value (% improvement over stage 1 alone)", ha="center",
                      fontsize=13.5, fontweight="bold")

            if model_df.empty:
                fig.text(0.5, 0.5, "No full-metric sweep results yet for this model.", ha="center",
                          fontsize=12, color="gray")
                pdf.savefig(fig)
                plt.close(fig)
                continue

            metric_names = PER_CHANNEL_METRICS + COMBINED_METRICS
            col_labels = ["Strategy", "Loss\nNorm"] + [m.upper() for m in metric_names]
            cell_text = []
            cell_signs = []  # matching shape, True = multistage better (for coloring)
            cell_values = []  # matching shape, raw multistage value (for per-column winner)
            for _, r in model_df.iterrows():
                row_text = [STRATEGY_LABELS[r["Strategy"]], r["Loss_Norm"].upper()]
                row_signs = [None, None]
                row_values = [None, None]
                for m in metric_names:
                    if m in COMBINED_METRICS:
                        s1, ms = r[f"{m}_stage1"], r[f"{m}_multistage"]
                    else:
                        s1 = (r[f"{m}_u_stage1"] + r[f"{m}_v_stage1"]) / 2
                        ms = (r[f"{m}_u_multistage"] + r[f"{m}_v_multistage"]) / 2
                    pct = 100 * (s1 - ms) / s1 if m in LOWER_IS_BETTER else 100 * (ms - s1) / s1
                    row_text.append(f"{ms:.4g} ({pct:+.1f}%)")
                    row_signs.append(pct >= 0)
                    row_values.append(ms)
                cell_text.append(row_text)
                cell_signs.append(row_signs)
                cell_values.append(row_values)
            n_missing = 8 - len(cell_text)
            for _ in range(n_missing):
                cell_text.append(["(not yet run)", "-"] + ["-"] * len(metric_names))
                cell_signs.append([None] * (2 + len(metric_names)))
                cell_values.append([None] * (2 + len(metric_names)))

            # Winning row per metric column (bolded below): best absolute
            # multistage value across the 8 strategy/loss-norm rows,
            # direction-aware (lower-is-better metrics use min, rest use max).
            n_real_rows = len(model_df)
            winner_row = {}
            for c, m in enumerate(metric_names, start=2):
                col_vals = [cell_values[i][c] for i in range(n_real_rows)]
                pick = min if m in LOWER_IS_BETTER else max
                winner_row[c] = pick(range(n_real_rows), key=lambda i: col_vals[i])

            ax = fig.add_axes([0.01, 0.08, 0.98, 0.8])
            ax.axis("off")
            strategy_w, loss_norm_w = 0.14, 0.045
            metric_w = (1 - strategy_w - loss_norm_w) / len(metric_names)
            col_widths = [strategy_w, loss_norm_w] + [metric_w] * len(metric_names)
            table = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center",
                              colWidths=col_widths, bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.8)
            for (r, c), cell in table.get_celld().items():
                if r == 0:
                    cell.set_facecolor("#dbe5f1")
                    cell.set_text_props(fontweight="bold")
                elif c >= 2:
                    sign = cell_signs[r - 1][c]
                    if sign is True:
                        cell.set_facecolor("#eafbea")
                    elif sign is False:
                        cell.set_facecolor("#fdeeea")
                    if winner_row.get(c) == r - 1:
                        cell.set_text_props(fontweight="bold")
            pdf.savefig(fig)
            plt.close(fig)

    # ---- 5 headline plot pages: best config per model ----
    test_dataset = None

    def get_test_dataset():
        global test_dataset
        if test_dataset is None:
            test_dataset = SuperResDataset(hr_files=TEST_FILES, downsample_factor=1, freq_threshold=10)
        return test_dataset

    for model in MODELS:
        rows = all_rows[model]
        if not rows:
            fig = plt.figure(figsize=(11, 4))
            fig.text(0.5, 0.5, f"{model}: no completed multistage configs yet -- "
                      "re-run this script once its queue slot finishes.",
                      ha="center", va="center", fontsize=12, color="gray")
            pdf.savefig(fig)
            plt.close(fig)
            continue

        best = min(rows, key=lambda r: r["multistage_final_mse"])
        strategy, loss_norm = best["strategy"], best["loss_norm"]
        weight_path = best["weight_path"]

        # -- loss convergence (epochs) --
        stage1_log = find_stage1_log(model, strategy)
        s1_epochs, s1_val = parse_stage1_log(stage1_log)
        offset = s1_epochs[-1] if s1_epochs else 0
        ms_log = multistage_log_path(weight_path, model)
        ms_epochs, ms_val = parse_multistage_log(ms_log)
        ms_epochs_shifted = [offset + e for e in ms_epochs]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"{model} -- best config: {STRATEGY_LABELS[strategy]} "
                      f"(stage 2 {loss_norm.upper()}-trained), "
                      f"Final MSE {best['multistage_final_mse']:.4g} "
                      f"({best['improvement_pct']:+.1f}% vs stage 1 alone)",
                      fontsize=12, fontweight="bold")

        ax1.semilogy(s1_epochs, s1_val, color="black", lw=1.5, label=f"Stage 1 ({strategy})")
        ax1.semilogy(ms_epochs_shifted, ms_val, color="tab:orange", lw=1.5,
                      label=f"Stage 1 + Stage 2 ({loss_norm}-trained)")
        if strategy != "mse_lpips":
            lpips_ref_log = find_stage1_log(model, "mse_lpips")
            if lpips_ref_log is not None:
                lp_epochs, lp_val = parse_stage1_log(lpips_ref_log)
                ax1.semilogy(lp_epochs, lp_val, color="tab:red", lw=1.5, label="Single-stage (mse_lpips)")
        ax1.axvline(offset, color="gray", ls=":", lw=1, label="Stage 1 -> Stage 2 handoff")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Validation MSE (log scale)")
        ax1.set_title("Loss convergence: staged vs single-stage", fontsize=10)
        ax1.grid(True, which="both", ls="--", alpha=0.3)
        ax1.legend(fontsize=8)

        # -- averaged power spectrum --
        multistage, stage1, stage2, epsilon = load_multistage(model, weight_path)
        ds = get_test_dataset()
        n_samples = min(15, len(ds))
        indices = np.linspace(0, len(ds) - 1, n_samples, dtype=int)

        hr_spectra, s1_spectra, final_spectra = [], [], []
        k_bins = None
        with torch.no_grad():
            for idx in indices:
                lr, hr = ds[int(idx)]
                lr, hr = lr.unsqueeze(0).to(device), hr.unsqueeze(0).to(device)
                stage1_pred = stage1(lr)
                final_pred = stage1_pred + epsilon * stage2(lr)
                k_bins, e_hr = compute_energy_spectrum(hr.squeeze(0).cpu().numpy())
                _, e_s1 = compute_energy_spectrum(stage1_pred.squeeze(0).cpu().numpy())
                _, e_final = compute_energy_spectrum(final_pred.squeeze(0).cpu().numpy())
                hr_spectra.append(e_hr)
                s1_spectra.append(e_s1)
                final_spectra.append(e_final)

        hr_arr, s1_arr, final_arr = np.stack(hr_spectra), np.stack(s1_spectra), np.stack(final_spectra)
        hr_mean, hr_std = hr_arr.mean(axis=0), hr_arr.std(axis=0)

        ax2.fill_between(k_bins, np.clip(hr_mean - hr_std, 1e-30, None), hr_mean + hr_std,
                          color="gray", alpha=0.4, label="Target (HR, mean +/- std)")
        ax2.loglog(k_bins, hr_mean, color="black", lw=1, alpha=0.6)
        ax2.loglog(k_bins, s1_arr.mean(axis=0), color="tab:purple", lw=2, ls="--", label="Stage 1 Alone")
        ax2.loglog(k_bins, final_arr.mean(axis=0), color="tab:orange", lw=2, label="Stage 1 + Stage 2 (Final)")
        ax2.set_xlabel("Wavenumber $k$")
        ax2.set_ylabel("$E(k)$")
        ax2.set_title(f"Power spectrum, averaged over {n_samples} samples", fontsize=10)
        ax2.grid(True, which="both", ls="--", alpha=0.3)
        ax2.legend(fontsize=8)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        del multistage, stage1, stage2
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- appendix: multistage MSE vs best single-stage LPIPS, per model ----
    # Same 3-curve layout as the very first SRCNN demo: stage-1 mse (black),
    # stage-1 mse + stage-2 mse-trained (orange), and whichever of
    # mse_lpips/mse_lpips_ke reaches the lower final validation MSE for that
    # model, as a single-stage reference (red). Deliberately NOT the "best
    # config" plots above -- those pick whatever strategy wins outright (often
    # mse_lpips itself), which makes the black curve noisy and removes the
    # reference line entirely (see "if strategy != mse_lpips" above). This
    # section always keeps the comparison the same shape: does two-stage MSE
    # training catch up to single-stage LPIPS training?
    for model in MODELS:
        stage1_mse_log = find_stage1_log(model, "mse")
        mse_row = next((r for r in all_rows[model]
                         if r["strategy"] == "mse" and r["loss_norm"] == "mse"), None)
        if stage1_mse_log is None or mse_row is None:
            fig = plt.figure(figsize=(9, 4))
            fig.text(0.5, 0.5, f"{model}: mse multistage not finished yet -- "
                      "re-run this script once it completes.",
                      ha="center", va="center", fontsize=12, color="gray")
            pdf.savefig(fig)
            plt.close(fig)
            continue

        s1_epochs, s1_val = parse_stage1_log(stage1_mse_log)
        offset = s1_epochs[-1] if s1_epochs else 0
        ms_log = multistage_log_path(mse_row["weight_path"], model)
        ms_epochs, ms_val = parse_multistage_log(ms_log)
        ms_epochs_shifted = [offset + e for e in ms_epochs]

        lpips_candidates = []
        for lpips_strategy in ["mse_lpips", "mse_lpips_ke"]:
            log = find_stage1_log(model, lpips_strategy)
            if log is not None:
                epochs, val = parse_stage1_log(log)
                if val:
                    lpips_candidates.append((lpips_strategy, epochs, val))
        if not lpips_candidates:
            continue
        best_lpips_strategy, lp_epochs, lp_val = min(lpips_candidates, key=lambda t: t[2][-1])

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.semilogy(s1_epochs, s1_val, color="black", lw=1.5, label="Stage 1 (mse)")
        ax.semilogy(ms_epochs_shifted, ms_val, color="tab:orange", lw=1.5,
                     label="Stage 1 + Stage 2 (mse, mse-trained)")
        ax.semilogy(lp_epochs, lp_val, color="tab:red", lw=1.5,
                     label=f"Single-stage ({best_lpips_strategy})")
        ax.axvline(offset, color="gray", ls=":", lw=1, label="Stage 1 -> Stage 2 handoff")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation MSE (log scale)")
        ax.set_title(f"{model} -- Loss convergence: multistage MSE vs best single-stage LPIPS")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend(fontsize=9)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

print(f"Wrote {OUT_PATH}")
