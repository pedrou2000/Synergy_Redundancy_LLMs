"""
Batch‑processing script for Pythia checkpoints
---------------------------------------------
Runs *exactly* the same activation‑recording ➜ time‑series ➜ PhiID pipeline you
already tested, but loops over every training snapshot from **step000000** to
**step143000** in 1 k increments.

The script expects your usual Hydra config folder (`../config`) and re‑uses all
paths that depend on `cfg.model.revision`.  If your paths are written with
`${model.revision}` interpolations (recommended), each checkpoint’s results
will automatically land in its own sub‑directory, e.g.:

    outputs/step050000/activations/
    outputs/step050000/phyid/

Adjust the `CHECKPOINT_RANGE` or CLI args as you like.  Heavy job – make sure
you have the storage and GPU availability.
"""

from __future__ import annotations

import gc
import argparse
from pathlib import Path
from typing import List, Any, Union

import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import sys

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- project‑local imports ----------------------------------------------------
from src.utils import perturb_model
from src.activation_recorder import ActivationRecorder, MultiPromptActivations
from src.time_series_activations import MultiPromptTimeSeries
from src.phyid_decomposition import MultiPromptPhyID


def _overlay_plot(
        curves: dict[int, tuple[np.ndarray, np.ndarray]],
        plot_dir: Union[str, None],
        ylabel: str,
        title: str,
        fname: str,
        ylims: tuple[float, float] | None = None,
    ) -> None:
        plt.figure(figsize=(10, 6))

        # deterministic colour palette (sorted by training step)
        cmap = plt.cm.get_cmap("viridis_r", len(curves))
        for i, step in enumerate(sorted(curves)):
            xs, ys = curves[step]
            plt.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                label=f"step {step:,}",
                color=cmap(i),
            )

        if ylims is not None:
            plt.ylim(*ylims)
        plt.xlabel("Source layer")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.legend(ncol=2, fontsize="small")
        plt.tight_layout()

        if plot_dir:
            out_dir = os.path.join(plot_dir, "checkpoint_comparison")
            os.makedirs(out_dir, exist_ok=True)
            fpath = os.path.join(out_dir, fname)
            plt.savefig(fpath, dpi=300)
            print(f"→ saved to {fpath}")
            plt.close()
        else:
            plt.show()


def _plot_synergy_vs_step(
    totals: dict[int, float],
    plot_dir: Union[str, None],
    ylabel: str = "Total synergy  Σ(sts)",
    title: str = "Total synergy across training",
    fname: str = "total_synergy_vs_step.png",
) -> None:
    plt.figure(figsize=(10, 6))

    steps_sorted = sorted(totals)
    ys = [totals[s] for s in steps_sorted]

    plt.plot(steps_sorted, ys, marker="o", linewidth=2)
    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    if plot_dir:
        out_dir = os.path.join(plot_dir, "checkpoint_comparison")
        os.makedirs(out_dir, exist_ok=True)
        fpath = os.path.join(out_dir, fname)
        plt.savefig(fpath, dpi=300)
        print(f"→ saved to {fpath}")
        plt.close()
    else:
        plt.show()


def plot_for_checkpoints(
    steps: List[int],
    base_cfg: Any,
    *,
    plot_dir: Union[str, None] = None,
) -> None:
    """
    Compare checkpoints by overlaying their per‑layer curves in a single figure.

    Parameters
    ----------
    steps      : list[int]
        Training steps to load (e.g. [1000, 2000, 4000, …]).
    base_cfg   : OmegaConf
        The *base* Hydra config.  A deep copy is made for every checkpoint.
    plot_dir   : str | None
        Where to save the comparison figures.  If None → `plt.show()`.
    """
    # --- lazily import heavy deps ------------------------------------------------

    # Containers for the curves
    synergy_curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    synred_curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    total_synergy: dict[int, float] = {}      

    # --------------------------------------------------------------------------
    # Loop over checkpoints – load the saved Φ‑ID results, extract the curves
    # --------------------------------------------------------------------------
    for step in steps:
        try:
            cfg = OmegaConf.create(base_cfg)
            cfg.model.revision = f"step{step}"
            cfg.model.shortcode = f"P-1-{step}"
            cfg.model.it = f"base-{step}"

            phyid = MultiPromptPhyID.load_average_prompt_phyid(dir_path=cfg.paths.data_phyid_dir)
            phyid.build_data_array()

            print(f"✓ step{step} loaded")
            dims_to_reduce = [
                d for d in phyid.data_array.dims if d not in ("atom", "source_layer")
            ]
            synergy_series = (
                phyid.data_array.sel(atom="sts").mean(dim=dims_to_reduce)
            )  # dims → ('source_layer',)

            xs = synergy_series.coords["source_layer"].values
            ys = synergy_series.values
            synergy_curves[step] = (xs, ys)

            # ------------------------------------------------------------------
            # 2)  Normalised (synergy – redundancy) rank per layer
            # ------------------------------------------------------------------
            rank_da = phyid.syn_minus_red_rank  # dims → ('source_node', 'source_layer')
            layer_mean = rank_da.mean(dim="source_node")
            v_min, v_max = layer_mean.min().item(), layer_mean.max().item()
            norm = (
                xr.zeros_like(layer_mean)
                if v_min == v_max
                else (layer_mean - v_min) / (v_max - v_min)
            )

            xs_nr = norm.coords["source_layer"].values
            ys_nr = norm.values
            synred_curves[step] = (xs_nr, ys_nr)

            # --- 3) *total* synergy (new) -------------------------------------
            tot = float(phyid.data_array.sel(atom="sts").sum().item())
            total_synergy[step] = tot                         #  << NEW <<
            print(f"✓ step{step} loaded  –  Σsts = {tot:.3g}")

            print(f"✓ step{step} loaded")
        except Exception as e:
            print(f"[warning] step{step} skipped: {e}")
            continue

    # --------------------------------------------------------------------------
    # Draw the two overlay figures
    # --------------------------------------------------------------------------
    cfg = OmegaConf.create(base_cfg)
    cfg.model.shortcode = f"P‑1"
    cfg.model.it = f"base"


    if synergy_curves:
        _overlay_plot(
            synergy_curves,
            plot_dir=cfg.paths.plot_synergy_through_training_dir,
            ylabel="Mean synergy (sts)",
            title="Evolution of mean synergy per layer",
            fname="synergy_per_layer_comparison.png",
        )

    if synred_curves:
        _overlay_plot(
            synred_curves,
            plot_dir=cfg.paths.plot_synergy_through_training_dir,
            ylabel="Normalised (synergy – redundancy) rank",
            title="Evolution of synergy–redundancy rank per layer",
            fname="syn_minus_red_per_layer_comparison.png",
            ylims=(-0.05, 1.05),
        )
    if total_synergy:                                   
        _plot_synergy_vs_step(
            total_synergy,
            plot_dir=cfg.paths.plot_synergy_through_training_dir,
        )



@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    steps = [2**i for i in range(0, 10)] + [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000]
    # steps = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000] 

    plot_for_checkpoints(steps, cfg)


if __name__ == "__main__":
    main()
