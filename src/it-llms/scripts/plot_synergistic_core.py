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
from pathlib import Path
from typing import List
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys, os

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from src.activation_recorder import MultiPromptActivations
from src.time_series_activations import MultiPromptTimeSeries
from src.phyid_decomposition import MultiPromptPhyID, PromptPhyID, PhyIDTimeSeries

# -----------------------------------------------------------------------------
# Helper: run the *full* pipeline for one checkpoint
# -----------------------------------------------------------------------------



@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    cfg = OmegaConf.create(cfg)

    print(f"Loading data activations from {cfg.paths.data_activations_file}", flush=True)
    data_activations_file = cfg.paths.data_activations_file
    activations = MultiPromptActivations.load(file_path=data_activations_file)

    print(f"Computing time series from activations", flush=True)
    time_series = MultiPromptTimeSeries.from_activations(
        activations, 
        node_type=cfg.time_series.node_type,
        node_activation=cfg.time_series.node_activation,
        projection_method=cfg.time_series.projection_method, 
        exclude_shared_expert_moe=cfg.time_series.exclude_shared_expert_moe, 
    )

    print(f"Plotting time series", flush=True)
    ts = time_series.prompts[5]
    ts.plot_two_series(series_a=(25,6), series_b=(25,7), tokens=range(0,30), plot_dir=cfg.paths.plot_time_series_dir)


if __name__ == "__main__":
    main()
