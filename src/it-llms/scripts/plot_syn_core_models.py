#!/usr/bin/env python
"""
Plot synergy-minus-redundancy rank for several models on a common
0-to-1 depth axis (x) and 0-to-1 min-max normalised score (y).

Models are listed in the Hydra config (`cfg.compare.models`).

All heavy path logic stays in your `paths.yaml`; we simply mutate
`cfg.model` for each model name and let the interpolations do
their job.
"""
from __future__ import annotations
import copy, os, numpy as np, matplotlib.pyplot as plt, xarray as xr
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra

# --- project imports ---------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
import sys
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from src.phyid_decomposition import MultiPromptPhyID       # <- your module

# -----------------------------------------------------------------------------


def _resolve(cfg: DictConfig, dotted: str):
    out = cfg
    for part in dotted.split("."):
        out = out[part]
    return out


def plot_models(cfg: DictConfig) -> None:
    models = cfg.compare.models
    rank_key       = cfg.compare.phyid_rank_key
    dir_field      = cfg.compare.phyid_dir_field

    series = []
    for name in models:
        # ------------------------------------------------------------------
        # 1.  Deep-copy the *base* cfg (keeps generation, phyid, etc.)
        m_cfg = copy.deepcopy(cfg)

        # 2.  Load the relevant model YAML and plug it in wholesale
        model_yaml = os.path.join(project_root, "config", "model", f"{name}.yaml")
        if not os.path.isfile(model_yaml):
            print(f"⚠  {name}: no YAML at {model_yaml}, skipping")
            continue
        m_cfg.model = OmegaConf.load(model_yaml)

        # 3.  Resolve ${...} interp olations so paths update
        OmegaConf.resolve(m_cfg)            # ← crucial

        # 4.  Now you can use the paths
        phy_dir = _resolve(m_cfg, dir_field)
        if not os.path.isdir(phy_dir):
            print(f"⚠  {name}: {phy_dir} missing, skipping")
            continue
            
        phy = MultiPromptPhyID.load_average_prompt_phyid(dir_path=phy_dir)
        rank_da    = getattr(phy, rank_key)
        layer_mean = rank_da.mean(dim="source_node")
        y = layer_mean.values.astype(float)
        y = (y - y.min()) / (np.ptp(y) or 1.0)          # np.ptp = max-min
        x = np.linspace(0, 1, len(y))
        series.append((m_cfg.model.plot_name, x, y))
        print(f"✓ {name} loaded")

    # ------------------------------------------------ plot
    if not series:
        print("Nothing to plot.")
        return

    # build output dir once from *any* valid cfg (use the last one)
    out_dir = os.path.join(
        _resolve(m_cfg, "paths.project_root"),
        "plots", "multi_model",
        _resolve(m_cfg, "generation.name"),
        "phyid", _resolve(m_cfg, "paths.phyid_method"),
    )
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "syn_minus_red_rank__x_norm_0to1.svg")

    plt.figure(figsize=(10,5))
    for name, x, y in series:
        plt.plot(x, y, marker="o", linewidth=2, label=name)
    plt.xlim(0,1); plt.ylim(-0.05,1.05)
    plt.xlabel("Normalised layer depth")
    plt.ylabel("Normalised synergy–redundancy rank")
    # plt.title("Synergy–Redundancy Rank per Layer")
    plt.grid(axis="y", linestyle="--", alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"✅  Saved {out_png}")


# -----------------------------------------------------------------------------


@hydra.main(config_path="../config", config_name="compare_syn_red", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    The Hydra config `compare_syn_red.yaml` extends your normal defaults
    and just adds:

    compare:
        models: [L32-1, L32-1-IT, L32-3, L32-3-IT, L31-8, L31-8-IT]
        phyid_rank_key: syn_minus_red_rank
        phyid_dir_field: paths.data_phyid_dir
    """
    print("Base config loaded:\n" + OmegaConf.to_yaml(cfg, resolve=True))
    plot_models(cfg)


if __name__ == "__main__":
    main()
