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


from src.utils import perturb_model
from src.activation_recorder import ActivationRecorder, MultiPromptActivations
from src.time_series_activations import MultiPromptTimeSeries
from src.phyid_decomposition import MultiPromptPhyID

# -----------------------------------------------------------------------------
# Helper: run the *full* pipeline for one checkpoint
# -----------------------------------------------------------------------------

def run_for_checkpoint(step: int, base_cfg: Any) -> None:
    """Run recording → PhiID for a single training snapshot."""

    # Deep‑copy the OmegaConf so each run is clean
    cfg = OmegaConf.create(base_cfg)

    revision = f"step{step}"
    cfg.model.revision = revision  # <-- this automatically updates paths if
                                   #     they use ${model.revision}
    new_shortcode = f"{cfg.model.shortcode}-{step}"
    cfg.model.shortcode = new_shortcode
    cfg.model.it = f"base/steps/{step}"

    print(f"\n=== Processing checkpoint {revision} ===")

    # ---------------------------------------------------------------------
    # 1)  Load model & tokenizer
    # ---------------------------------------------------------------------
    model_name: str = cfg.model.hf_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
        # use_safetensors=True,
    )
    model.eval()

    # ---------------------------------------------------------------------
    # 2)  Record activations
    # ---------------------------------------------------------------------
    prompts: List[str] = cfg.generation.prompts
    # flatten [ [..], [..] ] → [..]
    if isinstance(prompts, list) and all(isinstance(p, list) for p in prompts):
        prompts = [item for sublist in prompts for item in sublist]

    recorder = ActivationRecorder(model, tokenizer)
    activations: MultiPromptActivations = recorder.record_prompts(
        prompts,
        max_new_tokens=cfg.generation.max_new_tokens,
        prompt_template=cfg.model.apply_chat_template,
    )
    activations.verify_recorded_activations(
        prompts=prompts,
        max_new_tokens=cfg.generation.max_new_tokens,
        tokenizer=tokenizer,
        diff_q_size=True,
    )
    activations.save(cfg.paths.data_activations_file)

    # ---------------------------------------------------------------------
    # 3)  Time‑series + PhiID
    # ---------------------------------------------------------------------
    time_series = MultiPromptTimeSeries.from_activations(
        activations,
        node_type=cfg.time_series.node_type,
        node_activation=cfg.time_series.node_activation,
        projection_method=cfg.time_series.projection_method,
        exclude_shared_expert_moe=cfg.time_series.exclude_shared_expert_moe,
    )
    time_series.plot(
        token_x=True,
        ticks_all_layers=True,
        plot_dir=cfg.paths.plot_time_series_dir,
    )

    phyid_comp = MultiPromptPhyID.from_time_series(
        time_series,
        cfg.phyid.tau,
        cfg.phyid.kind,
        cfg.phyid.redundancy,
    )
    phyid_comp.save(dir_path=cfg.paths.data_phyid_dir)

    # optional extra plots / metrics
    data_phyid_path = cfg.paths.data_phyid_dir
    phyid = MultiPromptPhyID.load(dir_path=data_phyid_path)
    print("Building data array for phyid")
    phyid.build_data_array()
    print("Computing average prompt phyid")
    phyid = phyid.compute_average_prompt_phyid(save_dir_path=cfg.paths.data_phyid_dir)

    phyid.plot_mean_along('sts', 'source_layer', plot_dir=cfg.paths.plot_phyid_dir)
    phyid.plot_mean_along('rtr', 'source_layer', plot_dir=cfg.paths.plot_phyid_dir)
    node_ranking = phyid.syn_minus_red_rank
    phyid.plot_syn_minus_red_rank_per_node(node_ranking, plot_dir=cfg.paths.plot_phyid_dir)
    phyid.plot_syn_minus_red_rank_per_layer(node_ranking, plot_dir=cfg.paths.plot_phyid_dir)
    print(f"Synergy–Redundancy Rank of N03-L08: {node_ranking.sel(source_node=3, source_layer=8).values}")

    # ---------------------------------------------------------------------
    # 4)  House‑keeping – free GPU RAM
    # ---------------------------------------------------------------------
    del model, recorder, activations, time_series, phyid_comp, tokenizer
    torch.cuda.empty_cache()
    gc.collect()


# -----------------------------------------------------------------------------
# Main entry point – iterate over the whole training trajectory
# -----------------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    
    steps = [2**i for i in range(0, 10)] + [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000]
    for step in steps:
        run_for_checkpoint(step, cfg)



if __name__ == "__main__":
    main()
