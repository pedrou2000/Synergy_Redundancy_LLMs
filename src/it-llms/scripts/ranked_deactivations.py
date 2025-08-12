#!/usr/bin/env python
import multiprocessing as mp
import sys, os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import sys

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    from src.phyid_decomposition import MultiPromptPhyID, PromptPhyID, PhyIDTimeSeries
    from src.ranked_deactivation_analysis import RankedDeactivationAnalysis, RankedDeactivationResults, RankedDeactivationExperiment

    # ---------------- Load model and tokenizer ----------------
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.hf_name,
        torch_dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()                    # ← puts *every* sub‑module in eval mode
    torch.set_grad_enabled(False)   # optional but avoids autograd bookkeeping
    # model.config._attn_implementation = "flash_attention_2"

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.hf_name,
        use_fast=True,
        trust_remote_code=True,
        padding_side="left",
    )
    data_phyid_path = cfg.paths.data_phyid_dir

    # ---------------- Load phyid decomposition ----------------
    print(f"Loading phyid from {data_phyid_path}")
    phyid = MultiPromptPhyID.load_average_prompt_phyid(dir_path=data_phyid_path)
    node_ranking = phyid.syn_minus_red_rank
    phyid.plot_syn_minus_red_rank_per_node(node_ranking, plot_dir=cfg.paths.plot_phyid_dir)
    phyid.plot_syn_minus_red_rank_per_layer(node_ranking, plot_dir=cfg.paths.plot_phyid_dir)
    print(f"Synergy–Redundancy Rank of N03-L08: {node_ranking.sel(source_node=3, source_layer=8).values}")
    

    # ---------------- Run ranked deactivation experiment ----------------
    experiment = RankedDeactivationExperiment(
        analysis_kwargs=dict(
            model=model,
            tokenizer=tokenizer,
            prompts=OmegaConf.to_container(cfg.generation.prompts, resolve=True),
            chat_template=cfg.model.apply_chat_template,
            node_ranking=node_ranking,
            max_new_tokens=cfg.generation.max_new_tokens,
        )
    )
    experiment.run_default_and_random(
        deactivate_k_nodes_per_iteration=cfg.deactivation_analysis.deactivate_k_nodes_per_iteration,
        max_deactivated_nodes=cfg.deactivation_analysis.max_deactivated_nodes,
        micro_batch_size=cfg.deactivation_analysis.micro_batch_size,
        n_randomised_runs=cfg.deactivation_analysis.n_randomised_runs,
        reverse_kl=cfg.deactivation_analysis.reverse_kl,
        run_reverse_ranking=cfg.deactivation_analysis.run_reverse_ranking,
        noise_std=cfg.deactivation_analysis.noise_std,
        data_deactivation_dir=cfg.paths.data_deactivation_dir,
    )
    
    for fraction in cfg.deactivation_analysis.fractions_plot:
        experiment.plot_overall(
            plot_dir=cfg.paths.plot_deactivation_dir,
            aggregate_random=cfg.deactivation_analysis.aggregate_random,
            fraction=fraction,
        )
        experiment.plot_per_category(
            plot_dir=cfg.paths.plot_deactivation_dir,
            aggregate_random=cfg.deactivation_analysis.aggregate_random,
            fraction=fraction,
        )




# ----------------------------------------------------------------------
if __name__ == "__main__":                   
    mp.set_start_method("spawn", force=True) 
    main()
