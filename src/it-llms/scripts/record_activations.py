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

    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig
    from src.utils import perturb_model, randomize_model_weights

    load_model = True

    model_name = cfg.model.hf_name
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True
    )
    if load_model:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map='auto', 
            attn_implementation='eager',  
            trust_remote_code=True,
            revision=cfg.model.revision if hasattr(cfg.model, 'revision') else None,
        )
        if cfg.model.it == 'random':
            # randomize_model_weights(model, mean=0.0, std=0.02)
            perturb_model(model, scale=10.0)
        # model.generation_config = GenerationConfig.from_pretrained(model_name)
        # model.generation_config.pad_token_id = model.generation_config.eos_token_id
        model.eval()
    print(type(model))
    print(model)
    print(model.config)


    from src.activation_recorder import ActivationRecorder, MultiPromptActivations

    load_from_disk = False
    data_activations_file = cfg.paths.data_activations_file
    max_new_tokens=cfg.generation.max_new_tokens
    prompts = cfg.generation.prompts
    # if it not a list but a list of lists, flatten it
    if isinstance(prompts, list) and all(isinstance(p, list) for p in prompts):
        prompts = [item for sublist in prompts for item in sublist]

    prompt_template = cfg.model.apply_chat_template
    print(f"Using prompt_template: {prompt_template}")

    if not load_from_disk:
        recorder = ActivationRecorder(model, tokenizer)
        activations = recorder.record_prompts(prompts, max_new_tokens=max_new_tokens, prompt_template=prompt_template)
        print(f"Recorded {len(activations)} activations for {len(prompts)} prompts.")
        activations.save(data_activations_file)
        activations.verify_recorded_activations(prompts=prompts, max_new_tokens=max_new_tokens, tokenizer=tokenizer, diff_q_size=True)


if __name__ == "__main__":                   
    main()
