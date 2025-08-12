"""
MultiPromptActivations.py

Holds the top-level container for storing activations across multiple prompts.
Each prompt has its own PromptActivations object.
"""

from __future__ import annotations
import os, pickle, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import Dict, List
from src.activation_recorder.PromptActivations import PromptActivations
from src.utils import ModelInformation

class MultiPromptActivations:
    """
    Top-level container for all prompt-based activations, keyed by prompt_id.
    """

    def __init__(self, model_info: ModelInformation):
        """
        :param model_info: The ModelInformation object describing the model.
        """
        self.model_info = model_info
        self.prompts: Dict[int, PromptActivations] = {}

    def get_or_create_prompt_activations(self, prompt_id: int, prompt_text: str, completion_text: str = None, completion_tokens: List[str] = None) -> PromptActivations:
        """
        Retrieves or creates a PromptActivations object for the given prompt_id.
        """
        if prompt_id not in self.prompts:
            self.prompts[prompt_id] = PromptActivations(
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                completion_text=completion_text,
                completion_tokens=completion_tokens,
                model_info=self.model_info
            )
        return self.prompts[prompt_id]

    def get_prompt_activations(self, prompt_id: int) -> PromptActivations:
        """
        Return the PromptActivations for a given prompt_id (if already created).
        """
        return self.prompts[prompt_id] 
    
    def __len__(self):
        """
        Return the number of prompts recorded.
        """
        return len(self.prompts)

    def save(self, file_path: str) -> None:
        """
        Save the MultiPromptActivations object to a pickle file within the specified directory.
        If the directory does not exist, it will be created.

        :param dir_path: Directory where the pickle file will be saved.
        """
        dir_path = os.path.dirname(file_path)
        try:
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            with open(file_path, "wb") as f:
                # pickle.dump(self, f)
                torch.save(self, f) 
            print(f"MultiPromptActivations successfully saved to '{file_path}'.")
        except Exception as e:
            print(f"Error while saving MultiPromptActivations to '{dir_path}': {e}")
            raise

    @classmethod
    def load(cls, file_path: str) -> MultiPromptActivations:
        """
        Load and return a MultiPromptActivations object from the specified pickle file.

        :param file_path: Full path to the pickle file containing the saved activations.
        :return: The loaded MultiPromptActivations object.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"No file found at '{file_path}'.")
        try:
            with open(file_path, "rb") as f:
                # loaded_obj = pickle.load(f)
                loaded_obj = torch.load(f, map_location='cpu', weights_only=False)
            if not isinstance(loaded_obj, cls):
                print(f"Loaded object is not a MultiPromptActivations instance. Got type: {type(loaded_obj)}")
            print(f"MultiPromptActivations successfully loaded from '{file_path}'.")
            return loaded_obj
        except Exception as e:
            print(f"Error while loading MultiPromptActivations from '{file_path}': {e}")
            raise

    def verify_recorded_activations(self, prompts: dict, max_new_tokens: int = None, tokenizer: AutoTokenizer = None, diff_q_size: bool = False):
        """ Verify the recorded activations are correct in shape and value. """
        activations = self
        prompts = [prompt for prompt_list in prompts.values() for prompt in prompt_list]

        # Check shape of activations
        assert len(activations) == len(prompts), f'Expected {len(prompts)} prompts, got {len(activations)}'
        assert len(activations.prompts[0].steps) == max_new_tokens, f'Expected {max_new_tokens} steps, got {len(activations.prompts[0].steps)}'
        assert len(activations.prompts[0].steps[0].layers) == activations.model_info.num_layers, f'Expected {activations.model_info.num_layers} layers, got {len(activations.prompts[0].steps[0].layers)}'
        assert len(activations.prompts[0].steps[0].layers[0].attention.heads) == activations.model_info.num_attention_heads_per_layer, f'Expected {activations.model_info.num_attention_heads_per_layer} heads, got {len(activations.prompts[0].steps[0].layers[0].attention.heads)}'
        
        # Check shape of a particular head activation
        max_len_prompt = 0
        for prompt_id , prompt_activations in activations.prompts.items():
            prompt_len = len(tokenizer.encode(prompt_activations.prompt_text))
            max_len_prompt = max(max_len_prompt, prompt_len)
            prompt_activations.verify(diff_q_size=diff_q_size, prompt_len=prompt_len, max_new_tokens=max_new_tokens)


        # Print a success message
        print(f'Activations check passed!')
    
    def uncompress_moe_activations(self, node_activation: str = "expert_output") -> None:
        """ Uncompress MoE activations by filling missing nodes with zeros across all prompts. """
        for prompt_id, prompt_activations in self.prompts.items():
            prompt_activations.uncompress_moe_activations(node_activation)
