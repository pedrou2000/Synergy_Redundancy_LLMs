"""
AttentionLayerActivations.py

Holds classes related to storing attention head activations
and an attention container for a single layer & step.
"""

import torch
from typing import Dict
from src.utils import ModelInformation

class AttentionHeadActivations:
    """
    Stores raw data for a single attention head:
      - query
      - attention_weights
      - attention_outputs (before final projection)
      - projected_outputs (after projection)
    """

    def __init__(
        self,
        query: torch.Tensor,
        attention_weights: torch.Tensor,
        attention_outputs: torch.Tensor,
        projected_outputs: torch.Tensor,
        model_info: ModelInformation = None,
        layer_index: int = None,
        head_index: int = None
    ):
        self.layer_index = layer_index
        self.head_index = head_index
        self.model_info = model_info
        self.query = query
        self.attention_weights = attention_weights
        self.attention_outputs = attention_outputs
        self.projected_outputs = projected_outputs
    
    def __repr__(self):
        return (f"AttentionHeadActivations(query={self.query.shape}, "
                f"attention_weights={self.attention_weights.shape}, "
                f"attention_outputs={self.attention_outputs.shape}, "
                f"projected_outputs={self.projected_outputs.shape}, "
                f"model_info={self.model_info})")
    def __str__(self):
        return f"L{self.layer_index}-H{self.head_index}"

    def verify(self, diff_q_size: bool = False, prompt_len: int = 0, step_id: int = 0, max_new_tokens: int = 0):
        """ Verify the recorded activations are correct in shape and value. """
        if not diff_q_size:
            assert self.query.shape == (self.model_info.head_dim,), f'Expected query shape ({self.model_info.head_dim},), got {self.query.shape}'
        # assert head.attention_weights.shape == (max_len_prompt + self.max_new_tokens - 1,), f'Expected attention_weights shape ({max_len_prompt + max_new_tokens - 1},), got {head.attention_weights.shape}, for head {head_id} in layer {layer_id} at step {step_id}'
        assert torch.all(self.attention_weights[prompt_len + step_id:] == 0), f'Expected attention_weights to be zero after prompt_len + step_id, got {self.attention_weights[prompt_len + step_id:]} for head {self.head_index} in layer {self.layer_index}'
        assert self.attention_outputs.shape == (self.model_info.head_dim,), f'Expected attention_outputs shape ({self.model_info.head_dim},), got {self.attention_outputs.shape} for head {self.head_index} in layer {self.layer_index}'
        assert self.projected_outputs.shape == (self.model_info.hidden_size,), f'Expected projected_outputs shape ({self.model_info.hidden_size},), got {self.projected_outputs.shape} for head {self.head_index} in layer {self.layer_index}'


class AttentionLayerActivations:
    """
    Container for all attention heads in a single layer at a single step.
    """

    def __init__(self, model_info: ModelInformation, layer_index: int = None):
        self.model_info = model_info
        self.layer_index = layer_index
        self.heads: Dict[int, AttentionHeadActivations] = {}
    
    @property 
    def nodes(self) -> Dict[int, AttentionHeadActivations]:
        """ Return the dictionary of head activations. """
        return self.heads

    def add_head_activations(self, head_acts: AttentionHeadActivations):
        idx = head_acts.head_index
        self.heads[idx] = head_acts

    def get_head_activations(self, index: int) -> AttentionHeadActivations:
        return self.heads[index]

    def __len__(self):
        return len(self.heads)
    
    def is_complete(self):
        return len(self) == self.model_info.num_attention_heads_per_layer

    def __repr__(self):
        return f"AttentionLayerActivations(num_heads={len(self.heads)}, model_info={self.model_info})"
    
    def __str__(self):
        return f"L{self.layer_index} with {len(self.heads)} heads"
    
    def verify(self, diff_q_size: bool = False, prompt_len: int = 0, step_id: int = 0, max_new_tokens: int = 0):
        """ Verify the recorded activations for all heads in this layer. """
        for head_id, head in self.heads.items():
            head.verify(diff_q_size, prompt_len, step_id, max_new_tokens)


if __name__ == "__main__":
    """
    Simple test of creating some dummy attention heads and storing them.
    """
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    from activation_recorder.structures.ModelInformation import ModelInformation

    info = ModelInformation(model)
    attn_layer = AttentionLayerActivations(info)

    head1 = AttentionHeadActivations(
        query=torch.randn(4),
        key=torch.randn(4),
        value=torch.randn(4),
        attention_weights=torch.randn(8),
        attention_outputs=torch.randn(4),
        output=torch.randn(4),
        model_info=info
    )

    attn_layer.add_head_activations(head1)
    print("Number of heads:", len(attn_layer.heads))
    print("First head's query vector:", attn_layer.get_head_activations(0).query)
