"""
LayerActivations.py

Represents the activations at a single Transformer layer for a single step.
Can contain attention, MLP, and MoE structures.
"""

from typing import Optional
from src.activation_recorder.modules.AttentionLayerActivations import AttentionLayerActivations
from src.activation_recorder.modules.MLPLayerActivations import MLPLayerActivations
from src.activation_recorder.modules.MoELayerActivations import MoELayerActivations
from src.utils import ModelInformation

class LayerActivations:
    """
    For a single layer at a single step, stores sub-activations:
      - attention: AttentionLayerActivations
      - mlp: MLPLayerActivations
      - moe: MoELayerActivations
    """

    def __init__(self, layer_index: int, model_info: ModelInformation):
        """
        :param layer_index: The layer index in the model
        :param model_info: The ModelInformation describing the model
        """
        self.layer_index = layer_index
        self.model_info = model_info

        self.attention: Optional[AttentionLayerActivations] = None
        self.mlp: Optional[MLPLayerActivations] = None
        self.moe: Optional[MoELayerActivations] = None

    def get_or_create_attention(self) -> AttentionLayerActivations:
        if self.attention is None:
            self.attention = AttentionLayerActivations(self.model_info, self.layer_index)
        return self.attention

    def get_or_create_mlp(self) -> MLPLayerActivations:
        if self.mlp is None:
            self.mlp = MLPLayerActivations(self.model_info, self.layer_index)
        return self.mlp

    def get_or_create_moe(self) -> MoELayerActivations:
        if self.moe is None:
            self.moe = MoELayerActivations(self.model_info, self.layer_index)
        return self.moe
    
    def __len__(self):
        return len(self.attention)
    
    def is_complete(self):
        return len(self.attention) == self.model_info.num_attention_heads_per_layer

    def verify(self, diff_q_size: bool = False, prompt_len: int = 0, step_id: int = 0, max_new_tokens: int = 0):
        """ Verify the recorded activations for this layer. Checks shapes and values of attention and MLP activations."""
        if self.attention:
            self.attention.verify(diff_q_size, prompt_len, step_id, max_new_tokens)
        if self.moe:
            self.moe.verify(diff_q_size, prompt_len, step_id, max_new_tokens)
    
    def uncompress_moe_activations(self, node_activation: str = "expert_output") -> None:
        """ Uncompress MoE activations by filling missing nodes with zeros. """
        if self.moe:
            self.moe.uncompress_activations(node_activation)

if __name__ == "__main__":
    """
    Simple demonstration: create a LayerActivations, add attention & MLP sub-structures, and check.
    """
    from transformers import AutoModelForCausalLM
    from activation_recorder.structures.ModelInformation import ModelInformation

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    info = ModelInformation(model)

    layer_acts = LayerActivations(layer_index=0, model_info=info)
    attn = layer_acts.get_or_create_attention()
    mlp = layer_acts.get_or_create_mlp()

    print("LayerActivations for layer=0 created.")
    print("Attention object:", attn)
    print("MLP object:", mlp)
