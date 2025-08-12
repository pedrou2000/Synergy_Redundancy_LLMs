"""
MoELayerActivations.py

Holds classes for storing Mixture-of-Experts (MoE) activations,
including gating info and per-expert data.
"""

import torch
from typing import Dict
from src.utils import ModelInformation

class MoEExpertActivations:
    """
    Stores data for a single expert in an MoE layer:
      - expert_index
      - x_prime
      - x_double_prime
      - beta (gating weight, etc.)
      - y (final output)
    """

    def __init__(
        self,
        gate_value: torch.Tensor = None,
        mlp_output: torch.Tensor = None,
        expert_output: torch.Tensor = None,
        is_shared: bool = False,
        model_info: ModelInformation = None,
        layer_index: int = None,
        expert_index: int = None
    ):
        self.model_info = model_info
        self.layer_index = layer_index
        self.expert_index = expert_index

        self.gate_value = gate_value  # Gating value for this expert
        self.mlp_output = mlp_output  # Output after the MLP
        self.expert_output = expert_output  # Final output from the expert (gate_valye * mlp_output)
        self.is_shared = is_shared  # Whether this expert shares weights with others

    def __repr__(self):
        return (f"MoEExpertActivations(layer_index={self.layer_index}, "
                f"expert_index={self.expert_index}, "
                f"gate_value={self.gate_value.shape if self.gate_value is not None else None}, "
                f"mlp_output={self.mlp_output.shape if self.mlp_output is not None else None}, "
                f"expert_output={self.expert_output.shape if self.expert_output is not None else None}, "
                f"is_shared={self.is_shared}, "
                f"model_info={self.model_info})")
    
    def __str__(self):
        return f"L{self.layer_index}-E{self.expert_index}" + "S" if self.is_shared else ""

    def verify(self, diff_q_size: bool = False, prompt_len: int = 0, step_id: int = 0, max_new_tokens: int = 0):
        """ Verify the recorded activations for this expert. Checks shapes and values of expert activations. """
        if self.mlp_output is None or self.expert_output is None:
            # Empty expert case
            assert self.mlp_output is None and self.expert_output is None, "Both mlp_output and expert_output must be set if one is set."
            assert self.gate_value is None or self.gate_value == torch.tensor(0), "If mlp_output and expert_output are not set, gate_value must be 0."
            return
        
        # gate value should be a scalar tensor
        assert self.gate_value.dim() == 0, f"Expected gate_value to be a scalar tensor, got shape {self.gate_value.shape} with value {self.gate_value.item()}"
        assert self.mlp_output.shape == (self.model_info.hidden_size,), f'Expected mlp_output shape ({self.model_info.hidden_size},), got {self.mlp_output.shape}'
        assert self.expert_output.shape == (self.model_info.hidden_size,), f'Expected expert_output shape ({self.model_info.hidden_size},), got {self.expert_output.shape}'
    
    @classmethod
    def empty(cls, layer_index: int, expert_index: int, is_shared: bool = False, model_info: ModelInformation = None):
        return cls(
            gate_value=None,
            mlp_output=None,
            expert_output=None,
            is_shared=is_shared,
            model_info=model_info,
            layer_index=layer_index,
            expert_index=expert_index
        )
    @property
    def is_empty(self) -> bool:
        return self.mlp_output is None and self.expert_output is None


 

class MoELayerActivations:
    """
    Container for multiple experts in a single MoE layer at a single step.
    """

    def __init__(self, model_info: ModelInformation, layer_index: int = None):
        self.layer_index = layer_index
        self.model_info = model_info
        self.experts: Dict[int, MoEExpertActivations] = {}
    
    @property
    def nodes(self) -> Dict[int, MoEExpertActivations]:
        """ Return the dictionary of expert activations. """
        return self.experts

    def add_expert_activations(self, expert_acts: MoEExpertActivations):
        idx = expert_acts.expert_index
        self.experts[idx] = expert_acts

    def get_expert_activations(self, expert_index: int) -> MoEExpertActivations:
        return self.experts[expert_index]
    
    def __len__(self):
        return len(self.experts)
    
    def is_complete(self):
        return len(self) == self.model_info.n_routed_experts + self.model_info.n_shared_experts
    
    def __repr__(self):
        return (f"MoELayerActivations(layer_index={self.layer_index}, "
                f"num_experts={len(self.experts)}, "
                f"model_info={self.model_info})")

    def __str__(self):
        return f"L{self.layer_index} MoE with {len(self.experts)} experts"
    
    def verify(self, diff_q_size: bool = False, prompt_len: int = 0, step_id: int = 0, max_new_tokens: int = 0):
        """ Verify the recorded activations for this MoE layer. Checks shapes and values of expert activations. """
        n_active_experts = 0
        for expert_index, expert in self.experts.items():
            expert.verify(diff_q_size, prompt_len, step_id, max_new_tokens)
            if not expert.is_empty:
                n_active_experts += 1
        assert n_active_experts == self.model_info.num_experts_per_tok + 1, f'Expected {self.model_info.n_routed_experts + self.model_info.n_shared_experts} active experts, got {n_active_experts}'

    def uncompress_activations(self, node_activation: str = "expert_output") -> None:
        """ Fill in missing expert activations with zeros for consistency. """
        # Start by filling the shapes of each of the moe activations with zeros
        act_shape = None
        for expert_index, expert_acts in self.experts.items():
            if hasattr(expert_acts, node_activation) and getattr(expert_acts, node_activation) is not None:
                act_shape = getattr(expert_acts, node_activation).shape
                break
        
        # Now iterate through all experts and fill in the missing activations with zeros
        for expert_index, expert_acts in self.experts.items():
            if hasattr(expert_acts, node_activation) and getattr(expert_acts, node_activation) is None:
                zero_tensor = torch.zeros(act_shape, dtype=torch.float32)
                setattr(expert_acts, node_activation, zero_tensor)