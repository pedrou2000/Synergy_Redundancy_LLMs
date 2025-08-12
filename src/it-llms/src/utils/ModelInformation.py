"""
ModelInformation.py

Holds a ModelInformation class that extracts key metadata directly from a
Hugging Face model, including number of layers, attention heads, etc.
"""

from transformers import PreTrainedModel
from dataclasses import dataclass, asdict, field
from typing import Optional

@dataclass
class ModelInformation:
    model_name: str = ""
    model_architecture: str = ""
    num_layers: int = 0
    num_attention_heads_per_layer: int = 0
    total_num_attention_heads: int = 0
    hidden_size: int = 0
    head_dim: Optional[int] = None
    n_routed_experts: Optional[int] = None
    n_shared_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    attention_implementation: str = "default"

    def __init__(self, hf_model: Optional[PreTrainedModel] = None):
        if hf_model is None:
            return  # Allows dummy construction for from_dict

        config = hf_model.config
        self.model_architecture = config.architectures[0] if getattr(config, "architectures", []) else ""
        self.model_name = getattr(config, "name_or_path", "")
        self.num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", 0))
        self.num_attention_heads_per_layer = getattr(config, "num_attention_heads", getattr(config, "n_head", 0))
        self.total_num_attention_heads = self.num_layers * self.num_attention_heads_per_layer
        self.hidden_size = getattr(config, "hidden_size", getattr(config, "n_embd", 0))
        if getattr(config, "head_dim", None) is not None:
            self.head_dim = config.head_dim
        elif getattr(config, "v_head_dim", None) is not None:
            self.head_dim = config.v_head_dim
        elif self.hidden_size > 0 and self.num_attention_heads_per_layer > 0:
            self.head_dim = self.hidden_size // self.num_attention_heads_per_layer
        else:
            raise ValueError("Cannot determine head_dim from model config")
        self.n_routed_experts = getattr(config, "n_routed_experts", None)
        self.n_shared_experts = getattr(config, "n_shared_experts", None)
        self.num_experts_per_tok = getattr(config, "num_experts_per_tok", None)
        self.attention_implementation = getattr(config, "attn_implementation", "default")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelInformation":
        obj = cls.__new__(cls)
        for k, v in d.items():
            setattr(obj, k, v)
        return obj
