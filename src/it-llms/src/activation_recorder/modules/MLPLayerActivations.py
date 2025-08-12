"""
MLPLayerActivations.py

Holds a class for storing MLP-related activations at a single layer & step.
"""

import torch
from src.utils import ModelInformation

class MLPLayerActivations:
    """
    Stores MLP activations:
      - x_prime: post first linear + activation
      - y: final output of the MLP block
    """

    def __init__(self, model_info: ModelInformation, layer_index: int = None):
        """
        :param model_info: The ModelInformation describing the model
        """
        self.layer_index = layer_index
        self.model_info = model_info
        self.x_prime: torch.Tensor = None
        self.y: torch.Tensor = None

    def set_x_prime(self, x: torch.Tensor):
        self.x_prime = x

    def set_y(self, y: torch.Tensor):
        self.y = y
    
    def __repr__(self):
        return (f"MLPLayerActivations(layer_index={self.layer_index}, "
                f"x_prime={self.x_prime.shape if self.x_prime is not None else None}, "
                f"y={self.y.shape if self.y is not None else None}, "
                f"model_info={self.model_info})")
    
    def __str__(self):
        return f"L{self.layer_index}-MLP"


if __name__ == "__main__":
    """
    Simple usage: create an MLPLayerActivations and store some dummy data.
    """
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    from activation_recorder.structures.ModelInformation import ModelInformation

    info = ModelInformation(model)

    mlp_acts = MLPLayerActivations(info)
    mlp_acts.set_x_prime(torch.ones(5))
    mlp_acts.set_y(torch.zeros(5))

    print("MLP x_prime:", mlp_acts.x_prime)
    print("MLP y:", mlp_acts.y)
