from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Optional, Tuple, Dict, Union
import torch
import torch.nn as nn
from contextlib import contextmanager


def get_model_part_modules(model: PreTrainedModel, module_name: str = "self_attn") -> dict[int, nn.Module]:
    """
    Recursively collects all submodules whose name contains the specified `module_name`.
    
    :param model: A Hugging Face PreTrainedModel.
    :param module_name: Substring to search for in submodule names.
    :return:  A dictionary mapping layer indices to the corresponding modules.
    """
    modules: dict[int, nn.Module] = {}

    for name, module in model.named_modules():
        if module_name in name and hasattr(module, 'layer_idx'):
            layer_idx = getattr(module, 'layer_idx', None)
            modules[layer_idx] = module

    return modules

def group_nodes_by_layer(nodes_to_deactivate: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    grouped: Dict[int, List[int]] = {}
    for layer, node in nodes_to_deactivate:
        if layer not in grouped:
            grouped[layer] = []
        grouped[layer].append(node)
    return grouped

class ModuleDeactivator:
    @staticmethod
    def deactivate_self_attn(module: nn.Module, nodes: List[int], noise_std: Optional[float] = None):
        """
        Deactivate self-attention heads in the module by setting their weights to zero.
        
        :param module: The self-attention module to modify.
        :param nodes: List of node indices (heads) to deactivate.
        """
        if not hasattr(module, 'set_deactivated_heads'):
            raise ValueError("Module does not support deactivation. Implement `set_deactivated_heads`.")
        module.set_deactivated_heads(nodes, noise_std)


    @staticmethod
    def deactivate_mlp(module: nn.Module, nodes: List[int], noise_std: Optional[float] = None):
        raise NotImplementedError()
    
    @staticmethod
    def deactivate_selected_nodes(module: nn.Module, nodes: List[int], module_type: str = "self_attn", noise_std: Optional[float] = None):
        """
        Deactivate specific nodes in the module.

        :param module: The module to modify.
        :param nodes: List of node indices to deactivate.
        :param module_type: Type of the module (e.g., "self_attn", "mlp").
        """
        fn = getattr(ModuleDeactivator, f"deactivate_{module_type}", None)
        if fn is None:
            raise ValueError(f"Unsupported module type: {module_type}")
        return fn(module=module, nodes=nodes, noise_std=noise_std)



@contextmanager
def deactivate_model_parts(
    model: PreTrainedModel,
    nodes_to_deactivate: List[Tuple[int, int]],
    module_name: str = "self_attn", # "self_attn", "mlp", ...
    noise_std: Optional[float] = None
):
    """
    Temporarilly deactivate specific nodes in the model by setting their weights to zero.
    
    :param model: The pre-trained model to modify.
    :param nodes_to_deactivate: List of tuples (layer_index, node_index) indicating which nodes to deactivate.
    :param module_name: The name of the module where the nodes are located (e.g., "self_attn", "mlp").
    """

    modules_to_deactivate = get_model_part_modules(model, module_name)
    nodes_by_layer = group_nodes_by_layer(nodes_to_deactivate)

    for layer_idx, nodes in nodes_by_layer.items():
        if layer_idx not in modules_to_deactivate:
            print(f"Warning: No module found for layer {layer_idx}. Skipping deactivation.")
            continue
        
        module = modules_to_deactivate[layer_idx]
        if not hasattr(module, 'head_dim'):
            print(f"Warning: Module {module} does not have 'head_dim' attribute. Cannot proceed with deactivation.")
            continue

        ModuleDeactivator.deactivate_selected_nodes(
            module=module,
            nodes=nodes,
            module_type=module_name, # e.g., "self_attn", "mlp"
            noise_std=noise_std
        )

    try:
        yield model
    finally:
        for layer_idx, nodes in nodes_by_layer.items():
            if layer_idx not in modules_to_deactivate:
                continue
            
            module = modules_to_deactivate[layer_idx]
            if hasattr(module, 'clear_deactivated_heads'):
                module.clear_deactivated_heads()
            elif hasattr(module, 'deactivated_heads'):
                module.deactivated_heads = []
            else:
                print(f"Warning: Module {module} does not support clearing deactivated heads. Skipping reset.")