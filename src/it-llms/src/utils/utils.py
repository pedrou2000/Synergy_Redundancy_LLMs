from typing import Dict, List, Union, Tuple
import torch
import xarray as xr

def get_layer_node_indeces(node_idx: int, num_nodes_per_layer: int) -> tuple[int, int]:
    """
    Get the layer and node indices for a given node index in the model.

    :param model_info: ModelInformation object containing model details.
    :param node_idx: The node index to convert.
    :return: A tuple containing (layer_index, node_index).
    """
    layer_index = node_idx // num_nodes_per_layer
    node_index = node_idx % num_nodes_per_layer
    return layer_index, node_index

def get_node_index(layer_index: int, node_index: int, num_nodes_per_layer: int) -> int:
    """
    Get the node index for a given layer and node indices in the model.

    :param model_info: ModelInformation object containing model details.
    :param layer_index: The layer index.
    :param node_index: The node index within that layer.
    :return: The computed node index.
    """
    return layer_index * num_nodes_per_layer + node_index

def get_layer_modules(num_nodes_per_layer, num_layers) -> List[List[int]]:
    """
    One sub-list per Transformer layer, each containing the *flat* node
    indices of every attention head in that layer.
    """
    return [
        list(range(layer_index * num_nodes_per_layer, (layer_index + 1) * num_nodes_per_layer))
        for layer_index in range(num_layers)
    ]

def perturb_model(model, scale=10.0):
    for name, param in model.named_parameters():
        if param.requires_grad:
            noise = torch.randn_like(param) * scale
            param.data.add_(noise)


def randomize_model_weights(model, mean=0.0, std=0.02):
    for name, param in model.named_parameters():
        if param.requires_grad:
            torch.nn.init.normal_(param, mean=0.0, std=0.02)  # Or use other initializations


def invert_node_ranking(
    node_ranking: xr.DataArray,
    dim_layer: str = "source_layer",
    dim_node: str = "source_node",
    with_scores: bool = False
) -> Union[List[Tuple[int, int]], List[Tuple[Tuple[int, int], float]]]:
    """
    Given a DataArray of shape (n_layers, n_nodes) indexed by (dim_layer, dim_node),
    returns the list of (layer, node) pairs sorted from highest to lowest value.

    Parameters
    ----------
    node_ranking : xr.DataArray
        2‑D array with dims (dim_layer, dim_node).
    dim_layer : str
        Name of the layer dimension in node_ranking.
    dim_node : str
        Name of the node dimension in node_ranking.
    with_scores : bool
        If True, return [ ((layer, node), score), ... ].
        Otherwise, return just [ (layer, node), ... ].

    Returns
    -------
    List of tuples
        Sorted by descending rank.
    """
    # 1. Stack into one MultiIndex dimension
    stacked = node_ranking.stack(
        all_nodes=(dim_layer, dim_node)
    )

    # 2. Convert to pandas Series and sort
    sorted_series = stacked.to_series().sort_values(ascending=False)

    if with_scores:
        # [ ((layer, node), score), ... ]
        return list(zip(sorted_series.index.tolist(),
                        sorted_series.values.tolist()))
    else:
        # [ (layer, node), ... ]
        return sorted_series.index.tolist()
