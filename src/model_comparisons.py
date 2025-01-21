import matplotlib.pyplot as plt
import numpy as np
import os
import constants
from phid import *
from transformers import AutoTokenizer, AutoConfig 

def load_final_models_gradient_ranks():

    meta_gradient_ranks = dict()
    prompt_category_name = 'average_prompts'

    for constants.MODEL_CODE in constants.FINAL_MODELS:
        constants.MODEL_NAME = constants.MODEL_NAMES[constants.MODEL_CODE]["HF_NAME"]
        constants.FOLDER_MODEL_NAME = constants.MODEL_NAMES[constants.MODEL_CODE]["FOLDER_NAME"]
        constants.SAVED_DATA_DIR = "../data/" + constants.FOLDER_MODEL_NAME + "/"
        constants.MATRICES_DIR = constants.SAVED_DATA_DIR + "3-Synergy_Redundancy_Matrices/"
        base_plot_path = constants.PLOTS_SYNERGY_REDUNDANCY_DIR + prompt_category_name + '/'
        
        # print("Loading Model: ", constants.MODEL_NAME, "Number of heads per layer: ", constants.NUM_HEADS_PER_LAYER)
        global_matrices, synergy_matrices, redundancy_matrices = load_matrices(base_save_path=constants.MATRICES_DIR + prompt_category_name + '/' + prompt_category_name + '.pt')
        averages = calculate_average_synergy_redundancies_per_head(synergy_matrices, redundancy_matrices, within_layer=False)
        gradient_ranks = compute_gradient_rank(averages)
        meta_gradient_ranks[constants.MODEL_CODE] = gradient_ranks
    return meta_gradient_ranks


def normalize_values(values):
    """Normalize a list of values using min-max scaling to [0, 1]."""
    min_val, max_val = np.min(values), np.max(values)
    return (values - min_val) / (max_val - min_val), min_val, max_val

def plot_gradient_rank_overlay(models_gradient_ranks, figsize=None, save=False, base_plot_path=constants.MODEL_COMPARISON_GRADIENT_RANK_DIR + "1-gradient_ranks.jpg", plot_intralayer_std=False):
    """
    Create an overlay plot of normalized synergy-minus-redundancy ranks for multiple models,
    including confidence bands (standard deviation shading).

    Args:
        models_gradient_ranks: Dictionary where keys are model names and values are gradient_ranks.
        save: Whether to save the plot to a file.
        base_plot_path: File path for saving the plot if save=True.
    """
    if figsize is None:
        figsize = (16, 6) if not plot_intralayer_std else (16, 8)
    plt.figure(figsize=figsize)

    for model_code, gradient_ranks in models_gradient_ranks.items():
        # Compute mean and standard deviation of synergy-minus-redundancy rank per layer
        head_ranks = gradient_ranks['attention_outputs']
        heads = list(head_ranks.keys())
        ranks = [head_ranks[head] for head in heads]

        config = AutoConfig.from_pretrained(constants.MODEL_NAMES[model_code]["HF_NAME"])
        constants.NUM_HEADS_PER_LAYER = config.num_attention_heads if hasattr(config, 'num_attention_heads') else config.n_head
        num_heads_per_layer = constants.NUM_HEADS_PER_LAYER

        num_layers = len(ranks) // num_heads_per_layer
        ranks_array = np.array(ranks).reshape(num_layers, num_heads_per_layer)
        layer_means = np.mean(ranks_array, axis=1)
        layer_stds = np.std(ranks_array, axis=1)

        # Normalize mean values and determine scaling parameters
        normalized_y, min_val, max_val = normalize_values(layer_means)

        # Adjust standard deviation for normalized scale
        normalized_std = layer_stds / (max_val - min_val)

        # Normalize x-axis (layer index)
        normalized_x = np.linspace(0, 1, num_layers)

        # Plot the normalized mean with confidence band
        plt.plot(normalized_x, normalized_y, label=constants.MODEL_NAMES[model_code]["PLOT_NAME"])
        if plot_intralayer_std:
            plt.fill_between(normalized_x, normalized_y - normalized_std, normalized_y + normalized_std, alpha=0.2)

    # Plot customization
    plt.xlabel("Normalized Layer Index")
    plt.ylabel("Normalized Synergy - Redundancy Rank")
    plt.title("Synergistic Information Processing Core Across LLMs")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Save or show the plot
    if save:
        os.makedirs(os.path.dirname(base_plot_path), exist_ok=True)
        plt.savefig(base_plot_path)
    else:
        plt.show()
    plt.close()
