from phyid.calculate import calc_PhiID 
import numpy as np
import os
import matplotlib.pyplot as plt
import constants
from datetime import datetime

def compute_synergy_redundancy_PhiID(time_series, metrics, tau=1, kind="gaussian", redundancy_measure="MMI"):
    synergy_matrices = {}
    redundancy_matrices = {}
    for metric in metrics:
        num_layers = len(time_series[metric])
        num_heads_per_layer = len(time_series[metric][0])
        total_heads = num_layers * num_heads_per_layer
        global_synergy_matrix = np.zeros((total_heads, total_heads))
        global_redundancy_matrix = np.zeros((total_heads, total_heads))
        
        flat_time_series = [time_series[metric][layer][head] for layer in range(num_layers) for head in range(num_heads_per_layer)]
        
        for src_idx in range(total_heads):
            for trg_idx in range(total_heads):
                if src_idx != trg_idx:
                    src = flat_time_series[src_idx]
                    trg = flat_time_series[trg_idx]
                    
                    atoms_res, calc_res = calc_PhiID(src, trg, tau, kind, redundancy_measure)
                    
                    synergy_value = np.mean(atoms_res['sts'])
                    redundancy_value = np.mean(atoms_res['rtr'])
                    
                    global_synergy_matrix[src_idx, trg_idx] = synergy_value
                    global_redundancy_matrix[src_idx, trg_idx] = redundancy_value
        
        synergy_matrices[metric] = global_synergy_matrix
        redundancy_matrices[metric] = global_redundancy_matrix

    return synergy_matrices, redundancy_matrices

def plot_synergy_redundancy_PhiID(synergy_matrices, redundancy_matrices, plot_base_path=None):
    if not plot_base_path:
        plot_base_path = constants.PLOTS_SYNERGY_REDUNDANCY_DIR + datetime.now().strftime("%Y%m%d_%H%M%S") + '/'
    for metric, synergy_matrix in synergy_matrices.items():
        redundancy_matrix = redundancy_matrices[metric]
        plot_path = f"{plot_base_path}{metric}.png"

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        cax1 = axs[0].matshow(synergy_matrix, cmap='viridis')
        fig.colorbar(cax1, ax=axs[0])
        axs[0].set_title('Synergy Matrix - ' + metric)
        axs[0].set_xlabel('Attention Head')
        axs[0].set_ylabel('Attention Head')

        cax2 = axs[1].matshow(redundancy_matrix, cmap='viridis')
        fig.colorbar(cax2, ax=axs[1])
        axs[1].set_title('Redundancy Matrix - ' + metric)
        axs[1].set_xlabel('Attention Head')
        axs[1].set_ylabel('Attention Head')

        plt.tight_layout()
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        plt.close()

def compute_all_PhiID(time_series, metrics, tau=1, kind="gaussian", redundancy_measure="MMI"):
    results = {metric: {} for metric in metrics}
    
    for metric in metrics:
        num_layers = len(time_series[metric])
        num_heads_per_layer = len(time_series[metric][0])
        total_heads = num_layers * num_heads_per_layer
        
        # Assuming calc_PhiID can work with example inputs to determine keys
        example_res, _ = calc_PhiID(time_series[metric][0][0], time_series[metric][0][1], tau, kind, redundancy_measure)
        keys = list(example_res.keys())
        
        # Initialize global matrices for each key within this metric
        global_matrices = {key: np.zeros((total_heads, total_heads)) for key in keys}
        
        flat_time_series = [time_series[metric][layer][head] for layer in range(num_layers) for head in range(num_heads_per_layer)]
        
        for src_idx in range(total_heads):
            for trg_idx in range(total_heads):
                if src_idx != trg_idx:
                    src = flat_time_series[src_idx]
                    trg = flat_time_series[trg_idx]
                    
                    atoms_res, calc_res = calc_PhiID(src, trg, tau, kind, redundancy_measure)
                    
                    for key in keys:
                        global_matrices[key][src_idx, trg_idx] = np.mean(atoms_res[key])
        
        results[metric] = global_matrices
    
    return results

def plot_all_PhiID(global_matrices, plot_base_path=None):
    if not plot_base_path:
        plot_base_path = constants.PLOTS_ALL_PHID_DIR + datetime.now().strftime("%Y%m%d_%H%M%S") + '/'
    for metric, matrices in global_matrices.items():
        num_plots = len(matrices)
        fig, axs = plt.subplots(1, num_plots, figsize=(num_plots * 5, 5))  # Adjust as needed for the number of plots
        
        # Flatten axs if necessary (when num_plots is 1, axs is not an array)
        axs = np.array(axs).reshape(-1)
        
        for idx, (key, matrix) in enumerate(matrices.items()):
            cax = axs[idx].matshow(matrix, cmap='viridis')
            fig.colorbar(cax, ax=axs[idx])
            axs[idx].set_title(f'{metric} - {key}')
            axs[idx].set_xlabel('Attention Head')
            axs[idx].set_ylabel('Attention Head')
        
        plot_path = os.path.join(plot_base_path, f'{metric}_PhiID_matrices.png')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
