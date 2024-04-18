from phyid.calculate import calc_PhiID 
import numpy as np
import os
import matplotlib.pyplot as plt
import constants
from datetime import datetime
import pickle

def compute_PhiID(time_series, metrics, tau=1, kind="gaussian", redundancy_measure="MMI", save=False, base_save_path=None):
    results = {metric: {} for metric in metrics}
    
    for metric in metrics:
        num_layers = len(time_series[metric])
        num_heads_per_layer = len(time_series[metric][0])
        total_heads = num_layers * num_heads_per_layer
        
        example_res, _ = calc_PhiID(time_series[metric][0][0], time_series[metric][0][1], tau, kind, redundancy_measure)
        keys = list(example_res.keys())
        
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
    
    synergy_matrices = {metric: results[metric]['sts'] for metric in metrics}
    redundancy_matrices = {metric: results[metric]['rtr'] for metric in metrics}

    if save:
         save_matrices(results, synergy_matrices, redundancy_matrices, base_save_path=base_save_path)
    
    return results, synergy_matrices, redundancy_matrices

def save_matrices(results, synergy_matrices, redundancy_matrices, base_save_path=None):
    if not base_save_path:
        # Assuming 'constants.MATRICES_DIR' is defined and is a valid path
        base_save_path = constants.MATRICES_DIR + datetime.now().strftime("%Y%m%d_%H%M%S") + '.pkl'
    
    # Extract directory from base_save_path
    dir_path = os.path.dirname(base_save_path)
    
    # Create the directory if it does not exist
    os.makedirs(dir_path, exist_ok=True)
    
    # Now save the file
    with open(base_save_path, 'wb') as file:
        pickle.dump((results, synergy_matrices, redundancy_matrices), file)

def load_matrices(matrices_number, base_plot_path=None):
    # Sort the time_series files by name and load the time_series_number-th file
    time_series_files = sorted(os.listdir(constants.MATRICES_DIR))
    base_plot_path = constants.MATRICES_DIR + time_series_files[matrices_number]
    with open(base_plot_path, 'rb') as file:
        results, synergy_matrices, redundancy_matrices = pickle.load(file)

    return results, synergy_matrices, redundancy_matrices

def plot_synergy_redundancy_PhiID(synergy_matrices, redundancy_matrices, plot_base_path=None, save=True):
    if not plot_base_path:
        plot_base_path = constants.PLOTS_SYNERGY_REDUNDANCY_DIR + datetime.now().strftime("%Y%m%d_%H%M%S") + '/'
    for metric, synergy_matrix in synergy_matrices.items():
        redundancy_matrix = redundancy_matrices[metric]
        plot_path = f"{plot_base_path}{metric}.png"

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        cax1 = axs[0].matshow(synergy_matrix, cmap='viridis')
        fig.colorbar(cax1, ax=axs[0])
        axs[0].set_title('Synergy Matrix')
        axs[0].set_xlabel('Attention Head')
        axs[0].set_ylabel('Attention Head')

        cax2 = axs[1].matshow(redundancy_matrix, cmap='viridis')
        fig.colorbar(cax2, ax=axs[1])
        axs[1].set_title('Redundancy Matrix')
        axs[1].set_xlabel('Attention Head')
        axs[1].set_ylabel('Attention Head')

        if save:
            plt.tight_layout()
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path)
        else:
            plt.show()
        plt.close()

def plot_all_PhiID(global_matrices, plot_base_path=None, save=True):
    if not plot_base_path:
        plot_base_path = constants.PLOTS_ALL_PHID_DIR + datetime.now().strftime("%Y%m%d_%H%M%S") + '/'
    for metric, matrices in global_matrices.items():
        num_plots = len(matrices)
        fig, axs = plt.subplots(4, 4, figsize=(20, 20))  # Adjust the figsize as needed
        
        # Flatten axs if necessary (when num_plots is 1, axs is not an array)
        axs = np.array(axs).reshape(-1)
        
        for idx, (key, matrix) in enumerate(matrices.items()):
            cax = axs[idx].matshow(matrix, cmap='viridis')
            fig.colorbar(cax, ax=axs[idx])
            axs[idx].set_title(f'{key}')
            axs[idx].set_xlabel('Attention Head')
            axs[idx].set_ylabel('Attention Head')
        plt.tight_layout()
        
        if save:
            plot_path = os.path.join(plot_base_path, f'{metric}_PhiID_matrices.png')
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path)
        else:
            plt.show()
        plt.close()



####################### Other Visualizations #######################
def calculate_averages_per_head(synergy_matrices, redundancy_matrices):
    averages = {}
    for metric in synergy_matrices.keys():
        synergy_avg_per_head = np.mean(synergy_matrices[metric], axis=1)
        redundancy_avg_per_head = np.mean(redundancy_matrices[metric], axis=1)
        
        averages[metric] = {'synergy': synergy_avg_per_head, 'redundancy': redundancy_avg_per_head}
    return averages

def plot_averages_per_head(averages, plot_base_path=None, save=True):
    if not plot_base_path:
        plot_base_path = constants.PLOTS_SYNERGY_REDUNDANCY_PER_HEAD + datetime.now().strftime("%Y%m%d_%H%M%S") + '/'
    
    for metric, avg_data in averages.items():
        synergy_avgs = avg_data['synergy']
        redundancy_avgs = avg_data['redundancy']
        heads = np.arange(len(synergy_avgs))  # Assuming the number of heads is the same for synergy and redundancy
        
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.plot(heads, synergy_avgs, marker='o', linestyle='-', color='b', label='Synergy')
        ax.plot(heads, redundancy_avgs, marker='s', linestyle='-', color='r', label='Redundancy')
        
        ax.set_xlabel('Attention Head')
        ax.set_ylabel('Average Value')
        ax.set_title(f'Average Synergy and Redundancy per Head for {metric}')
        ax.legend()

        plot_path = f"{plot_base_path}{metric}_averages.png"
        
        if save:
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(plot_path)
        else:
            plt.show()
        plt.close()

def compute_synergy_redundancy_rank_gradient(averages):
    rank_gradients = {}
    for metric, avg_data in averages.items():
        # Calculate ranks (1 is highest value, hence most redundant/synergistic)
        synergy_ranks = np.argsort(-avg_data['synergy']) + 1
        redundancy_ranks = np.argsort(-avg_data['redundancy']) + 1
        
        # Compute the gradient: Redundancy rank - Synergy rank
        rank_gradient = redundancy_ranks - synergy_ranks
        
        rank_gradients[metric] = rank_gradient
    return rank_gradients

def plot_synergy_redundancy_rank_gradient(rank_gradients, plot_base_path=None, save=True):
    if not plot_base_path:
        plot_base_path = constants.PLOTS_SYNERGY_REDUNDANCY_RANK_GRADIENT + datetime.now().strftime("%Y%m%d_%H%M%S") + '/'
    
    for metric, gradient in rank_gradients.items():
        heads = np.arange(len(gradient))  # Assuming the number of heads is consistent
        
        fig, ax = plt.subplots(figsize=(16, 6))
        bars = ax.bar(heads, gradient, color=np.where(gradient < 0, 'skyblue', 'salmon'))
        
        ax.set_xlabel('Attention Head')
        ax.set_ylabel('Synergy-Redundancy Rank Gradient')
        ax.set_title(f'Synergy-Redundancy Rank Gradient for {metric}')
        ax.axhline(0, color='grey', linewidth=0.8)
        
        # Highlighting the zero line for reference
        for bar in bars:
            if bar.get_height() < 0:
                bar.set_edgecolor('blue')
            else:
                bar.set_edgecolor('red')
        
        plot_path = f"{plot_base_path}{metric}_rank_gradient.png"
        
        if save:
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(plot_path)
        else:
            plt.show()
        plt.close()

def compute_gradient_percentile(averages):
    gradient_percentiles = {}
    for metric, avg_data in averages.items():
        # Calculate synergy minus redundancy for each head
        gradient_diff = avg_data['synergy'] - avg_data['redundancy']
        
        # Calculate ranks based on the gradient difference
        ranks = gradient_diff.argsort().argsort() + 1  # +1 to start ranks at 1 instead of 0
        
        # Convert ranks to percentiles
        percentiles = 100.0 * (ranks - 1) / (len(gradient_diff) - 1)
        
        gradient_percentiles[metric] = percentiles
    return gradient_percentiles

def plot_gradient_percentile(gradient_percentiles, plot_base_path=None, save=False):
    if not plot_base_path:
        plot_base_path = constants.PLOTS_GRADIENT_PERCENTILE + datetime.now().strftime("%Y%m%d_%H%M%S") + '/'
    
    for metric, percentiles in gradient_percentiles.items():
        heads = np.arange(len(percentiles))  # Assuming the number of heads is consistent
        
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(heads, percentiles, marker='o', linestyle='-', color='darkblue')
        
        ax.set_xlabel('Attention Head')
        ax.set_ylabel('Gradient Percentile')
        ax.set_title(f'Synergy-Redundancy Gradient Percentile for {metric}')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        plot_path = f"{plot_base_path}{metric}_gradient_percentile.png"
        
        if save:
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(plot_path)
        else:
            plt.show()
        plt.close()
    
def compute_gradient_rank(averages, method='synergy-redundancy'):
    gradient_ranks = {}
    for metric, avg_data in averages.items():
        # Calculate synergy minus redundancy for each head
        if method == 'synergy-redundancy':
            gradient_diff = avg_data['synergy'] - avg_data['redundancy']
        elif method == 'synergy':
            gradient_diff = avg_data['synergy']
        elif method == 'redundancy':
            gradient_diff = avg_data['redundancy']
        
        # Sort gradient_diff and get indices of sorted array
        sorted_indices = np.argsort(gradient_diff)
        
        # Create an empty array of the same length as gradient_diff to store ranks
        ranks = np.empty_like(sorted_indices)
        
        # Assign ranks; since sorted_indices are 0-based, we add 1 to make them start from 1
        ranks[sorted_indices] = np.arange(1, len(gradient_diff) + 1)
        
        # Create a dictionary where the key is the head number (1-indexed) and value is the rank
        head_ranks = {head_number: rank for head_number, rank in enumerate(ranks, start=1)}
        
        gradient_ranks[metric] = head_ranks
    return gradient_ranks

def plot_gradient_rank(gradient_ranks, plot_base_path=None, save=False):
    if not plot_base_path:
        # Assuming constants.PLOTS_GRADIENT_RANK exists and has a similar purpose as constants.PLOTS_GRADIENT_PERCENTILE
        plot_base_path = "path/to/your/gradient/rank/plots/" + datetime.now().strftime("%Y%m%d_%H%M%S") + '/'
    
    for metric, head_ranks in gradient_ranks.items():
        # Convert head_ranks dictionary back to a list of ranks for plotting
        # Since head numbers are 1-indexed, we adjust accordingly
        heads = list(head_ranks.keys())
        ranks = [head_ranks[head] for head in heads]

        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(heads, ranks, marker='o', linestyle='-', color='darkblue')
        
        ax.set_xlabel('Attention Head')
        ax.set_ylabel('Gradient Rank')
        ax.set_title(f'Synergy-Redundancy Gradient Rank for {metric}')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        plot_path = f"{plot_base_path}{metric}_gradient_rank.png"
        
        if save:
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(plot_path)
        else:
            plt.show()
        plt.close()


