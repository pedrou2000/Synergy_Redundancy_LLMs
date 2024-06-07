from phyid.calculate import calc_PhiID 
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import os
import matplotlib.pyplot as plt
import constants
from datetime import datetime
import pickle
import seaborn as sns
import torch

# Function to check and modify the vector if variance is zero
def check_and_modify_variance(vector, index, epsilon=1e-2):
    variance = torch.var(vector)
    if variance == 0:
        random_idx = np.random.randint(len(vector))
        vector[random_idx] -= epsilon
        print(f"Variance for head {index} was zero. Modified element at index {random_idx}.")
    return vector

def compute_PhiID(time_series, tau=1, kind="gaussian", redundancy_measure="MMI", save=False, base_save_path=None):
    metrics = list(time_series.keys())
    all_matrices = {metric: {} for metric in metrics}
    
    for metric in metrics:
        print(f"Calculating PhiID for metric: {metric}")
        num_layers = len(time_series[metric])
        num_heads_per_layer = len(time_series[metric][0])
        total_heads = num_layers * num_heads_per_layer
        
        example_res, _ = calc_PhiID(time_series[metric][0][0], time_series[metric][0][1], tau, kind, redundancy_measure)
        keys = list(example_res.keys())
        
        global_matrices = {key: np.zeros((total_heads, total_heads)) for key in keys}
        
        flat_time_series = [time_series[metric][layer][head] for layer in range(num_layers) for head in range(num_heads_per_layer)] # Flatten the time series
        
        for src_idx in range(total_heads):
            if src_idx % 50 == 0:
                print(f"Calculating PhiID for head {src_idx}...")
            for trg_idx in range(total_heads):
                if src_idx != trg_idx:
                    src = flat_time_series[src_idx] # Extract the source and target time series
                    trg = flat_time_series[trg_idx]
                    check_and_modify_variance(src, src_idx)
                    check_and_modify_variance(trg, trg_idx)

                    atoms_res, calc_res = calc_PhiID(src, trg, tau, kind, redundancy_measure)
                    
                    for key in keys:
                        global_matrices[key][src_idx, trg_idx] = np.mean(atoms_res[key])
        
        all_matrices[metric] = global_matrices
    
    synergy_matrices = {metric: all_matrices[metric]['sts'] for metric in metrics}
    redundancy_matrices = {metric: all_matrices[metric]['rtr'] for metric in metrics}

    if save:
         save_matrices(all_matrices, synergy_matrices, redundancy_matrices, base_save_path=base_save_path)
    
    return all_matrices, synergy_matrices, redundancy_matrices

def process_pair(src_idx, trg_idx, src, trg, tau, kind, redundancy_measure):
    check_and_modify_variance(src, src_idx)
    check_and_modify_variance(trg, trg_idx)
    atoms_res, _ = calc_PhiID(src, trg, tau, kind, redundancy_measure)
    return (src_idx, trg_idx, atoms_res)

def compute_PhiID_parallel(time_series, tau=1, kind="gaussian", redundancy_measure="MMI", save=False, base_save_path=None):
    metrics = list(time_series.keys())
    all_matrices = {metric: {} for metric in metrics}

    for metric in metrics:
        print(f"Calculating PhiID for metric: {metric}")
        num_layers = len(time_series[metric])
        num_heads_per_layer = len(time_series[metric][0])
        total_heads = num_layers * num_heads_per_layer
        
        example_res, _ = calc_PhiID(time_series[metric][0][0], time_series[metric][0][1], tau, kind, redundancy_measure)
        keys = list(example_res.keys())
        
        global_matrices = {key: np.zeros((total_heads, total_heads)) for key in keys}
        
        flat_time_series = [time_series[metric][layer][head] for layer in range(num_layers) for head in range(num_heads_per_layer)] # Flatten the time series
        
        tasks = []
        num_workers = 4  # Set the number of workers you want to use
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for src_idx in range(total_heads):
                if src_idx % 50 == 0:
                    print(f"Submitting tasks for head {src_idx}...")
                for trg_idx in range(total_heads):
                    if src_idx != trg_idx:
                        src = flat_time_series[src_idx]
                        trg = flat_time_series[trg_idx]
                        tasks.append(executor.submit(process_pair, src_idx, trg_idx, src, trg, tau, kind, redundancy_measure))
            
            for future in as_completed(tasks):
                src_idx, trg_idx, atoms_res = future.result()
                for key in keys:
                    global_matrices[key][src_idx, trg_idx] = np.mean(atoms_res[key])
        
        all_matrices[metric] = global_matrices
    
    synergy_matrices = {metric: all_matrices[metric]['sts'] for metric in metrics}
    redundancy_matrices = {metric: all_matrices[metric]['rtr'] for metric in metrics}

    if save:
        save_matrices(all_matrices, synergy_matrices, redundancy_matrices, base_save_path=base_save_path)
    
    return all_matrices, synergy_matrices, redundancy_matrices

def save_matrices(all_matrices, synergy_matrices, redundancy_matrices, base_save_path=None):
    if not base_save_path:
        # Assuming 'constants.MATRICES_DIR' is defined and is a valid path
        base_save_path = constants.MATRICES_DIR + 'matrices.pkl'
    
    # Extract directory from base_save_path
    dir_path = os.path.dirname(base_save_path)
    
    # Create the directory if it does not exist
    os.makedirs(dir_path, exist_ok=True)
    
    # Now save the file
    with open(base_save_path, 'wb') as file:
        pickle.dump((all_matrices, synergy_matrices, redundancy_matrices), file)

def load_matrices(matrices_number=0, base_save_path=None):
    # Sort the time_series files by name and load the time_series_number-th file
    time_series_files = sorted(os.listdir(constants.MATRICES_DIR))
    if not base_save_path:
        base_save_path = constants.MATRICES_DIR + time_series_files[matrices_number]
    with open(base_save_path, 'rb') as file:
        all_matrices, synergy_matrices, redundancy_matrices = pickle.load(file)

    return all_matrices, synergy_matrices, redundancy_matrices

def average_synergy_redundancies_matrices_cognitive_tasks(all_matrices, synergy_matrices, redundancy_matrices):
    # Initialize the result dictionary dynamically
    average_matrices = {}
    num_matrices = len(all_matrices.keys())

    # Extract the structure dynamically
    for key1 in all_matrices.keys():
        for key2 in all_matrices[key1].keys():
            if key2 not in average_matrices:
                average_matrices[key2] = {}
            for key3 in all_matrices[key1][key2].keys():
                if key3 not in average_matrices[key2]:
                    average_matrices[key2][key3] = np.zeros(all_matrices[key1][key2][key3].shape)

    # Iterate over each key in the first dimension of all_matrices
    for key1 in all_matrices.keys():
        for key2 in all_matrices[key1].keys():
            for key3 in all_matrices[key1][key2].keys():
                # Sum the matrices
                average_matrices[key2][key3] += all_matrices[key1][key2][key3]
    
    # Compute the average by dividing by the number of matrices
    for key2 in average_matrices.keys():
        for key3 in average_matrices[key2].keys():
            average_matrices[key2][key3] /= num_matrices

    average_synergy_matrices = {metric: average_matrices[metric]['sts'] for metric in average_matrices.keys()}
    average_redundancy_matrices = {metric: average_matrices[metric]['rtr'] for metric in average_matrices.keys()}

    return average_matrices, average_synergy_matrices, average_redundancy_matrices


def layer_wise_matrix(matrix, block_size=constants.NUM_HEADS_PER_LAYER):
    """
    Averages each block_size x block_size square in the matrix.
    """
    new_size = matrix.shape[0] // block_size
    layer_wise_matrix = np.zeros((new_size, new_size))
    
    for i in range(new_size):
        for j in range(new_size):
            block = matrix[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            layer_wise_matrix[i, j] = np.mean(block)
    
    return layer_wise_matrix

def plot_synergy_redundancy_PhiID(synergy_matrices, redundancy_matrices, base_plot_path=None, save=True):
    # plt.rcParams.update({'font.size': 15})

    if not base_plot_path:
        base_plot_path = constants.PLOTS_SYNERGY_REDUNDANCY_DIR 
    
    for metric, synergy_matrix in synergy_matrices.items():
        redundancy_matrix = redundancy_matrices[metric]
        plot_path = f"{base_plot_path}{metric}/1-Synergy-Redundancy_Matrices.png"
        layer_wise_synergy = layer_wise_matrix(synergy_matrix)
        layer_wise_redundancy = layer_wise_matrix(redundancy_matrix)
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        cax1 = axs[0, 0].matshow(synergy_matrix, cmap='viridis')
        fig.colorbar(cax1, ax=axs[0, 0])
        axs[0, 0].set_title('Synergy Matrix')
        axs[0, 0].set_xlabel('Attention Head')
        axs[0, 0].set_ylabel('Attention Head')
        axs[0, 0].xaxis.set_ticks_position('bottom')
        axs[0, 0].xaxis.set_label_position('bottom')

        cax2 = axs[0, 1].matshow(redundancy_matrix, cmap='viridis')
        fig.colorbar(cax2, ax=axs[0, 1])
        axs[0, 1].set_title('Redundancy Matrix')
        axs[0, 1].set_xlabel('Attention Head')
        axs[0, 1].set_ylabel('Attention Head')
        axs[0, 1].xaxis.set_ticks_position('bottom')
        axs[0, 1].xaxis.set_label_position('bottom')
        
        cax3 = axs[1, 0].matshow(layer_wise_synergy, cmap='viridis')
        fig.colorbar(cax3, ax=axs[1, 0])
        axs[1, 0].set_title('Layer-wise Synergy Matrix')
        axs[1, 0].set_xlabel('Layer')
        axs[1, 0].set_ylabel('Layer')
        axs[1, 0].xaxis.set_ticks_position('bottom')
        axs[1, 0].xaxis.set_label_position('bottom')

        cax4 = axs[1, 1].matshow(layer_wise_redundancy, cmap='viridis')
        fig.colorbar(cax4, ax=axs[1, 1])
        axs[1, 1].set_title('Layer-wise Redundancy Matrix')
        axs[1, 1].set_xlabel('Layer')
        axs[1, 1].set_ylabel('Layer')
        axs[1, 1].xaxis.set_ticks_position('bottom')
        axs[1, 1].xaxis.set_label_position('bottom')

        if save:
            plt.tight_layout()
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path)
        else:
            plt.show()
        plt.close()


def plot_all_PhiID(global_matrices, base_plot_path=None, save=True):
    if not base_plot_path:
        base_plot_path = constants.PLOTS_SYNERGY_REDUNDANCY_DIR 
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
            plot_path = os.path.join(base_plot_path, f'{metric}/2-All_PhiID_Matrices.png')
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path)
        else:
            plt.show()
        plt.close()



####################### Other Visualizations #######################
def calculate_average_synergy_redundancies_per_head(synergy_matrices, redundancy_matrices, within_layer=False, num_heads_per_layer=constants.NUM_HEADS_PER_LAYER):
    averages = {}
    for metric in synergy_matrices.keys():
        num_heads = synergy_matrices[metric].shape[0]  # total number of heads
        synergy_avg_per_head = np.zeros(num_heads)
        redundancy_avg_per_head = np.zeros(num_heads)

        if within_layer:
            # Calculate averages per head only within the same layer
            for head_index in range(num_heads):
                layer_start = (head_index // num_heads_per_layer) * num_heads_per_layer
                layer_end = layer_start + num_heads_per_layer

                # Calculate average for the current head across its layer
                synergy_avg_per_head[head_index] = np.mean(synergy_matrices[metric][head_index, layer_start:layer_end])
                redundancy_avg_per_head[head_index] = np.mean(redundancy_matrices[metric][head_index, layer_start:layer_end])
        else:
            # Calculate averages globally across all heads
            synergy_avg_per_head = np.mean(synergy_matrices[metric], axis=1)
            redundancy_avg_per_head = np.mean(redundancy_matrices[metric], axis=1)

        averages[metric] = {'synergy': synergy_avg_per_head, 'redundancy': redundancy_avg_per_head}
    return averages

def plot_averages_per_head(averages, base_plot_path=None, save=False, use_heatmap=False, num_heads_per_layer=constants.NUM_HEADS_PER_LAYER):
    # plt.rcParams.update({'font.size': 12})  # Adjust the 14 to larger sizes as needed

    if not base_plot_path:
        base_plot_path = constants.PLOTS_SYNERGY_REDUNDANCY_DIR 

    for metric, avg_data in averages.items():
        synergy_avgs = avg_data['synergy']
        redundancy_avgs = avg_data['redundancy']

        if use_heatmap:
            num_layers = len(synergy_avgs) // num_heads_per_layer
            synergy_matrix = synergy_avgs.reshape((num_layers, num_heads_per_layer))
            redundancy_matrix = redundancy_avgs.reshape((num_layers, num_heads_per_layer))

            # Plot and save synergy heatmap
            fig, ax = plt.subplots(figsize=(12, 5))
            heatmap = sns.heatmap(synergy_matrix.T, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=0.5, 
                                  linecolor='gray', cbar_kws={"shrink": 0.8})
            cbar = heatmap.collections[0].colorbar
            cbar.set_label('Synergy', size=16)  # Adjust the size as needed

            ax.set_title(f'Average Synergy per Head')
            ax.set_xlabel("Layer")
            ax.set_ylabel("Head")

            ax.set_xticks(np.arange(num_layers) + 0.5)  # Change: Added set_xticks
            ax.set_xticklabels([f"{i+1}" for i in range(num_layers)])
            ax.set_yticks(np.arange(num_heads_per_layer) + 0.5)  # Change: Added set_yticks
            ax.set_yticklabels([f"{i+1}" for i in range(num_heads_per_layer)], rotation=0)
            plt.tight_layout()

            synergy_plot_path = f"{base_plot_path}{metric}/3-Synergy_per_Head.png"
            if save:
                os.makedirs(os.path.dirname(synergy_plot_path), exist_ok=True)
                plt.savefig(synergy_plot_path, bbox_inches='tight')
            else:
                plt.show()
            plt.close()

            # Plot and save redundancy heatmap
            fig, ax = plt.subplots(figsize=(12, 5))
            heatmap = sns.heatmap(redundancy_matrix.T, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=0.5, linecolor='gray', 
                                cbar_kws={"shrink": 0.8})
            cbar = heatmap.collections[0].colorbar
            cbar.set_label('Redundancy', size=16)  # Adjust the size as needed

            ax.set_title(f'Average Redundancy per Head')
            ax.set_xlabel("Layer")
            ax.set_ylabel("Head")
            ax.set_xticks(np.arange(num_layers) + 0.5)  # Change: Added set_xticks
            ax.set_xticklabels([f"{i+1}" for i in range(num_layers)], rotation=0)
            ax.set_yticks(np.arange(num_heads_per_layer) + 0.5)  # Change: Added set_yticks
            ax.set_yticklabels([f"{i+1}" for i in range(num_heads_per_layer)], rotation=0)
            plt.tight_layout()

            redundancy_plot_path = f"{base_plot_path}{metric}/4-Redundancy_per_Head.png"
            if save:
                os.makedirs(os.path.dirname(redundancy_plot_path), exist_ok=True)
                plt.savefig(redundancy_plot_path, bbox_inches='tight')
            else:
                plt.show()
            plt.close()

        else:
            # Continue using the existing line plot code if heatmap is not selected
            fig, ax = plt.subplots(figsize=(16, 5))
            heads = np.arange(len(synergy_avgs))  # Assuming the number of heads is the same for synergy and redundancy
            ax.plot(heads, synergy_avgs, marker='o', linestyle='-', color='b', label='Synergy')
            ax.plot(heads, redundancy_avgs, marker='s', linestyle='-', color='r', label='Redundancy')
            
            ax.set_xlabel('Attention Head')
            ax.set_ylabel('Average Value')
            ax.set_title(f'Average Synergy and Redundancy per Head for {metric}')
            ax.legend()

            line_plot_path = f"{base_plot_path}{metric}/averages_line.png"
            if save:
                os.makedirs(os.path.dirname(line_plot_path), exist_ok=True)
                plt.savefig(line_plot_path, bbox_inches='tight')
            else :
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

def plot_synergy_redundancy_rank_gradient(rank_gradients, base_plot_path=None, save=True):
    if not base_plot_path:
        base_plot_path = constants.PLOTS_SYNERGY_REDUNDANCY_RANK_GRADIENT 
    
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
        
        plot_path = f"{base_plot_path}{metric}/rank_gradient.png"
        
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

def plot_gradient_percentile(gradient_percentiles, base_plot_path=None, save=False):
    if not base_plot_path:
        base_plot_path = constants.PLOTS_GRADIENT_PERCENTILE 
    
    for metric, percentiles in gradient_percentiles.items():
        heads = np.arange(len(percentiles))  # Assuming the number of heads is consistent
        
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(heads, percentiles, marker='o', linestyle='-', color='darkblue')
        
        ax.set_xlabel('Attention Head')
        ax.set_ylabel('Gradient Percentile')
        ax.set_title(f'Synergy-Redundancy Gradient Percentile for {metric}')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        plot_path = f"{base_plot_path}{metric}/gradient_percentile.png"
        
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

def plot_gradient_rank(gradient_ranks, base_plot_path=None, save=False, use_heatmap=False, num_heads_per_layer=constants.NUM_HEADS_PER_LAYER):
    if not base_plot_path:
        # Set a default path if not provided
        base_plot_path = constants.PLOTS_SYNERGY_REDUNDANCY_DIR 

    for metric, head_ranks in gradient_ranks.items():
        # Convert head_ranks dictionary back to a list of ranks for plotting
        # Since head numbers are 1-indexed, we adjust accordingly
        heads = list(head_ranks.keys())
        ranks = [head_ranks[head] for head in heads]

        if use_heatmap:
            # Assume ranks can be reshaped into a num_layers x num_heads_per_layer matrix
            num_layers = len(ranks) // num_heads_per_layer
            ranks_matrix = np.array(ranks).reshape((num_layers, num_heads_per_layer))

            fig, ax = plt.subplots(figsize=(12, 5))
            sns.heatmap(ranks_matrix.T, annot=True, fmt="d", cmap="viridis", cbar=True, linewidths=0.5,
                        linecolor='gray', cbar_kws={"shrink": 0.8, "label": 'Synergy - Redundancy Rank'})
            ax.set_xticks(np.arange(num_layers) + 0.5)  # Change: Added set_yticks
            ax.set_xticklabels([f"{i+1}" for i in range(num_layers)], rotation=45, ha="right")
            ax.set_yticks(np.arange(num_heads_per_layer) + 0.5)  # Change: Added set_yticks
            ax.set_yticklabels([f"{i+1}" for i in range(num_heads_per_layer)], rotation=0)
            ax.set_title(f'Synergy - Redundancy Rank Heatmap for {metric}')
            ax.set_xlabel("Layer")
            ax.set_ylabel("Head")
            plt.tight_layout()
        else:
            fig, ax = plt.subplots(figsize=(16, 6))
            ax.plot(heads, ranks, marker='o', linestyle='-', color='darkblue')

            ax.set_xlabel('Attention Head')
            ax.set_ylabel('Synergy - Redundancy Rank')
            ax.set_title(f'Synergy-Redundancy Rank for {metric}')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        plot_path = f"{base_plot_path}{metric}/5-Synergy-Redundancy_Rank_per_Head.png"

        if save:
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path)
        else:
            plt.show()
        plt.close()

def plot_averages_per_layer(averages, base_plot_path=None, save=False, num_heads_per_layer=constants.NUM_HEADS_PER_LAYER):
    if not base_plot_path:
        base_plot_path = constants.PLOTS_SYNERGY_REDUNDANCY_DIR 

    for metric, avg_data in averages.items():
        synergy_avgs = avg_data['synergy']
        redundancy_avgs = avg_data['redundancy']

        # Compute the average across all heads per layer
        num_layers = len(synergy_avgs) // num_heads_per_layer
        synergy_per_layer = np.mean(synergy_avgs.reshape(num_layers, num_heads_per_layer), axis=1)
        redundancy_per_layer = np.mean(redundancy_avgs.reshape(num_layers, num_heads_per_layer), axis=1)

        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(range(1, num_layers+1), synergy_per_layer, marker='o', linestyle='-', color='b', label='Synergy')
        ax.plot(range(1, num_layers+1), redundancy_per_layer, marker='s', linestyle='-', color='r', label='Redundancy')

        ax.set_xlabel('Layer')
        ax.set_ylabel('Average Value')
        ax.set_title(f'Average Synergy and Redundancy per Layer for {metric}')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Set x-axis to label each layer explicitly
        ax.set_xticks(range(1, num_layers + 1))  # Set tick positions
        ax.set_xticklabels([str(i) for i in range(1, num_layers + 1)])  # Label each tick with the layer number

        plot_path = f"{base_plot_path}{metric}/averages_per_layer.png"

        if save:
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path)
        else:
            plt.show()
        plt.close()

def plot_average_ranks_per_layer(gradient_ranks, base_plot_path=None, save=False, num_heads_per_layer=constants.NUM_HEADS_PER_LAYER):
    # plt.rcParams.update({'font.size': 15})  # Adjust the font size as needed
    if not base_plot_path:
        base_plot_path = constants.PLOTS_SYNERGY_REDUNDANCY_DIR 
    
    ranks_per_layer_mean, ranks_per_layer_std = {}, {}

    for metric, head_ranks in gradient_ranks.items():
        # Convert head_ranks dictionary back to a list of ranks for plotting
        heads = list(head_ranks.keys())
        ranks = [head_ranks[head] for head in heads]

        # Compute the average and standard deviation across all heads per layer
        num_layers = len(ranks) // num_heads_per_layer
        ranks_array = np.array(ranks).reshape(num_layers, num_heads_per_layer)
        ranks_per_layer_mean[metric] = np.mean(ranks_array, axis=1)
        ranks_per_layer_std[metric] = np.std(ranks_array, axis=1)

        fig, ax = plt.subplots(figsize=(16, 6))
        # Include error bars
        ax.errorbar(range(1, num_layers + 1), ranks_per_layer_mean[metric], yerr=ranks_per_layer_std[metric], marker='o', linestyle='-', color='darkblue', ecolor='gray', capsize=5)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Average Rank')
        ax.set_title(f'Average Synergy Minus Redundancy Rank per Layer with Error Bars')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Set x-axis to label each layer explicitly
        ax.set_xticks(range(1, num_layers + 1))  # Set tick positions
        ax.set_xticklabels([str(i) for i in range(1, num_layers + 1)])  # Label each tick with the layer number

        plot_path = f"{base_plot_path}{metric}/6-Average_Synergy-Redundancy_Ranks_per_Layer.png"

        if save:
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path)
        else:
            plt.show()
        plt.close()
    return ranks_per_layer_mean, ranks_per_layer_std






