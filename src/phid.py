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
import pandas as pd

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
    # Obtain and Print the maximum number of threads
    max_threads = os.cpu_count()
    print(f"Maximum number of threads: {max_threads}")

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
        with ProcessPoolExecutor(max_workers=max_threads) as executor:
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
    if not base_save_path:
        time_series_files = sorted(os.listdir(constants.MATRICES_DIR))
        base_save_path = constants.MATRICES_DIR + time_series_files[matrices_number]
    with open(base_save_path, 'rb') as file:
        all_matrices, synergy_matrices, redundancy_matrices = pickle.load(file)

    return all_matrices, synergy_matrices, redundancy_matrices

def average_synergy_redundancies_matrices_cognitive_tasks(all_matrices, synergy_matrices, redundancy_matrices):
    # Initialize the result dictionary dynamically
    # all_matrices is a
    average_matrices = {}
    num_matrices = len(all_matrices.keys()) * len(all_matrices[list(all_matrices.keys())[0]].keys())

    # Extract the structure dynamically
    for task_category in all_matrices.keys():
        for n_prompt in all_matrices[task_category].keys():
            for metric in all_matrices[task_category][n_prompt].keys():
                if metric not in average_matrices:
                    average_matrices[metric] = {}
                for phid_atom in all_matrices[task_category][n_prompt][metric].keys():
                    if phid_atom not in average_matrices[metric]:
                        average_matrices[metric][phid_atom] = np.zeros(all_matrices[task_category][n_prompt][metric][phid_atom].shape)

    # Iterate over each key in the first dimension of all_matrices
    for task_category in all_matrices.keys():
        for n_prompt in all_matrices[task_category].keys():
            for metric in all_matrices[task_category][n_prompt].keys():
                for phid_atom in all_matrices[task_category][n_prompt][metric].keys():
                    # Sum the matrices
                    average_matrices[metric][phid_atom] += all_matrices[task_category][n_prompt][metric][phid_atom]
    
    # Compute the average by dividing by the number of matrices
    for metric in average_matrices.keys():
        for phid_atom in average_matrices[metric].keys():
            average_matrices[metric][phid_atom] /= num_matrices

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
        axs[0, 0].set_title('Synergy')
        axs[0, 0].set_xlabel('Attention Head')
        axs[0, 0].set_ylabel('Attention Head')
        axs[0, 0].xaxis.set_ticks_position('bottom')
        axs[0, 0].xaxis.set_label_position('bottom')

        cax2 = axs[0, 1].matshow(redundancy_matrix, cmap='viridis')
        fig.colorbar(cax2, ax=axs[0, 1])
        axs[0, 1].set_title('Redundancy')
        axs[0, 1].set_xlabel('Attention Head')
        axs[0, 1].set_ylabel('Attention Head')
        axs[0, 1].xaxis.set_ticks_position('bottom')
        axs[0, 1].xaxis.set_label_position('bottom')
        
        cax3 = axs[1, 0].matshow(layer_wise_synergy, cmap='viridis')
        fig.colorbar(cax3, ax=axs[1, 0])
        axs[1, 0].set_title('Layer-wise Synergy')
        axs[1, 0].set_xlabel('Layer')
        axs[1, 0].set_ylabel('Layer')
        axs[1, 0].xaxis.set_ticks_position('bottom')
        axs[1, 0].xaxis.set_label_position('bottom')

        cax4 = axs[1, 1].matshow(layer_wise_redundancy, cmap='viridis')
        fig.colorbar(cax4, ax=axs[1, 1])
        axs[1, 1].set_title('Layer-wise Redundancy')
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

def plot_matrix(matrix, title, xlabel, ylabel, save_path=None, save=True):
    """Utility function to plot a single matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(matrix, cmap='viridis')
    fig.colorbar(cax, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')
    if save and save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_synergy_redundancy_combined_vertical(synergy_matrix, redundancy_matrix, layer_wise_synergy, layer_wise_redundancy, save_path_headwise=None, save_path_layerwise=None, save=True):
    """Utility function to create two vertical plots: one for head-wise matrices and one for layer-wise matrices."""

    # Head-wise plot (Synergy and Redundancy stacked vertically, sharing color bar)
    fig, axs = plt.subplots(2, 1, figsize=(4, 6), constrained_layout=True)
    combined_min = min(synergy_matrix.min(), redundancy_matrix.min())
    combined_max = max(synergy_matrix.max(), redundancy_matrix.max())

    cax1 = axs[0].matshow(synergy_matrix, cmap='viridis', vmin=combined_min, vmax=combined_max)
    axs[0].set_title('Synergy')
    axs[0].set_xlabel('Attention Head')
    axs[0].set_ylabel('Attention Head')

    cax2 = axs[1].matshow(redundancy_matrix, cmap='viridis', vmin=combined_min, vmax=combined_max)
    axs[1].set_title('Redundancy')
    axs[1].set_xlabel('Attention Head')
    axs[1].set_ylabel('Attention Head')

    fig.colorbar(cax2, ax=axs, orientation='vertical', fraction=0.1, pad=0.02)

    if save and save_path_headwise:
        os.makedirs(os.path.dirname(save_path_headwise), exist_ok=True)
        plt.savefig(save_path_headwise)
    else:
        plt.show()
    plt.close()

    # Layer-wise plot (Synergy and Redundancy stacked vertically, sharing color bar)
    fig, axs = plt.subplots(2, 1, figsize=(4, 6), constrained_layout=True)
    combined_min = min(layer_wise_synergy.min(), layer_wise_redundancy.min())
    combined_max = max(layer_wise_synergy.max(), layer_wise_redundancy.max())

    cax3 = axs[0].matshow(layer_wise_synergy, cmap='viridis', vmin=combined_min, vmax=combined_max)
    axs[0].set_title('Layer-wise Synergy')
    axs[0].set_xlabel('Layer')
    axs[0].set_ylabel('Layer')

    cax4 = axs[1].matshow(layer_wise_redundancy, cmap='viridis', vmin=combined_min, vmax=combined_max)
    axs[1].set_title('Layer-wise Redundancy')
    axs[1].set_xlabel('Layer')
    axs[1].set_ylabel('Layer')

    fig.colorbar(cax4, ax=axs, orientation='vertical', fraction=0.1, pad=0.02)

    if save and save_path_layerwise:
        os.makedirs(os.path.dirname(save_path_layerwise), exist_ok=True)
        plt.savefig(save_path_layerwise)
    else:
        plt.show()
    plt.close()

def plot_synergy_redundancy_PhiID(synergy_matrices, redundancy_matrices, base_plot_path=None, save=True, save_separately=False, save_combined_vertical=False):
    # plt.rcParams.update({'font.size': 15})

    if not base_plot_path:
        base_plot_path = constants.PLOTS_SYNERGY_REDUNDANCY_DIR 
    
    for metric, synergy_matrix in synergy_matrices.items():
        redundancy_matrix = redundancy_matrices[metric]
        layer_wise_synergy = layer_wise_matrix(synergy_matrix)
        layer_wise_redundancy = layer_wise_matrix(redundancy_matrix)

        if save_separately:
            plot_matrix(
                synergy_matrix,
                title='Synergy',
                xlabel='Attention Head',
                ylabel='Attention Head',
                save_path=f"{base_plot_path}{metric}/1-Synergy_Matrix.png" if save else None,
                save=save
            )

            plot_matrix(
                redundancy_matrix,
                title='Redundancy',
                xlabel='Attention Head',
                ylabel='Attention Head',
                save_path=f"{base_plot_path}{metric}/2-Redundancy_Matrix.png" if save else None,
                save=save
            )

            plot_matrix(
                layer_wise_synergy,
                title='Layer-wise Synergy',
                xlabel='Layer',
                ylabel='Layer',
                save_path=f"{base_plot_path}{metric}/3-Layer_wise_Synergy_Matrix.png" if save else None,
                save=save
            )

            plot_matrix(
                layer_wise_redundancy,
                title='Layer-wise Redundancy',
                xlabel='Layer',
                ylabel='Layer',
                save_path=f"{base_plot_path}{metric}/4-Layer_wise_Redundancy_Matrix.png" if save else None,
                save=save
            )
        elif save_combined_vertical:
            headwise_path = f"{base_plot_path}{metric}/1-Headwise_Matrices.png"
            layerwise_path = f"{base_plot_path}{metric}/2-Layerwise_Matrices.png"
            plot_synergy_redundancy_combined_vertical(
                synergy_matrix, redundancy_matrix,
                save_path_headwise=headwise_path if save else None,
                save_path_layerwise=layerwise_path if save else None,
                layer_wise_synergy=layer_wise_synergy, 
                layer_wise_redundancy=layer_wise_redundancy,
                save=save
            )
        else:
            plot_path = f"{base_plot_path}{metric}/1-Synergy-Redundancy_Matrices.png"
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))

            cax1 = axs[0, 0].matshow(synergy_matrix, cmap='viridis')
            fig.colorbar(cax1, ax=axs[0, 0])
            axs[0, 0].set_title('Synergy')
            axs[0, 0].set_xlabel('Attention Head')
            axs[0, 0].set_ylabel('Attention Head')
            axs[0, 0].xaxis.set_ticks_position('bottom')
            axs[0, 0].xaxis.set_label_position('bottom')

            cax2 = axs[0, 1].matshow(redundancy_matrix, cmap='viridis')
            fig.colorbar(cax2, ax=axs[0, 1])
            axs[0, 1].set_title('Redundancy')
            axs[0, 1].set_xlabel('Attention Head')
            axs[0, 1].set_ylabel('Attention Head')
            axs[0, 1].xaxis.set_ticks_position('bottom')
            axs[0, 1].xaxis.set_label_position('bottom')

            cax3 = axs[1, 0].matshow(layer_wise_synergy, cmap='viridis')
            fig.colorbar(cax3, ax=axs[1, 0])
            axs[1, 0].set_title('Layer-wise Synergy')
            axs[1, 0].set_xlabel('Layer')
            axs[1, 0].set_ylabel('Layer')
            axs[1, 0].xaxis.set_ticks_position('bottom')
            axs[1, 0].xaxis.set_label_position('bottom')

            cax4 = axs[1, 1].matshow(layer_wise_redundancy, cmap='viridis')
            fig.colorbar(cax4, ax=axs[1, 1])
            axs[1, 1].set_title('Layer-wise Redundancy')
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



import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import constants  # assuming you have this module

def plot_all_PhiID(global_matrices, base_plot_path=None, save=True):
    if not base_plot_path:
        base_plot_path = constants.PLOTS_SYNERGY_REDUNDANCY_DIR 

    for metric, matrices in global_matrices.items():
        num_plots = len(matrices)
        fig, axs = plt.subplots(2, 8, figsize=(40, 10))  # Adjust layout based on num_plots
        axs = np.array(axs).reshape(-1)  # Flatten array

        # Compute global min/max for consistent color scaling
        global_min = min(matrix.min() for matrix in matrices.values())
        global_max = max(matrix.max() for matrix in matrices.values())

        # Custom colormap: Blue (negative) → Light Blue (near zero) → Light Red (small positive) → Red (large positive)
        custom_cmap = LinearSegmentedColormap.from_list(
            "BlueRed",
            ["darkblue", "lightyellow", "red", "darkred"],
            N=500  # Smooth transition
        )

        # Plot each matrix with the common color scale
        for idx, (key, matrix) in enumerate(matrices.items()):
            im = axs[idx].matshow(matrix, cmap=custom_cmap, vmin=global_min, vmax=global_max)
            axs[idx].set_title(f'{constants.ATOM_NAMES[key]}', fontsize=14)
            axs[idx].set_xlabel('Attention Head')
            axs[idx].set_ylabel('Attention Head')

        # Hide unused subplots
        for j in range(idx + 1, len(axs)):
            axs[j].axis('off')

        plt.tight_layout()

        # Common colorbar
        sm = ScalarMappable(cmap=custom_cmap, norm=Normalize(vmin=global_min, vmax=global_max))
        sm.set_array([])
        fig.colorbar(sm, ax=axs.ravel().tolist(), orientation='horizontal', fraction=0.02, pad=0.04, label="Relative Contribution of Information Atoms")

        if save:
            plot_path = os.path.join(base_plot_path, f'{metric}/2-All_PhiID_Matrices.png')
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path, dpi=300)
        else:
            plt.show()
        plt.close()




def plot_single_matrix(matrix, title, save_path=None, save=True):
    fig, ax = plt.subplots(figsize=(10, 10))  # Adjust the figsize as needed
    cax = ax.matshow(matrix, cmap='viridis')
    fig.colorbar(cax, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Attention Head')
    ax.set_ylabel('Attention Head')
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_label_position('left')
    plt.tight_layout()
    
    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path + 'matrix.png')
        plt.close()
    else:
        plt.show()

def compute_average(matrix, key, top_percentile=1):
    axis = 0 if key in constants.ATOMS_AVERAGE_VERTICALLY else 1
    result = []

    if top_percentile < 1:
        # Process each row or column individually
        if axis == 0:  # Average vertically, so iterate over columns
            for col in matrix.T:
                threshold = np.percentile(col, 100 - (top_percentile * 100))
                top_values = col[col >= threshold]
                result.append(np.mean(top_values))
        else:  # Average horizontally, so iterate over rows
            for row in matrix:
                threshold = np.percentile(row, 100 - (top_percentile * 100))
                top_values = row[row >= threshold]
                result.append(np.mean(top_values))
        return np.array(result)
    else:
        # Compute the mean of the entire matrix along the specified axis
        return np.mean(matrix, axis=axis)

def plot_head_averages_heatmap(head_averages, key, save_path=None, save=True, heatmap=True):
    n_layers = constants.NUM_LAYERS
    n_heads_per_layer = constants.NUM_HEADS_PER_LAYER
    if heatmap:
        head_averages_matrix = head_averages.reshape((n_layers, n_heads_per_layer))
        fig, ax = plt.subplots(figsize=(12, 5))
        heatmap = sns.heatmap(head_averages_matrix.T, annot=True, fmt=".2f", cmap="viridis", cbar=True, linewidths=0.5, linecolor='gray', cbar_kws={"shrink": 0.8})
        cbar = heatmap.collections[0].colorbar
        cbar.set_label(key, size=16)  # Adjust the size as needed
    else:
        fig, ax = plt.subplots(figsize=(16, 6))
        heads = np.arange(len(head_averages))
        ax.plot(heads, head_averages, marker='o', linestyle='-', color='darkblue')
        ax.set_xlabel('Attention Head')
        ax.set_ylabel('Average Value')
        ax.set_title(f'Average {key} per Head')
    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file_path = save_path + 'head_averages'
        file_path += '_heatmap.png' if heatmap else '_line.png'
        plt.savefig(file_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def compute_pairwise_correlations(matrices):
    keys = list(matrices.keys())
    num_keys = len(keys)
    correlation_matrix = np.zeros((num_keys, num_keys))

    for i in range(num_keys):
        for j in range(i, num_keys):
            matrix1 = matrices[keys[i]].flatten()
            matrix2 = matrices[keys[j]].flatten()
            correlation = np.corrcoef(matrix1, matrix2)[0, 1]
            correlation_matrix[i, j] = correlation
            correlation_matrix[j, i] = correlation

    return keys, correlation_matrix

def plot_matrix_correlation(correlation_matrix, keys, title, save_path):
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation coefficient')
    plt.xticks(ticks=np.arange(len(keys)), labels=keys, rotation=20)
    plt.yticks(ticks=np.arange(len(keys)), labels=keys)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_all_PhiID_separately(global_matrices, base_plot_path=None, save=True, percentiles=[1, 0.1, 0.01]):
    results = {}
    if not base_plot_path:
        base_plot_path = constants.PLOTS_SYNERGY_REDUNDANCY_DIR 
    for metric, matrices in global_matrices.items():
        base_save_path = os.path.join(base_plot_path, metric)
        base_save_path = os.path.join(base_save_path, '8-All_PhiID_Separate/')
        results[metric] = {}

        # Compute the mutual info as the sum of all the atoms 
        mutual_info = np.zeros((constants.NUM_TOTAL_HEADS, constants.NUM_TOTAL_HEADS))
        for key, matrix in matrices.items():
            mutual_info += matrix
        # Prevent division by 0
        mutual_info += 10e-9
        
        # Compute the normalized matrices 
        normalized_matrices = {key: matrix / mutual_info for key, matrix in matrices.items()}

        all_matrices = {"Unnormalized": matrices, "Normalized": normalized_matrices}

        for normalized in ["Unnormalized", "Normalized"]:
            
            # Compute the different Information Dynamics Quantitites
            for key, atoms in constants.INFORMATION_DYNAMICS.items():
                matrix = np.zeros((constants.NUM_TOTAL_HEADS, constants.NUM_TOTAL_HEADS))
                for atom in atoms:
                    if type(atom) == tuple:
                        multiplier, atom = atom
                        matrix += multiplier * all_matrices[normalized][atom]
                    elif type(atom) == str:
                        matrix += all_matrices[normalized][atom]
                    else:
                        print("Invalid atom type in constants.INFORMATION_DYNAMICS: ", atom)
                all_matrices[normalized][key] = matrix

            # Plot all the Unnormalized PhiID Atoms separately
            for key, matrix in all_matrices[normalized].items():
                save_path = base_save_path + key + '/' + normalized + '/'
                plot_single_matrix(matrix, title=key, save_path=save_path, save=save)
                head_averages = {}
                for percentile in percentiles:
                    final_save_path = save_path + f'{percentile}/'
                    head_averages[percentile] = compute_average(matrix, key, top_percentile=percentile)
                    plot_head_averages_heatmap(head_averages[percentile], key, save_path=final_save_path, heatmap=True)
                    plot_head_averages_heatmap(head_averages[percentile], key, save_path=final_save_path, heatmap=False)
                results[metric][key] = {} if key not in results[metric] else results[metric][key]
                results[metric][key][normalized] = {} if normalized not in results[metric][key] else results[metric][key][normalized]
                results[metric][key][normalized]['matrix'] = matrix
                results[metric][key][normalized]['head_averages'] = head_averages

            # Compute correlations between all matrices
            matrices = {}
            for key in all_matrices[normalized].keys():
                print(key)
                matrices[key] = results[metric][key][normalized]['matrix']
            keys, correlation_matrix = compute_pairwise_correlations(matrices)
            correlation_title = f"{metric} - {normalized} Correlations"
            correlation_save_path = os.path.join(base_save_path, f"{normalized}_correlations.png")
            plot_matrix_correlation(correlation_matrix, keys, correlation_title, correlation_save_path)
            results[metric][f"{normalized}_correlations"] = correlation_matrix

            # Compute correlations between the head averages activations
            for percentile in percentiles:
                head_averages = {}
                for key in all_matrices[normalized].keys():
                    print(key)
                    head_averages[key] = results[metric][key][normalized]['head_averages'][percentile]
                keys, correlation_matrix = compute_pairwise_correlations(head_averages)
                correlation_title = f"Head Averages Correlations for Percentile {percentile}"
                correlation_save_path = os.path.join(base_save_path, f"{normalized}_head_averages_correlations_percentile_{percentile}.png")
                plot_matrix_correlation(correlation_matrix, keys, correlation_title, correlation_save_path)
                results[metric][f"{normalized}_head_averages_correlations"] = {} if f"{normalized}_head_averages_correlations" not in results[metric] else results[metric][f"{normalized}_head_averages_correlations"]
                results[metric][f"{normalized}_head_averages_correlations"][percentile] = correlation_matrix

    
    return results

def plot_box_plot_information_dynamics(results_all_phid, atom_or_dynamics = "atoms", save=False, base_plot_path=None):
    if atom_or_dynamics == "atoms":
        information_dynamics = ['rtr', 'rtx', 'rty', 'rts', 'xtr', 'xtx', 'xty', 'xts', 'ytr', 'ytx', 'yty', 'yts', 'str', 'stx', 'sty', 'sts']
    elif atom_or_dynamics == "dynamics":
        information_dynamics = ["storage", "transfer", "downward_causation", "upward_causation", "erasure", "copy"]
    
    if len(list(results_all_phid.keys())) == 1:
        key = list(results_all_phid.keys())[0]
    else:
        key = constants.ATTENTION_MEASURE

    information_dynamics_dict = {information_dynamic: results_all_phid[key][information_dynamic]["Unnormalized"]["head_averages"][1] for information_dynamic in information_dynamics}

    # Convert the dictionary to a DataFrame for easier plotting
    df = pd.DataFrame(information_dynamics_dict)

    # Melt the DataFrame to have a long format
    df_melted = df.melt(var_name='Information Dynamic', value_name='Values')

    # Create the box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Information Dynamic', y='Values', data=df_melted)

    # Add titles and labels
    if atom_or_dynamics == "atoms":
        plt.title('Box Plot of Information Atoms')
        plt.xlabel('Information Atom')
    elif atom_or_dynamics == "dynamics":
        plt.title('Box Plot of Information Dynamics')
        plt.xlabel('Information Dynamic')
    plt.ylabel('Values')

    if save:
        if base_plot_path is None:
            base_plot_path = constants.PLOTS_SYNERGY_REDUNDANCY_DIR + 'random_walk_time_series' + '/'
        base_plot_path = os.path.join(base_plot_path, key)
        os.makedirs(base_plot_path, exist_ok=True)

        if atom_or_dynamics == "atoms":
            plt.savefig(os.path.join(base_plot_path, '9-box_plot_information_atoms.png'))
        elif atom_or_dynamics == "dynamics":
            plt.savefig(os.path.join(base_plot_path, '10-box_plot_information_dynamics.png'))
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
            ax.set_xticklabels([f"{i}" for i in range(num_layers)])
            ax.set_yticks(np.arange(num_heads_per_layer) + 0.5)  # Change: Added set_yticks
            ax.set_yticklabels([f"{i}" for i in range(num_heads_per_layer)], rotation=0)
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
            ax.set_xticklabels([f"{i}" for i in range(num_layers)], rotation=0)
            ax.set_yticks(np.arange(num_heads_per_layer) + 0.5)  # Change: Added set_yticks
            ax.set_yticklabels([f"{i}" for i in range(num_heads_per_layer)], rotation=0)
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

def plot_gradient_rank(gradient_ranks, base_plot_path=None, save=False, figsize=(12, 5), use_heatmap=False, show_numbers=True, num_heads_per_layer=constants.NUM_HEADS_PER_LAYER, title=True):
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

            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(ranks_matrix.T, annot=show_numbers, fmt="d" if show_numbers else None, cmap="viridis", cbar=True, linewidths=0.5,
                        linecolor='gray', cbar_kws={"shrink": 0.8, "label": 'Synergy - Redundancy Rank'})
            ax.set_xticks(np.arange(num_layers) + 0.5)
            ax.set_xticklabels([f"{i}" for i in range(num_layers)], rotation=45, ha="right")
            ax.set_yticks(np.arange(num_heads_per_layer) + 0.5)
            ax.set_yticklabels([f"{i}" for i in range(num_heads_per_layer)], rotation=0)
            if title:
                ax.set_title(f'Synergy - Redundancy Rank Heatmap')
            ax.set_xlabel("Layer")
            ax.set_ylabel("Head")
            plt.tight_layout()
        else:
            fig, ax = plt.subplots(figsize=(16, 6))
            ax.plot(heads, ranks, marker='o', linestyle='-', color='darkblue')

            ax.set_xlabel('Attention Head')
            ax.set_ylabel('Synergy - Redundancy Rank')
            if title:
                ax.set_title(f'Synergy-Redundancy Rank')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        plot_path = f"{base_plot_path}{metric}/5-Synergy-Redundancy_Rank_per_Head.png"

        if save:
            plt.tight_layout()
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path, bbox_inches='tight')
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






