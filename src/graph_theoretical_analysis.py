from bct import efficiency_wei, modularity_louvain_und
import matplotlib.pyplot as plt
import numpy as np
import constants
import pickle, os

def compare_synergy_redundancy(synergy_matrices, redundancy_matrices, selected_metrics=constants.METRICS_TRANSFORMER, verbose=False):
    """
    Compare synergy and redundancy matrices in terms of global efficiency and modularity.

    Parameters:
    - synergy_matrices: dict of synergy matrices, where each key is a metric.
    - redundancy_matrices: dict of redundancy matrices, where each key is a metric.
    - selected_metrics: list of metrics to consider in the analysis.
    - verbose: bool, if True, prints detailed comparison results.

    Returns:
    - A dictionary of the form results[metric][measure][matrix_type], where: metric is a key in selected_metrics, measure is either 
        "global_efficiency" or "modularity", and matrix_type is either "synergy" or "redundancy".
    """

    # Normalize the matrices
    synergy_matrices_norm = {metric: synergy_matrices[metric] / np.max(synergy_matrices[metric]) for metric in selected_metrics}
    redundancy_matrices_norm = {metric: redundancy_matrices[metric] / np.max(redundancy_matrices[metric]) for metric in selected_metrics}

    # Make sure the matrices are symmetric
    synergy_matrices_norm = {metric: (matrix + matrix.T) / 2 for metric, matrix in synergy_matrices_norm.items()}
    redundancy_matrices_norm = {metric: (matrix + matrix.T) / 2 for metric, matrix in redundancy_matrices_norm.items()}

    # Initialize dictionaries to store comparison results
    results = {}
    for metric in selected_metrics:
        results[metric] = {}
        results[metric]["global_efficiency"] = {}
        results[metric]["modularity"] = {}

    for metric in selected_metrics:
        glob_eff_syn = efficiency_wei(synergy_matrices_norm[metric])
        glob_eff_red = efficiency_wei(redundancy_matrices_norm[metric])
        results[metric]["global_efficiency"]["synergy"] = glob_eff_syn
        results[metric]["global_efficiency"]["redundancy"] = glob_eff_red
        
        if verbose:
            print(f"Global Efficiency for Synergy Matrix ({metric}): {glob_eff_syn}, Global Efficiency for Redundancy Matrix ({metric}): {glob_eff_red}")

    for metric in selected_metrics:
        c, modularity_synergy = modularity_louvain_und(synergy_matrices_norm[metric])
        c, modularity_redundancy = modularity_louvain_und(redundancy_matrices_norm[metric])
        results[metric]["modularity"]["synergy"] = modularity_synergy
        results[metric]["modularity"]["redundancy"] = modularity_redundancy

        if verbose:
            print(f"Modularity of Synergy Matrix ({metric}): {modularity_synergy}, Modularity of Redundancy Matrix ({metric}): {modularity_redundancy}")

    return results

def plot_graph_theoretical_results(results, save=True, base_plot_path=None):
    """
    Plot the results of the graph theoretical analysis.

    Parameters:
    - results: dict, results of the graph theoretical analysis.
    - plot: bool, if True, plots the results.

    Returns:
    - None
    """

    for metric, measures in results.items():
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"Graph Theoretical Analysis for {metric}")

        ax[0].bar(["Synergy", "Redundancy"], [measures["global_efficiency"]["synergy"], measures["global_efficiency"]["redundancy"]])
        ax[0].set_title("Global Efficiency")
        ax[0].set_ylabel("Value")
        ax[0].set_xlabel("Matrix Type")

        ax[1].bar(["Synergy", "Redundancy"], [measures["modularity"]["synergy"], measures["modularity"]["redundancy"]])
        ax[1].set_title("Modularity")
        ax[1].set_ylabel("Value")
        ax[1].set_xlabel("Matrix Type")

        if save:
            if base_plot_path is None:
                base_plot_path = constants.PLOTS_SYNERGY_REDUNDANCY_DIR + "/" + metric + "/"
            plt.savefig(base_plot_path + metric + "/7-Graph_Theoretical_Properties.png")
        else:
            plt.show()
        plt.close()

def save_graph_theoretical_results(results, base_save_path=None, file_name="graph_theoretical_results"):
    if base_save_path is None:
        base_save_path = constants.GRAPH_METRICS_DIR 
    if not os.path.exists(base_save_path):
        os.makedirs(base_save_path)
    with open(base_save_path + file_name + ".pkl", "wb") as f:
        pickle.dump(results, f)

def load_graph_theoretical_results(base_save_path=None, file_name="graph_theoretical_results"):
    if base_save_path is None:
        base_save_path = constants.GRAPH_METRICS_DIR 
    with open(base_save_path + file_name + ".pkl", "rb") as f:
        results = pickle.load(f)
    return results

def load_graph_theoretical_results(model_number, base_save_path=None, file_name="graph_theoretical_results"):
    model_folder = constants.MODEL_NAMES[model_number]["FOLDER_NAME"]
    if base_save_path is None:
        base_save_path = os.path.join("../data", model_folder, "5-Graph_Theoretical_Properties/")
    with open(os.path.join(base_save_path, file_name + ".pkl"), "rb") as f:
        results = pickle.load(f)
    return results

def compare_graph_theoretical_results(model_numbers, cognitive_task_category='average_prompts', save=True, base_plot_path=None):
    """
    Compare the results of the graph theoretical analysis for multiple models.

    Parameters:
    - model_numbers: list of int, the model numbers to compare.
    - cognitive_task_category: str, the cognitive task category to load the results from.
    - save: bool, if True, saves the plot.
    - base_plot_path: str, the base path to save the plots.

    Returns:
    - None
    """

    all_results = {model_num: load_graph_theoretical_results(model_num, file_name=cognitive_task_category) for model_num in model_numbers}

    metrics = list(all_results[model_numbers[0]].keys())
    model_names = [constants.MODEL_NAMES[model_num]["HF_NAME"].split("/")[-1] for model_num in model_numbers]
    bar_width = 0.35
    index = range(len(model_numbers))

    for metric in metrics:
        fig, ax = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
        fig.suptitle(f"Graph Theoretical Analysis Comparison for {metric}")

        # Global Efficiency Plot
        synergy_values_ge = [all_results[model_num][metric]["global_efficiency"]["synergy"] for model_num in model_numbers]
        redundancy_values_ge = [all_results[model_num][metric]["global_efficiency"]["redundancy"] for model_num in model_numbers]

        bar1 = ax[0].bar(index, synergy_values_ge, bar_width, label='Synergy Matrix')
        bar2 = ax[0].bar([p + bar_width for p in index], redundancy_values_ge, bar_width, label='Redundancy Matrix')

        ax[0].set_title("Global Efficiency")
        ax[0].set_xlabel("Model")
        ax[0].set_ylabel("Value")
        ax[0].set_xticks([p + bar_width/2 for p in index])
        ax[0].set_xticklabels(model_names)

        # Modularity Plot
        synergy_values_mod = [all_results[model_num][metric]["modularity"]["synergy"] for model_num in model_numbers]
        redundancy_values_mod = [all_results[model_num][metric]["modularity"]["redundancy"] for model_num in model_numbers]

        bar3 = ax[1].bar(index, synergy_values_mod, bar_width, label='Synergy Matrix')
        bar4 = ax[1].bar([p + bar_width for p in index], redundancy_values_mod, bar_width, label='Redundancy Matrix')

        ax[1].set_title("Modularity")
        ax[1].set_xlabel("Model")
        ax[1].set_xticks([p + bar_width/2 for p in index])
        ax[1].set_xticklabels(model_names)

        # Create a single legend for the figure
        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(handles, ['Synergy Matrix', 'Redundancy Matrix'], loc='upper right')

        if save:
            if base_plot_path is None:
                base_plot_path = constants.MODEL_COMPARISON_GRAPH_THEORETICAL_DIR
            if not os.path.exists(base_plot_path):
                os.makedirs(base_plot_path)
            plt.savefig(os.path.join(base_plot_path, metric + "-Comparison.png"))
        else:
            plt.show()
        plt.close()
    
if __name__ == "__main__":
    compare_graph_theoretical_results([2, 3, 4], save=True)
