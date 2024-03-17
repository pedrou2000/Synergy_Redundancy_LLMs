from bct import efficiency_wei, modularity_louvain_und

import numpy as np

def compare_synergy_redundancy(synergy_matrices, redundancy_matrices, selected_metrics, verbose=False):
    """
    Compare synergy and redundancy matrices in terms of global efficiency and modularity.

    Parameters:
    - synergy_matrices: dict of synergy matrices, where each key is a metric.
    - redundancy_matrices: dict of redundancy matrices, where each key is a metric.
    - selected_metrics: list of metrics to consider in the analysis.
    - verbose: bool, if True, prints detailed comparison results.

    Returns:
    - A tuple containing two dictionaries, the first with global efficiency comparison results and the second with modularity comparison results.
    """

    # Normalize the matrices
    synergy_matrices_norm = {metric: synergy_matrices[metric] / np.max(synergy_matrices[metric]) for metric in selected_metrics}
    redundancy_matrices_norm = {metric: redundancy_matrices[metric] / np.max(redundancy_matrices[metric]) for metric in selected_metrics}

    # Make sure the matrices are symmetric
    synergy_matrices_norm = {metric: (matrix + matrix.T) / 2 for metric, matrix in synergy_matrices_norm.items()}
    redundancy_matrices_norm = {metric: (matrix + matrix.T) / 2 for metric, matrix in redundancy_matrices_norm.items()}

    # Initialize dictionaries to store comparison results
    efficiency_results = {}
    modularity_results = {}

    for metric in selected_metrics:
        glob_eff_syn = efficiency_wei(synergy_matrices_norm[metric])
        glob_eff_red = efficiency_wei(redundancy_matrices_norm[metric])
        synergy_bigger_redundancy = glob_eff_syn > glob_eff_red
        efficiency_results[metric] = {'Synergy': glob_eff_syn, 'Redundancy': glob_eff_red, 'Synergy > Redundancy': synergy_bigger_redundancy}
        
        if verbose:
            print(f"Synergy bigger than Redundancy for {metric}: {synergy_bigger_redundancy}")
            print(f"Global Efficiency for Synergy Matrix ({metric}): {glob_eff_syn}, Global Efficiency for Redundancy Matrix ({metric}): {glob_eff_red}")

    for metric in selected_metrics:
        c, Q_synergy = modularity_louvain_und(synergy_matrices_norm[metric])
        c, Q_redundancy = modularity_louvain_und(redundancy_matrices_norm[metric])
        redundancy_bigger_synergy = Q_redundancy > Q_synergy
        modularity_results[metric] = {'Synergy': Q_synergy, 'Redundancy': Q_redundancy, 'Redundancy > Synergy': redundancy_bigger_synergy}

        if verbose:
            print(f"Redundancy bigger than Synergy for {metric}: {redundancy_bigger_synergy}")
            print(f"Modularity of Synergy Matrix ({metric}): {Q_synergy}, Modularity of Redundancy Matrix ({metric}): {Q_redundancy}")

    return efficiency_results, modularity_results
