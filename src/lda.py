import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
from time_series_generation import *

def reshape_data(all_attention_weights):
    categories = all_attention_weights.keys()
    X = []  # This will store the flattened attention weights
    y = []  # This will store the labels (categories)

    for label, category in enumerate(categories):
        num_layers, num_heads_per_layer = all_attention_weights[category].shape[:2]
        data = all_attention_weights[category].reshape(-1, num_heads_per_layer*num_layers)  # Reshape data
        X.append(data)
        y.extend([label] * data.shape[0])

    X = np.vstack(X)
    y = np.array(y)
    return X, y

def apply_lda(X, y, n_components=2):
    lda = LDA(n_components=n_components)
    X_r = lda.fit_transform(X, y)
    return X_r, lda

def plot_lda_results(X_r, y, labels, save=False, base_plot_path=None):
    # plt.rcParams.update({'font.size': 15}) 
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], alpha=0.8, label=label)
    plt.title('Discriminant Analysis of Attention Head Activations Across Cognitive Task Categories')
    plt.xlabel('Linear Combination of Head Activations Differentiating Cognitive Task - Axis 1')
    plt.ylabel('Linear Combination of Head Activations Differentiating Cognitive Task - Axis 2')
    plt.legend(loc='best', shadow=False, scatterpoints=1)

    if save:
        if not base_plot_path:
            base_plot_path = constants.PLOTS_LDA 
        
        plot_path = f"{base_plot_path}3-LDA.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def perform_lda_analysis(all_attention_weights, n_components=2, save=False, base_plot_path=None):
    """
    Performs LDA analysis on given attention weights and plots the results.

    Args:
    all_attention_weights (dict): Dictionary of attention weights with categories as keys.
    n_components (int): Number of components for LDA.

    Returns:
    None. Displays a plot of the LDA results.
    """
    if base_plot_path is None:
        base_plot_path = constants.PLOTS_HEAD_ACTIVATIONS_COGNITIVE_TASKS 
    for metric, attention_weights in all_attention_weights.items():
        # Reshape the data
        X, y = reshape_data(attention_weights)

        # Apply LDA
        X_r, lda = apply_lda(X, y, n_components)

        # Get the labels from the keys of the dictionary, which represent categories
        labels = list(attention_weights.keys())

        # Plot the results
        plot_lda_results(X_r, y, labels, save, base_plot_path+ metric + '/')
