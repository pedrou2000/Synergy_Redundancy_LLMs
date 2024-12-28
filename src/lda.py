import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
from time_series_generation import *
from matplotlib.colors import ListedColormap
import seaborn as sns

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


def plot_category_heatmap(lda, category_index, num_layers, num_heads, category_name, save=False, plot_path=None):
    """
    Plots a heatmap for a single category showing attention head contributions.

    Args:
    lda (LDA object): Fitted LDA model.
    category_index (int): Index of the category in LDA coefficients.
    num_layers (int): Number of transformer layers.
    num_heads (int): Number of heads per layer.
    category_name (str): Name of the cognitive task category.
    save (bool): Whether to save the plot.
    plot_path (str): Directory to save the plot.

    Returns:
    None. Displays or saves the heatmap.
    """
    # Extract coefficients for the specific category and reshape to (num_layers, num_heads)
    category_coefficients = lda.coef_[category_index].reshape(num_layers, num_heads)[::-1, :]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        category_coefficients,
        cmap="coolwarm",
        annot=False,
        cbar=True,
        xticklabels=[f"H{h+1}" for h in range(num_heads)],
        yticklabels=[f"L{l+1}" for l in reversed(range(num_layers))]
    )
    plt.title(f"Attention Head Contributions for {category_name}")
    plt.xlabel("Attention Heads")
    plt.ylabel("Layers")
    plt.tight_layout()

    if save:
        if not plot_path:
            plot_path = constants.PLOTS_LDA 
        os.makedirs(plot_path, exist_ok=True)
        plt.savefig(f"{plot_path}{category_name.replace(' ', '_')}_heatmap.png", bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_global_head_contributions(lda, num_layers, num_heads, save=False, base_plot_path=None):
    """
    Plots heatmaps showing how the first and second LDA directions are composed of attention heads.

    Args:
    lda (LDA object): Fitted LDA model.
    num_layers (int): Number of transformer layers.
    num_heads (int): Number of heads per layer.
    save (bool): Whether to save the plots.
    base_plot_path (str): Directory path to save the plots.

    Returns:
    None. Displays or saves the plots.
    """
    # Extract coefficients for the first and second discriminant directions
    first_direction = lda.scalings_[:, 0].reshape(num_layers, num_heads)  # First LDA direction
    second_direction = lda.scalings_[:, 1].reshape(num_layers, num_heads)  # Second LDA direction

    directions = {"First Direction": first_direction, "Second Direction": second_direction}

    for direction_name, direction in directions.items():
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            direction,
            cmap="coolwarm",
            annot=False,
            cbar=True,
            xticklabels=[f"H{h+1}" for h in range(num_heads)],
            yticklabels=[f"L{l+1}" for l in reversed(range(num_layers))],  # Reversed for bottom-to-top layers
        )
        plt.title(f"Attention Head Contributions to {direction_name}")
        plt.xlabel("Attention Heads")
        plt.ylabel("Layers")
        plt.tight_layout()

        if save:
            if not base_plot_path:
                base_plot_path = "plots/global_head_contributions/"
            os.makedirs(base_plot_path, exist_ok=True)
            plt.savefig(f"{base_plot_path}{direction_name.replace(' ', '_').lower()}_heatmap.png", bbox_inches="tight")
            plt.close()
        else:
            plt.show()

import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

def plot_head_category_map(lda, num_layers, num_heads, categories, save=False, base_plot_path=None):
    """
    Plots a map showing the cognitive category contributing most to each attention head.

    Args:
    lda (LDA object): Fitted LDA model.
    num_layers (int): Number of transformer layers.
    num_heads (int): Number of heads per layer.
    categories (list): List of cognitive task category names.
    save (bool): Whether to save the plot.
    base_plot_path (str): Directory path to save the plot.

    Returns:
    None. Displays or saves the plot.
    """
    # Extract coefficients for all categories and reshape into (num_layers, num_heads, num_categories)
    category_contributions = lda.coef_.reshape(len(categories), num_layers, num_heads).transpose(1, 2, 0)  # (layers, heads, categories)

    # Find the index of the category with the maximum absolute contribution for each head
    max_contribution_indices = np.argmax(np.abs(category_contributions), axis=2)  # Shape: (layers, heads)

    # Create a colormap with the exact number of categories
    cmap = ListedColormap(plt.cm.tab10.colors[:len(categories)])

    plt.figure(figsize=(12, 8))
    # Pass the boundaries of each category directly to the colorbar
    bounds = list(range(len(categories) + 1))  # One boundary per category
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    ax = sns.heatmap(
        max_contribution_indices,
        cmap=cmap,
        norm=norm,
        annot=False,
        cbar=True,
        xticklabels=[f"H{h+1}" for h in range(num_heads)],
        yticklabels=[f"L{l+1}" for l in reversed(range(num_layers))],  # Reversed for bottom-to-top layers
        cbar_kws={"boundaries": bounds, "label": "Cognitive Categories"}
    )
    
    # Set colorbar labels dynamically
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([i + 0.5 for i in range(len(categories))])  # Midpoints between boundaries
    cbar.set_ticklabels(categories)

    plt.title("Cognitive Category with Highest Contribution per Attention Head")
    plt.xlabel("Attention Heads")
    plt.ylabel("Layers")
    plt.tight_layout()

    if save:
        if not base_plot_path:
            base_plot_path = "plots/global_head_category_map/"
        os.makedirs(base_plot_path, exist_ok=True)
        plt.savefig(f"{base_plot_path}head_category_map.png", bbox_inches="tight")
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
    None. Displays or saves plots of the LDA results.
    """
    if base_plot_path is None:
        base_plot_path = constants.PLOTS_HEAD_ACTIVATIONS_COGNITIVE_TASKS
    for metric, attention_weights in all_attention_weights.items():
        # Remove resting state category if in data
        if constants.RESTING_STATE_CATEGORY in attention_weights.keys():
            attention_weights.pop(constants.RESTING_STATE_CATEGORY)

        # Reshape the data
        X, y = reshape_data(attention_weights)

        # Apply LDA
        X_r, lda = apply_lda(X, y, n_components)

        # Get the labels from the keys of the dictionary, which represent categories
        labels = list(attention_weights.keys())
        # print(f"Labels: {labels}")

        # Plot the LDA projection
        plot_lda_results(X_r, y, labels, save, base_plot_path + metric + '/')
        num_layers, num_heads_per_layer = attention_weights[labels[0]].shape[:2]
        plot_global_head_contributions(
            lda=lda,
            num_layers=num_layers,
            num_heads=num_heads_per_layer,
            save=True,  # Set to True to save the plots
            base_plot_path=base_plot_path + metric + '/5-Component_Contributions/'
        )


        # Plot separate heatmaps for each category
        for category_index, category_name in enumerate(labels):
            plot_category_heatmap(
                lda,
                category_index,
                num_layers,
                num_heads_per_layer,
                category_name,
                save=save,
                plot_path=base_plot_path + metric + '/4-PCA_Head_Contributions/'
            )
        

        # Plot map of most contributing categories per head
        plot_head_category_map(
            lda,
            num_layers,
            num_heads_per_layer,
            categories=labels,
            save=save,
            base_plot_path=base_plot_path + metric + '/6-Cognitive_Category-Head_Map/'
        )
