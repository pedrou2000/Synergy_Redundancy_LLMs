from time_series_generation import *

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr

def plot_all_category_diffs_vs_syn_red_grad_rank(stats_dict, gradient_ranks, rest_category_prefix='rest', save=False, base_plot_path=None, reorder=True, mean_instead_of_rest=False):
    categories = [cat for cat in stats_dict.keys() if not cat.startswith('rest')]
    slopes = []
    correlations = []
    regression_params = []  # To store slope and intercept for each category

    # Set up the subplot dimensions
    n = len(categories)
    cols = 3  # Define how many columns you want in your subplot grid
    rows = n // cols + (n % cols > 0)  # Calculate required number of rows
    fig, axs = plt.subplots(rows, cols, figsize=(cols*6, rows*4), squeeze=False)

    # Compute the global mean across all categories
    all_means = np.vstack([stats_dict[cat][:, :, 0].flatten() for cat in categories])
    global_mean = np.mean(all_means, axis=0)  # This is the mean of all categories' means


    for i, category in enumerate(categories):
        # Perform the existing process for each category
        stats_category = stats_dict[category]
        stats_rest = stats_dict[rest_category_prefix]
        
        stats_category_means = stats_category[:, :, 0].flatten()
        stats_rest_means = stats_rest[:, :, 0].flatten()
        diff_means = stats_category_means - stats_rest_means
        if mean_instead_of_rest:
            diff_means = stats_category_means - global_mean

        gradient_ranks_ordered = np.array([gradient_ranks[i] for i in range(1, len(diff_means) + 1)])
        reorder_indices = np.argsort(gradient_ranks_ordered)

        diff_means_reordered = diff_means
        if reorder:
            diff_means_reordered = diff_means[reorder_indices]

        x = np.arange(len(diff_means_reordered))
        slope, intercept = np.polyfit(x, diff_means_reordered, 1)
        regression_params.append((slope, intercept))  # Store slope and intercept
        
        # Calculate Pearson correlation coefficient
        correlation_coefficient, _ = pearsonr(x, diff_means_reordered)

        # Plotting in subplots
        ax = axs[i % rows, i // rows]
        ax.plot(diff_means_reordered, marker='o', linestyle='-', color='darkblue', label='Original Data')
        ax.plot(x, slope * x + intercept, color='red', label=f'Slope = {slope:.5f}')
        
        ax.set_title(f'{category}')
        if reorder:
            ax.set_xlabel('Synergy - Redundancy Gradient Rank')
        else:
            ax.set_xlabel('Head Index')
        ax.set_ylabel('Diff in Avg Activation')
        ax.legend()

        slopes.append(slope)
        correlations.append(correlation_coefficient)

    # Adjust layout and hide empty subplots
    plt.tight_layout()
    for j in range(i+1, rows*cols):
        axs.flatten()[j].axis('off')

    if save:
        if not base_plot_path:
            base_plot_path = constants.PLOTS_ACTIVATIONS_SYN_RED_GRAD + datetime.now().strftime("%Y%m%d_%H%M%S") + '/'
        
        plot_path = f"{base_plot_path}1-diff_activations_per_head.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, bbox_inches='tight')

    # Final plot for categories vs slopes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(np.arange(len(categories)) - 0.2, slopes, 0.4, label='Slope', tick_label=categories)
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Slope of Fitted Line', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_title('Slope and Pearson Correlation for Each Category')
    ax1.set_xticks(np.arange(len(categories)))
    ax1.set_xticklabels(categories, rotation=45, ha="right")

    # Create another y-axis for the Pearson correlation coefficients
    ax2 = ax1.twinx()
    ax2.bar(np.arange(len(categories)) + 0.2, correlations, 0.4, label='Pearson Correlation', color='tab:orange')
    ax2.set_ylabel('Pearson Correlation Coefficient', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    fig.tight_layout()
    
    if save:
        if not base_plot_path:
            base_plot_path = constants.PLOTS_ACTIVATIONS_SYN_RED_GRAD + datetime.now().strftime("%Y%m%d_%H%M%S") + '/'
        
        plot_path = f"{base_plot_path}2-slope_per_category.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, bbox_inches='tight')

    # Overlay plot for regression lines
    plt.figure(figsize=(10, 6))
    for i, (slope, intercept) in enumerate(regression_params):
        x = np.linspace(0, len(diff_means_reordered) - 1, num=len(diff_means_reordered))
        plt.plot(x, slope * x + intercept, label=f'{categories[i]} (Slope: {slope:.4f})')

    plt.title('Overlay of Regression Lines for All Categories')
    plt.xlabel('Synergy - Redundancy Gradient Rank')
    plt.ylabel('Diff in Avg Activation')
    plt.legend()

    if save:
        if not base_plot_path:
            base_plot_path = constants.PLOTS_ACTIVATIONS_SYN_RED_GRAD + datetime.now().strftime("%Y%m%d_%H%M%S") + '/'
        
        plot_path = f"{base_plot_path}3-overlay_regression_slopes.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, bbox_inches='tight')

    plt.show()

def get_head_number(layer, head_index):
    """
    Given a layer and a head index, calculate the global head number.
    
    Args:
    layer (int): Zero-indexed number of the layer.
    head_index (int): Zero-indexed number of the head within the given layer.

    Returns:
    int: One-indexed global head number.
    """
    return layer * 8 + head_index + 1

def get_layer_and_head(head_number):
    """
    Given a one-indexed global head number, calculate the corresponding zero-indexed layer and head index.
    
    Args:
    head_number (int): One-indexed global head number.

    Returns:
    tuple: (layer, head_index), where both are zero-indexed.
    """
    head_number -= 1  # Adjust for zero-indexing
    return head_number // 8, head_number % 8

def plot_most_syn_red_tasks(stats_dict, gradient_ranks, top_n=10):
    # Filter out categories starting with 'rest'
    categories = [cat for cat in stats_dict.keys() if not cat.startswith('rest')]

    # Initialize task counts for both synergy and redundancy
    task_count_synergy = {task: 0 for task in categories}
    task_count_redundancy = {task: 0 for task in categories}

    # Sorting heads based on synergy and redundancy
    sorted_by_synergy = sorted(gradient_ranks.items(), key=lambda item: item[1], reverse=True)[:top_n]
    sorted_by_redundancy = sorted(gradient_ranks.items(), key=lambda item: item[1])[:top_n]
    # print(sorted_by_synergy)
    # print(sorted_by_redundancy)

    # Iterate over the top synergistic heads
    for head in [head for head, rank in sorted_by_synergy]:
        layer, head_idx = get_layer_and_head(head)  # Using the utility function
        # Only consider activations for relevant categories
        activations = {task: stats_dict[task][layer][head_idx][0] for task in categories}
        max_task = max(activations, key=activations.get)
        task_count_synergy[max_task] += 1

    # Iterate over the top redundant heads
    for head in [head for head, rank in sorted_by_redundancy]:
        layer, head_idx = get_layer_and_head(head)  # Using the utility function
        activations = {task: stats_dict[task][layer][head_idx][0] for task in categories}
        max_task = max(activations, key=activations.get)
        task_count_redundancy[max_task] += 1

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))
    axs[0].bar(task_count_synergy.keys(), task_count_synergy.values(), color='blue')
    axs[0].set_title('Top ' +str(top_n) +' Synergistic Heads and Their Predominant Tasks')
    axs[0].set_xlabel('Task')
    axs[0].set_ylabel('Count of Top Synergistic Heads')
    axs[0].tick_params(axis='x', rotation=10)

    axs[1].bar(task_count_redundancy.keys(), task_count_redundancy.values(), color='red')
    axs[1].set_title('Top ' +str(top_n) +' Redundant Heads and Their Predominant Tasks')
    axs[1].set_xlabel('Task')
    axs[1].set_ylabel('Count of Top Redundant Heads')
    axs[1].tick_params(axis='x', rotation=10)

    plt.tight_layout()
    plt.show()

def plot_rank_most_activated_heads_per_task(stats_dict, gradient_ranks, top_n=10):
    categories = [cat for cat in stats_dict.keys() if not cat.startswith('rest')]
    
    # Initialize a dictionary to store the average ranks for each task
    avg_rank_per_task = {task: [] for task in categories}

    n_layers = stats_dict[categories[0]].shape[0]
    n_heads = stats_dict[categories[0]].shape[1]
    
    # Calculate the average activation for each head across all tasks
    head_avg_activations = {}
    for layer in range(n_layers):  # Assuming there are 18 layers
        for head_idx in range(n_heads):  # Assuming 8 heads per layer
            activations = [stats_dict[task][layer][head_idx][0] for task in categories]
            head_number = get_head_number(layer, head_idx)
            head_avg_activations[head_number] = np.mean(activations)
    
    # For each task, find the top 10 heads based on their activation compared to their average
    for task in categories:
        head_activations = []
        for layer in range(n_layers):
            for head_idx in range(n_heads):
                head_number = get_head_number(layer, head_idx)
                activation = stats_dict[task][layer][head_idx][0]
                if activation > head_avg_activations[head_number]:
                    head_activations.append((head_number, activation- head_avg_activations[head_number]))
        
        # Sort by activation and take the top N
        top_heads = sorted(head_activations, key=lambda x: x[1], reverse=True)[:top_n]
        # print(task, top_heads)
        
        # Gather the synergy-redundancy ranks of these heads
        ranks = [gradient_ranks[head] for head, _ in top_heads]
        avg_rank_per_task[task].append(np.mean(ranks))  # Store the average rank

    # Plot the average synergy-redundancy ranks for each task
    fig, ax = plt.subplots(figsize=(14, 5))
    tasks = list(avg_rank_per_task.keys())
    avg_ranks = [np.mean(avg_rank_per_task[task]) for task in tasks]
    
    ax.bar(tasks, avg_ranks, color='green')
    ax.set_title('Average Synergy-Redundancy Rank of Top ' +str(top_n) +' Activated Heads per Task')
    ax.set_xlabel('Task')
    ax.set_ylabel('Average Rank')
    ax.tick_params(axis='x', rotation=5)

    plt.tight_layout()
    plt.show()

def plot_average_head_activation_per_task(stats_dict):
    categories = [cat for cat in stats_dict.keys() if not cat.startswith('rest')]
    
    avg_activation_per_task = {}
    sd_activation_per_task = {}
    
    # Iterate through each task and calculate the average activation and gather the precomputed standard deviations
    for task in categories:
        n_layers = stats_dict[task].shape[0]
        n_heads = stats_dict[task].shape[1]
        total_activation = 0
        sd_values = []
        count = 0
        
        for layer in range(n_layers):
            for head_idx in range(n_heads):
                activation = stats_dict[task][layer][head_idx][0]
                sd = stats_dict[task][layer][head_idx][1]  # Directly use the precomputed SD
                total_activation += activation
                sd_values.append(sd)
                count += 1
        
        avg_activation_per_task[task] = total_activation / count
        sd_activation_per_task[task] = np.mean(sd_values)  # Use the average of the SDs for each task
    
    # Plotting the average activations for each task using points with error bars
    fig, ax = plt.subplots(figsize=(14, 5))
    tasks = list(avg_activation_per_task.keys())
    avg_activations = [avg_activation_per_task[task] for task in tasks]
    sd_activations = [sd_activation_per_task[task] for task in tasks]
    
    # Convert task names to indices for plotting
    task_indices = np.arange(len(tasks))
    
    ax.errorbar(task_indices, avg_activations, yerr=sd_activations, fmt='o', color='blue', ecolor='lightgray', elinewidth=3, capsize=0)
    ax.set_title('Average Activation Per Task')
    ax.set_xlabel('Task')
    ax.set_ylabel('Average Activation')
    ax.set_xticks(task_indices)
    ax.set_xticklabels(tasks, rotation=5)

    plt.tight_layout()
    plt.show()

