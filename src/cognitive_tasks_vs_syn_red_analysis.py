from time_series_generation import *

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr

def sort_categories(categories):
    # swaps_subplots = [(0,2), (2,3), (4,3), (2,3), (3,4)]
    swaps_overlay = [(1,2), (0,1)]
    for i, j in swaps_overlay:
        categories[i], categories[j] = categories[j], categories[i]
    # print(categories)
    return categories


def plot_all_category_diffs_vs_syn_red_grad_rank(stats_dicts, gradient_ranks, ranks_per_layer_mean, save=False, base_plot_path=None, 
            reorder=True, per_layer=False, num_heads_per_layer=constants.NUM_HEADS_PER_LAYER, baseline_rest=False):
    results = {}
    for metric in constants.METRICS_TRANSFORMER:
        results[metric] = {}

    for metric, stats_dict in stats_dicts.items():
        categories = list(constants.PROMPT_CATEGORIES )
        categories.remove(constants.RESTING_STATE_CATEGORY) if constants.USING_REST_STATE else None
        categories = sort_categories(categories)
        slopes = []
        correlations = []
        regression_params = []  # To store slope and intercept for each category

        if not base_plot_path:
            base_plot_path = constants.PLOT_SYNERGY_REDUNDANCY_TASK_CORRELATIONS 

        # Set up the subplot dimensions
        n = len(categories)
        cols = 2  # Define how many columns you want in your subplot grid
        rows = n // cols + (n % cols > 0)  # Calculate required number of rows
        fig, axs = plt.subplots(rows, cols, figsize=(cols*6, rows*4), squeeze=False)

        # Compute the global mean across all categories
        if per_layer:
            all_means = np.vstack([np.mean(stats_dict[cat][:, :, 0].reshape(-1, num_heads_per_layer), axis=1) for cat in categories])
        else:
            all_means = np.vstack([stats_dict[cat][:, :, 0].flatten() for cat in categories])
        global_mean = np.mean(all_means, axis=0)

        for i, category in enumerate(categories):
            stats_category = stats_dict[category]

            if baseline_rest:
                stats_rest = stats_dict[constants.RESTING_STATE_CATEGORY]

                if per_layer:
                    stats_category_means = np.mean(stats_category[:, :, 0].reshape(-1, num_heads_per_layer), axis=1)
                    stats_rest_means = np.mean(stats_rest[:, :, 0].reshape(-1, num_heads_per_layer), axis=1)
                else:
                    stats_category_means = stats_category[:, :, 0].flatten()
                    stats_rest_means = stats_rest[:, :, 0].flatten()

                diff_means = stats_category_means - stats_rest_means
            else:
                if per_layer:
                    stats_category_means = np.mean(stats_category[:, :, 0].reshape(-1, num_heads_per_layer), axis=1)
                else:
                    stats_category_means = stats_category[:, :, 0].flatten()

                diff_means = stats_category_means - global_mean

            gradient_ranks_ordered = np.array([gradient_ranks[i] for i in range(1, len(diff_means) + 1)])
            reorder_indices = np.argsort(gradient_ranks_ordered)

            diff_means_reordered = diff_means[reorder_indices] if reorder else diff_means

            x = np.arange(len(diff_means_reordered))
            slope, intercept = np.polyfit(x, diff_means_reordered, 1)
            regression_params.append((slope, intercept))

            correlation_coefficient, _ = pearsonr(x, diff_means_reordered)
            ax = axs[i % rows, i // rows]
            if per_layer:
                num_layers = all_means.shape[1]
                # Set x-axis to label each layer explicitly
                ax.set_xticks(range(num_layers))  # Set tick positions starting from 0
                ax.set_xticklabels([str(i+1) for i in range(num_layers)])  # Label from 1 to num_layers

            # Compute the correlation
            correlation_matrix_syn_red = np.corrcoef(ranks_per_layer_mean[metric], diff_means_reordered)
            correlation_coefficient_syn_red = correlation_matrix_syn_red[0, 1]


            ax.plot(diff_means_reordered, marker='o', linestyle='-', color='darkblue', label='Original Data')
            ax.plot(x, slope * x + intercept, color='red', label=f'Linear Regression (Pearson Corr {correlation_coefficient:.2f}, Syn-Red Corr {correlation_coefficient_syn_red:.2f})')
            ax.set_title(f'{category}')
            ax.set_xlabel('Synergy - Redundancy Gradient Rank' if reorder else 'Layer Index' if per_layer else 'Head Index')
            ax.set_ylabel('Diff in Avg Activation')
            ax.legend()

            slopes.append(slope)
            correlations.append(correlation_coefficient)

            results[metric][category] = diff_means_reordered

        # Adjust layout and hide empty subplots
        plt.tight_layout()
        for j in range(i+1, rows*cols):
            axs.flatten()[j].axis('off')

        if save:
            if not base_plot_path:
                base_plot_path = constants.PLOT_SYNERGY_REDUNDANCY_TASK_CORRELATIONS + '/'
            
            plot_path = f"{base_plot_path}{metric}/1-Relative_Activations_per_Head.png"
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()  # Close the plot to prevent it from displaying

        # Final plot for categories vs slopes
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.bar(np.arange(len(categories)) - 0.2, slopes, 0.4, label='Slope', tick_label=categories)
        ax1.set_xlabel('Category')
        ax1.set_ylabel('Slope of Fitted Line', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_title('Slope and Pearson Correlation for Each Category')
        ax1.set_xticks(np.arange(len(categories)))
        ax1.set_xticklabels(categories, rotation=10)

        # Create another y-axis for the Pearson correlation coefficients
        ax2 = ax1.twinx()
        ax2.bar(np.arange(len(categories)) + 0.2, correlations, 0.4, label='Pearson Correlation', color='tab:orange')
        ax2.set_ylabel('Pearson Correlation Coefficient', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        fig.tight_layout()
        
        if save:
            if not base_plot_path:
                base_plot_path = constants.PLOT_SYNERGY_REDUNDANCY_TASK_CORRELATIONS 
            
            plot_path = f"{base_plot_path}{metric}/2-Regression_Slope_per_Category.png"
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()  # Close the plot to prevent it from displaying

        # Overlay plot for regression lines
        plt.figure(figsize=(10, 6))
        for i, (slope, intercept) in enumerate(regression_params):
            x = np.linspace(0, len(diff_means_reordered) - 1, num=len(diff_means_reordered))
            plt.plot(x, slope * x + intercept, label=f'{categories[i]}')

        plt.title('Overlay of Regression Lines for Cognitive Task Categories')
        plt.xlabel('Synergy - Redundancy Gradient Rank' if reorder else 'Layer Index' if per_layer else 'Head Index')
        plt.ylabel('Difference in Average Head Activation')
        plt.legend()

        # Set the x-axis to label each layer explicitly
        if per_layer:
            num_layers = all_means.shape[1]
            plt.xticks(range(num_layers), [str(i+1) for i in range(num_layers)])


        if save:
            if not base_plot_path:
                base_plot_path = constants.PLOT_SYNERGY_REDUNDANCY_TASK_CORRELATIONS 
            
            plot_path = f"{base_plot_path}{metric}/3-Overlay_Linear_Regressions.png"
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()  # Close the plot to prevent it from displaying
        else:
            plt.show()

    return results

def get_head_number(layer, head_index):
    """
    Given a layer and a head index, calculate the global head number.
    
    Args:
    layer (int): Zero-indexed number of the layer.
    head_index (int): Zero-indexed number of the head within the given layer.

    Returns:
    int: One-indexed global head number.
    """
    return layer * constants.NUM_HEADS_PER_LAYER + head_index + 1

def get_layer_and_head(head_number):
    """
    Given a one-indexed global head number, calculate the corresponding zero-indexed layer and head index.
    
    Args:
    head_number (int): One-indexed global head number.

    Returns:
    tuple: (layer, head_index), where both are zero-indexed.
    """
    # print("Number of heads per layer: ", constants.NUM_HEADS_PER_LAYER, "Number heads: ", head_number//constants.NUM_HEADS_PER_LAYER, " Head Number: ", head_number)
    head_number -= 1  # Adjust for zero-indexing
    return head_number // constants.NUM_HEADS_PER_LAYER, head_number % constants.NUM_HEADS_PER_LAYER

def plot_most_syn_red_tasks(stats_dicts, gradient_ranks, top_n=10):
    for metric, stats_dict in stats_dicts.items():
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

def plot_rank_most_activated_heads_per_task(stats_dicts, gradient_rankss, top_ns=[1,3,5,10,20,50], save=False, base_plot_path=None):

    for metric, gradient_ranks in gradient_rankss.items():
        if save:
            if not base_plot_path:
                base_plot_path = constants.PLOT_SYNERGY_REDUNDANCY_TASK_CORRELATIONS + metric+ '/4-Average_Rank_Top_Heads_Activated_per_Task/'
                os.makedirs(base_plot_path, exist_ok=True)
        for top_n in top_ns:
            # for metric, stats_dict in stats_dicts.items():
            stats_dict = stats_dicts[metric]
            categories = list(stats_dict.keys())
            if constants.RESTING_STATE_CATEGORY in categories:
                categories.remove(constants.RESTING_STATE_CATEGORY)

            # Reorder such that 'basic_numerical_reasoning' goes first if it exists in the list
            if 'basic_numerical_reasoning' in categories:
                categories.remove('basic_numerical_reasoning')
                categories.insert(0, 'basic_numerical_reasoning')           
            # Initialize a dictionary to store the average ranks for each task
            avg_rank_per_task = {task: [] for task in categories}

            # Calculate the average activation for each head across all tasks
            head_avg_activations = {}
            for layer in range(constants.NUM_LAYERS):  
                for head_idx in range(constants.NUM_HEADS_PER_LAYER): 
                    activations = [stats_dict[task][layer][head_idx][0] for task in categories]
                    head_number = get_head_number(layer, head_idx)
                    head_avg_activations[head_number] = np.mean(activations)
            
            # For each task, find the top 10 heads based on their activation compared to their average
            for task in categories:
                head_activations = []
                for layer in range(constants.NUM_LAYERS):
                    for head_idx in range(constants.NUM_HEADS_PER_LAYER):
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
            if save:
                plot_path = f"{base_plot_path}{top_n}_Most_Activated_Heads_per_Task.png"
                os.makedirs(os.path.dirname(plot_path), exist_ok=True)
                plt.savefig(plot_path, bbox_inches='tight')
                plt.close()  # Close the plot to prevent it from displaying
            else:
                plt.show()
        base_plot_path = None

def plot_average_head_activation_per_task(stats_dicts):
    categories = constants.PROMPT_CATEGORIES
    for metric, stats_dict in stats_dicts.items():
        
        avg_activation_per_task = {}
        sd_activation_per_task = {}
        
        # Iterate through each task and calculate the average activation and gather the precomputed standard deviations
        for task in categories:
            constants.NUM_LAYERS = stats_dict[task].shape[0]
            constants.NUM_HEADS_PER_LAYER = stats_dict[task].shape[1]
            total_activation = 0
            sd_values = []
            count = 0
            
            for layer in range(constants.NUM_LAYERS):
                for head_idx in range(constants.NUM_HEADS_PER_LAYER):
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



def compute_gradient_ranks(synergy_per_head, redundancy_per_head):
    gradient_diff = synergy_per_head - redundancy_per_head
    
    # Sort gradient_diff and get indices of sorted array
    sorted_indices = np.argsort(gradient_diff)
    
    # Create an empty array of the same length as gradient_diff to store ranks
    ranks = np.empty_like(sorted_indices)
    
    # Assign ranks; since sorted_indices are 0-based, we add 1 to make them start from 1
    ranks[sorted_indices] = np.arange(1, len(gradient_diff) + 1)
    
    # Create a dictionary where the key is the head number (1-indexed) and value is the rank
    head_ranks = {head_number: rank for head_number, rank in enumerate(ranks, start=1)}
    gradient_ranks_array = np.array([head_ranks[key] for key in sorted(head_ranks.keys())])
    return gradient_ranks_array

def compute_and_plot_gradient_activations_correlation(results_all_phid, summary_stats_prompts, per_layer=False, save=True, base_plot_path=None):

    for metric in constants.METRICS_TRANSFORMER:
        for normalized in ['Normalized', 'Unnormalized']:
            if base_plot_path is None:
                base_plot_path = constants.PLOT_SYNERGY_REDUNDANCY_TASK_CORRELATIONS
            base_save_path = base_plot_path + metric + '/' + '5-Correlations_with_Average_Activation/' + normalized + '/' + ('per_layer/' if per_layer else 'per_head/')

            synergy_per_head = results_all_phid[metric]['sts'][normalized]['head_averages']
            redundancy_per_head = results_all_phid[metric]['rtr'][normalized]['head_averages']
            gradient_ranks_array = compute_gradient_ranks(synergy_per_head, redundancy_per_head)

            if per_layer:
                # Reshape synergy and redundancy to be [layer, head, 1]
                synergy_per_head = synergy_per_head.reshape(constants.NUM_LAYERS, constants.NUM_HEADS_PER_LAYER)
                redundancy_per_head = redundancy_per_head.reshape(constants.NUM_LAYERS, constants.NUM_HEADS_PER_LAYER)
                gradient_ranks_array = gradient_ranks_array.reshape(constants.NUM_LAYERS, constants.NUM_HEADS_PER_LAYER)

                # Compute synergy and redundancy per layer
                synergy_per_head = synergy_per_head.mean(axis=1)
                redundancy_per_head = redundancy_per_head.mean(axis=1)
                gradient_ranks_array = gradient_ranks_array.mean(axis=1)


            # Compute average activation vector accross all cognitive tasks
            activation_vector = {}
            if not per_layer:
                average_activation_vector = np.zeros(constants.NUM_LAYERS * constants.NUM_HEADS_PER_LAYER)
            else:
                average_activation_vector = np.zeros(constants.NUM_LAYERS)
            for cognitive_task in constants.PROMPT_CATEGORIES:
                activation_vector[cognitive_task] = summary_stats_prompts[metric][cognitive_task][:, :, 0]
                if not per_layer:
                    activation_vector[cognitive_task] = activation_vector[cognitive_task].flatten()
                else: 
                    activation_vector[cognitive_task] = activation_vector[cognitive_task].mean(axis=1)

                average_activation_vector += activation_vector[cognitive_task]
            average_activation_vector /= len(constants.PROMPT_CATEGORIES)

            correlation_synergy, correlation_redundancy, correlation_gradient_rank = {}, {}, {}
            for cognitive_task in constants.PROMPT_CATEGORIES:
                
                activation_vector[cognitive_task] = activation_vector[cognitive_task] - average_activation_vector

                # Compute correlation between average activation and synergy and redundancy 
                correlation_synergy[cognitive_task] = np.corrcoef(activation_vector[cognitive_task], synergy_per_head)[0, 1]
                correlation_redundancy[cognitive_task] = np.corrcoef(activation_vector[cognitive_task], redundancy_per_head)[0, 1]
                correlation_gradient_rank[cognitive_task] = np.corrcoef(activation_vector[cognitive_task], gradient_ranks_array)[0, 1]

                plt.figure(figsize=(10, 6))
                plt.scatter(activation_vector[cognitive_task], gradient_ranks_array, c='b', alpha=0.5)
                plt.xlabel('Average Activation')
                plt.ylabel('Gradient Rank')
                plt.title('Gradient Rank vs Average Activation')

                plt.plot([], [], ' ', label='Gradient Rank Correlation: %.2f' % correlation_gradient_rank[cognitive_task])
                plt.plot([], [], ' ', label='Synergy Correlation: %.2f' % correlation_synergy[cognitive_task])
                plt.plot([], [], ' ', label='Redundancy Correlation: %.2f' % correlation_redundancy[cognitive_task])
                plt.legend(loc='upper right')

                plt.grid()

                if save:
                    os.makedirs(base_save_path, exist_ok=True)
                    plt.savefig(base_save_path + cognitive_task + '.png')
                else:
                    plt.show()
                plt.close()

            # Plot of the synergy, redundancy and gradient rank correlation with average activation accrross all cognitive tasks   
            plt.figure(figsize=(12, 8))
            x = np.arange(len(constants.PROMPT_CATEGORIES))
            bar_width = 0.2
            plt.bar(x, list(correlation_synergy.values()), bar_width, label='Synergy')
            plt.bar(x + bar_width, list(correlation_redundancy.values()), bar_width, label='Redundancy')
            plt.bar(x + 2 * bar_width, list(correlation_gradient_rank.values()), bar_width, label='Gradient Rank')
            plt.xticks(x + bar_width, constants.PROMPT_CATEGORIES, rotation=10)
            plt.ylabel('Correlation')
            plt.title('Correlation with Average Activation')
            plt.legend(loc='upper right')
            if save:
                os.makedirs(base_save_path, exist_ok=True)
                plt.savefig(base_save_path + 'correlations.png')
            else:
                plt.show()
            plt.close()