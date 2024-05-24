from time_series_generation import *

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr

def generate_and_analyze_prompts(prompts, model, tokenizer, device, num_tokens_to_generate=128, aggregation_type='norm', 
                                 attention_measure="attention_weights", temperature=0.7):
    # List to hold averaged attention weights for each prompt
    averaged_attention_weights_per_prompt = []

    for prompt in prompts:
        # Generate text and attention parameters for each prompt
        generated_text, attention_params = generate_text_with_attention(model, tokenizer, num_tokens_to_generate, 
                    device, prompt, temperature=temperature, modified_output_attentions=constants.MODIFIED_OUTPUT_ATTENTIONS)
        
        # Compute attention metrics norms for each generated text
        time_series = compute_attention_metrics_norms(attention_params, constants.METRICS_TRANSFORMER, num_tokens_to_generate, aggregation_type=aggregation_type)

        # Average the attention weights across all timesteps for each prompt
        avg_attention_weights = torch.mean(time_series[attention_measure], axis=2)
        # Collect averaged attention weights
        averaged_attention_weights_per_prompt.append(avg_attention_weights)

    category_attention_weights_tensor = torch.tensor(np.array(averaged_attention_weights_per_prompt))
    
    # Since we want [num_layers, num_heads, num_prompts], we transpose the axes
    final_tensor = category_attention_weights_tensor.transpose(0, 1).transpose(1, 2)
    return final_tensor

def generate_and_analyze_rest(n_prompts, model, tokenizer, device, num_tokens_to_generate=128, aggregation_type='norm', 
                              random_input_length=10, temperature=2, attention_measure="attention_weights"):
    # List to hold averaged attention weights for each prompt
    averaged_attention_weights_per_prompt = []
    all_time_series = []

    for i in range(n_prompts):
        generated_text, attention_params = simulate_resting_state_attention(model, tokenizer, num_tokens_to_generate, device, temperature=temperature, random_input_length=random_input_length)
        time_series = compute_attention_metrics_norms(attention_params, constants.METRICS_TRANSFORMER, num_tokens_to_generate, aggregation_type=aggregation_type)
        all_time_series.append(time_series)

        # Average the attention weights across all timesteps for each prompt
        avg_attention_weights = torch.mean(time_series[attention_measure], axis=2)
        # Collect averaged attention weights
        averaged_attention_weights_per_prompt.append(avg_attention_weights)

    category_attention_weights_tensor = torch.tensor(np.array(averaged_attention_weights_per_prompt))
    
    # Since we want [num_layers, num_heads, num_prompts], we transpose the axes
    final_tensor = category_attention_weights_tensor.transpose(0, 1).transpose(1, 2)
    return final_tensor, time_series

def average_rest_attention_weights(time_series, attention_measure="attention_weights"):
    # List to hold averaged attention weights for each prompt
    averaged_attention_weights_per_prompt = []

    for time_serie in time_series:
        # Average the attention weights across all timesteps for each prompt
        avg_attention_weights = torch.mean(time_serie[attention_measure], axis=2)
        # Collect averaged attention weights
        averaged_attention_weights_per_prompt.append(avg_attention_weights)

    category_attention_weights_tensor = torch.tensor(np.array(averaged_attention_weights_per_prompt))
    
    # Since we want [num_layers, num_heads, num_prompts], we transpose the axes
    category_attention_weights_tensor = category_attention_weights_tensor.transpose(0, 1).transpose(1, 2)
    return category_attention_weights_tensor

def compute_and_plot_attention_heatmap(time_series_attention_weights, plot_heatmap=True, save=True, base_plot_path=None):
    # Convert to numpy array for easier manipulation
    time_series_np = np.array(time_series_attention_weights)

    # Compute the average across time (last dimension)
    average_attention = np.mean(time_series_np, axis=2)

    if plot_heatmap:
        plt.figure(figsize=(12, 5))
        # Transpose average_attention for plotting (swap layers and heads)
        average_attention_transposed = average_attention.T
        ax = sns.heatmap(average_attention_transposed, annot=True, fmt=".2f", cmap="coolwarm", cbar=True,
                         linewidths=0.5, linecolor='gray', cbar_kws={"shrink": 0.8, "label": 'Average Attention Weight Norm'})
        ax.set_xticks(np.arange(average_attention.shape[0]) + 0.5)
        ax.set_xticklabels([f"{i+1}" for i in range(average_attention.shape[0])], rotation=45, ha="right")  # Layers
        ax.set_yticks(np.arange(average_attention.shape[1]) + 0.5)
        ax.set_yticklabels([f"{i+1}" for i in range(average_attention.shape[1])], rotation=0)  # Heads

        plt.title("Average Attention Weights Norm Across Time", pad=20)
        plt.xlabel("Layer", labelpad=10)
        plt.ylabel("Head", labelpad=10)
        plt.tight_layout()
        if save:
            if not base_plot_path:
                base_plot_path = constants.PLOTS_ACTIVATION_HEATMAPS + datetime.now().strftime("%Y%m%d_%H%M%S") + '/'
            
            plot_path = f"{base_plot_path}.png"
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path, bbox_inches='tight')
        else:
            plt.show()

    return average_attention_transposed

def plot_attention_weights_comparison(all_attention_weights, categories, save=True, base_plot_path=None, 
                                      synergy_redundancy_heads_averages=None, attention_measure="attention_weights", 
                                      layer_indices=None, filename='head_activations_comparison'):
    plt.rcParams.update({'font.size': 12})  # Adjust the 14 to larger sizes as needed
    num_layers = len(all_attention_weights[categories[0]])
    num_heads = len(all_attention_weights[categories[0]][0])
    num_categories = len(categories)
    
    # Initialize the dictionary to store means and stds for all layers
    stats_dict = {category: np.zeros((num_layers, num_heads, 2)) for category in categories}
    if synergy_redundancy_heads_averages:
        stats_dict.update({k: np.zeros((num_layers, num_heads, 2)) for k in ['synergy', 'redundancy']})

    # Compute stats for all layers
    for layer_idx in range(num_layers):
        for category in categories:
            attention_weights = all_attention_weights[category]
            layer_attention = np.array(attention_weights[layer_idx])
            means = np.mean(layer_attention, axis=1).flatten()
            stds = np.std(layer_attention, axis=1).flatten()
            stats_dict[category][layer_idx, :, 0] = means
            stats_dict[category][layer_idx, :, 1] = stds
        if synergy_redundancy_heads_averages:
            for category in ['synergy', 'redundancy']:
                means = synergy_redundancy_heads_averages[attention_measure][category][layer_idx*num_heads:(layer_idx+1)*num_heads]
                stats_dict[category][layer_idx, :, 0] = means
                stats_dict[category][layer_idx, :, 1] = np.nan

    if layer_indices:
        layer_start, layer_end = layer_indices
        layer_start = max(layer_start - 1, 0)
        layer_end = min(layer_end, num_layers)
    else:
        layer_start, layer_end = 0, num_layers

    # plt.figure()
    plt.figure(figsize=(20, max(2, (layer_end - layer_start) * 3 *(1/0.8))))
    colors = ['b', 'y', 'g', 'r', 'c', 'm', 'k']
    markers = ['o', '^', 's', 'p', '*', '+', 'x']

    # Collect handles and labels for a global legend
    handles, labels = [], []

    for layer_idx in range(layer_start, layer_end):
        ax = plt.subplot(layer_end - layer_start, 1, layer_idx - layer_start + 1)
        
        for cat_idx, category in enumerate(categories + ['synergy', 'redundancy'] if synergy_redundancy_heads_averages else categories):
            means = stats_dict[category][layer_idx, :, 0]
            stds = stats_dict[category][layer_idx, :, 1] if not np.isnan(stats_dict[category][layer_idx, 0, 1]) else None
            label = f'{category}' if stds is not None else category
            if stds is not None:
                n = len(all_attention_weights[category][layer_idx][0])
                se = stds / np.sqrt(n)
                t_critical = stats.t.ppf(1 - 0.025, n - 1)
                margins_of_error = t_critical * se
                positions = np.arange(num_heads) + cat_idx * 0.1 - (num_categories - 1) * 0.05
                line, caplines, errorlinecols = plt.errorbar(positions, means, yerr=margins_of_error, fmt=markers[cat_idx % len(markers)], 
                                                            ecolor=colors[cat_idx % len(colors)], capthick=2, capsize=5, 
                                                            label=label, linestyle='None')
            else:
                line, = plt.plot(np.arange(num_heads), means, marker=markers[cat_idx % len(markers)], 
                                 linestyle='-', color=colors[cat_idx % len(colors)], label=label)
            if layer_idx == layer_start:
                handles.append(line)
                labels.append(label)

        plt.title(f'Layer {layer_idx + 1}')
        if layer_idx == layer_end - 1:
            plt.xlabel('Head Index', fontsize=18)
        plt.ylabel('Attention Weights', fontsize=18)
        plt.xticks(np.arange(num_heads))


    # Place the super title above everything, adjusting spacing as needed
    plt.suptitle("Average Attention Weights Norm Comparison by Prompt Category", fontsize=20, y=0.98)  # Increased font size here

    # Add the figure-wide legend at the top, but below the super title
    plt.figlegend(handles, labels, loc='upper center', ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.97), fontsize='x-large')


    # Adjust layout to prevent overlap and make sure everything fits
    plt.subplots_adjust(top=0.915)  # You might need to adjust this value based on your specific plot configuration
    # plt.tight_layout(rect=[0, 0.03, 1, 0.98])




    if save:
        if not base_plot_path:
            base_plot_path = constants.PLOTS_HEAD_ACTIVATIONS_COMPARISON + datetime.now().strftime("%Y%m%d_%H%M%S") + '/'
        plot_path = f"{base_plot_path}{filename}.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, bbox_inches='tight')
    else:
        plt.show()

    return stats_dict

def calculate_weighted_synergy_redundancy(all_attention_weights, synergy_redundancy_heads_averages, categories, attention_measure="attention_weights"):
    """
    Calculate weighted synergy and redundancy scores for each category.
    """
    weighted_scores = {category: {'synergy': 0, 'redundancy': 0} for category in categories}
    num_layers = len(all_attention_weights[categories[0]])
    num_heads = len(all_attention_weights[categories[0]][0])

    for category in categories:
        for layer_idx in range(num_layers):
            layer_attention = np.array(all_attention_weights[category][layer_idx])
            means_attention = np.mean(layer_attention, axis=1).flatten()

            # Retrieve synergy and redundancy means for the current layer
            synergy_means = synergy_redundancy_heads_averages[attention_measure]["synergy"][layer_idx*num_heads:(layer_idx+1)*num_heads]
            redundancy_means = synergy_redundancy_heads_averages[attention_measure]["redundancy"][layer_idx*num_heads:(layer_idx+1)*num_heads]

            # Weighted sum of synergy and redundancy for this layer
            weighted_synergy = np.sum(means_attention * synergy_means)
            weighted_redundancy = np.sum(means_attention * redundancy_means)

            # Aggregate across layers
            weighted_scores[category]['synergy'] += weighted_synergy
            weighted_scores[category]['redundancy'] += weighted_redundancy

    return weighted_scores

def plot_weighted_scores_line(weighted_scores, categories, save=True, base_plot_path=None):
    synergy_scores = [weighted_scores[category]['synergy'] for category in categories]
    redundancy_scores = [weighted_scores[category]['redundancy'] for category in categories]
    x = np.arange(len(categories))  # the x locations for the groups

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Synergy plot
    axs[0].plot(x, synergy_scores, 'o-', color='blue', label='Synergy')
    axs[0].set_title('Weighted Synergy Scores by Category')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(categories)
    axs[0].set_ylabel('Synergy Scores')
    axs[0].legend()

    # Redundancy plot
    axs[1].plot(x, redundancy_scores, 's-', color='red', label='Redundancy')
    axs[1].set_title('Weighted Redundancy Scores by Category')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(categories)
    axs[1].set_ylabel('Redundancy Scores')
    axs[1].legend()

    fig.tight_layout()

    if save:
        if not base_plot_path:
            base_plot_path = constants.PLOTS_HEAD_ACTIVATIONS_COMPARISON + datetime.now().strftime("%Y%m%d_%H%M%S") + '/'
        
        plot_path = f"{base_plot_path}weighted_scores_line_comparison.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, bbox_inches='tight')
    else:
        plt.show()

def plot_and_save_attention_analysis(prompts_dict, model, tokenizer, device, num_tokens_to_generate=128, save=True, 
                                     synergy_redundancy_heads_averages=None, rest_time_series=None, generate_rest=False, 
                                     random_input_length=10, temperature_prompts=0.7, temperature_rest=2, attention_measure="attention_weights", split_half=False, split_third=False,
                                     all_attention_weights=None):
    
    # plt.rcParams.update({'font.size': 15})  # Adjust the 14 to larger sizes as needed
        
    # Create a base directory path with timestamp
    base_plot_path = constants.PLOTS_HEAD_ACTIVATIONS_ANALYSIS + datetime.now().strftime("%Y%m%d_%H%M%S") + '/'
    if save:
        os.makedirs(base_plot_path, exist_ok=True)
        
    if all_attention_weights is None: 
        all_attention_weights = {}
        time_series = None

        # Generate and analyze prompts for each category only once
        for category in list(prompts_dict.keys()):
            print(f"Analyzing category: {category}")
            attention_weights = generate_and_analyze_prompts(prompts_dict[category], model, tokenizer, device, num_tokens_to_generate, 
                                                             attention_measure=attention_measure, temperature=temperature_prompts)
            all_attention_weights[category] = attention_weights
            n_prompts = attention_weights.shape[2]

        if rest_time_series is not None:
            print(f"Analyzing category: rest")
            all_attention_weights['rest_loaded'] = average_rest_attention_weights(rest_time_series, attention_measure=attention_measure)
        if generate_rest:
            print("Analyzing and Generating category: rest")

            if isinstance(temperature_rest, list) and len(temperature_rest) > 1:
                for temp in temperature_rest:
                    key = f'rest_temp_{temp}'
                    all_attention_weights[key], time_series = generate_and_analyze_rest(n_prompts, model, tokenizer, device, num_tokens_to_generate, 
                                    random_input_length=random_input_length, temperature=temp, attention_measure=attention_measure)   
            else:
                all_attention_weights['rest'], time_series = generate_and_analyze_rest(n_prompts, model, tokenizer, device, num_tokens_to_generate, 
                                random_input_length=random_input_length, temperature=temperature_rest, attention_measure=attention_measure)


    # Plot category comparison using the collected attention weights
    if split_half:
        num_layers = len(all_attention_weights[list(all_attention_weights.keys())[0]])
        half_layers = num_layers // 2
        stats_dict = plot_attention_weights_comparison(all_attention_weights, list(all_attention_weights.keys()), save=save, base_plot_path=base_plot_path, 
                        synergy_redundancy_heads_averages=synergy_redundancy_heads_averages, layer_indices=(1,half_layers), filename='head_activations_comparison_1')
        stats_dict = plot_attention_weights_comparison(all_attention_weights, list(all_attention_weights.keys()), save=save, base_plot_path=base_plot_path, 
            synergy_redundancy_heads_averages=synergy_redundancy_heads_averages, layer_indices=(half_layers+1, num_layers), filename='head_activations_comparison_2')
    elif split_third:
        num_layers = len(all_attention_weights[list(all_attention_weights.keys())[0]])
        third_layers = num_layers // 3
        stats_dict = plot_attention_weights_comparison(all_attention_weights, list(all_attention_weights.keys()), save=save, base_plot_path=base_plot_path, 
                        synergy_redundancy_heads_averages=synergy_redundancy_heads_averages, layer_indices=(1,third_layers), filename='head_activations_comparison_1')
        stats_dict = plot_attention_weights_comparison(all_attention_weights, list(all_attention_weights.keys()), save=save, base_plot_path=base_plot_path, 
            synergy_redundancy_heads_averages=synergy_redundancy_heads_averages, layer_indices=(third_layers+1, 2*third_layers), filename='head_activations_comparison_2')
        stats_dict = plot_attention_weights_comparison(all_attention_weights, list(all_attention_weights.keys()), save=save, base_plot_path=base_plot_path, 
            synergy_redundancy_heads_averages=synergy_redundancy_heads_averages, layer_indices=(2*third_layers + 1, num_layers), filename='head_activations_comparison_3')
    else:
        stats_dict = plot_attention_weights_comparison(all_attention_weights, list(all_attention_weights.keys()), save=save, base_plot_path=base_plot_path, 
                                                    synergy_redundancy_heads_averages=synergy_redundancy_heads_averages)
    weighted_scores = None
    if synergy_redundancy_heads_averages:
        weighted_scores = calculate_weighted_synergy_redundancy(all_attention_weights, synergy_redundancy_heads_averages, list(all_attention_weights.keys()), attention_measure=attention_measure)
        plot_weighted_scores_line(weighted_scores, list(all_attention_weights.keys()), save=save, base_plot_path=base_plot_path)

    # Plot individual heatmaps for each category
    for category, attention_weights in all_attention_weights.items():
        print(f"{category.capitalize()} Average Attention Heatmap:")
        compute_and_plot_attention_heatmap(attention_weights, plot_heatmap=True, save=save, base_plot_path=base_plot_path + f"{category}_heatmap")

    return weighted_scores, stats_dict, all_attention_weights







