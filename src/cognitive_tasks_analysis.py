from time_series_generation import *

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def generate_and_analyze_prompts(prompts, model, tokenizer, device, num_tokens_to_generate=128, aggregation_type='norm'):
    # List to hold averaged attention weights for each prompt
    averaged_attention_weights_per_prompt = []

    for prompt in prompts:
        # Generate text and attention parameters for each prompt
        generated_text, attention_params = generate_text_with_attention(model, tokenizer, num_tokens_to_generate, device, prompt, temperature=1)
        
        # Compute attention metrics norms for each generated text
        time_series = compute_attention_metrics_norms(attention_params, constants.METRICS_TRANSFORMER, num_tokens_to_generate, aggregation_type=aggregation_type)

        # Average the attention weights across all timesteps for each prompt
        avg_attention_weights = torch.mean(time_series["attention_weights"], axis=2)
        # Collect averaged attention weights
        averaged_attention_weights_per_prompt.append(avg_attention_weights)

    category_attention_weights_tensor = torch.tensor(np.array(averaged_attention_weights_per_prompt))
    
    # Since we want [num_layers, num_heads, num_prompts], we transpose the axes
    final_tensor = category_attention_weights_tensor.transpose(0, 1).transpose(1, 2)
    return final_tensor

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
        ax.set_xticklabels([f"{i+1}" for i in range(average_attention.shape[0])], rotation=45, ha="right")  # Layers
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

def plot_attention_weights_comparison(all_attention_weights, categories, save=True, base_plot_path=None, synergy_redundancy_heads_averages=None):
    num_layers = len(all_attention_weights[categories[0]])
    num_heads = len(all_attention_weights[categories[0]][0])
    num_categories = len(categories)
    
    # Create a large figure to accommodate all layer subplots
    plt.figure(figsize=(15, num_layers * 5))

    # Define colors/markers for different categories for visual distinction
    colors = ['b', 'y', 'g', 'r', 'c', 'm', 'k']
    markers = ['o', '^', 's', 'p', '*', '+', 'x']
    
    for layer_idx in range(num_layers):
        plt.subplot(num_layers, 1, layer_idx + 1)
        
        for cat_idx, category in enumerate(categories + ['synergy', 'redundancy'] if synergy_redundancy_heads_averages else categories):
            if category in ['synergy', 'redundancy']:
                means = synergy_redundancy_heads_averages["attention_weights"][category][layer_idx*num_heads:(layer_idx+1)*num_heads]
                # For synergy and redundancy, we plot without error bars
                plt.plot(np.arange(num_heads), means, marker=markers[cat_idx % len(markers)], 
                         linestyle='-', color=colors[cat_idx % len(colors)], label=f'{category}')
            else:
                # Original processing for attention weights
                attention_weights = all_attention_weights[category]
                layer_attention = np.array(attention_weights[layer_idx])
                means = np.mean(layer_attention, axis=1).flatten()
                stds = np.std(layer_attention, axis=1).flatten()

                n = layer_attention.shape[1]
                se = stds / np.sqrt(n)
                t_critical = stats.t.ppf(1 - 0.025, n - 1)
                margins_of_error = t_critical * se

                positions = np.arange(num_heads) + cat_idx * 0.1 - (num_categories - 1) * 0.05
                plt.errorbar(positions, means, yerr=margins_of_error, fmt=markers[cat_idx % len(markers)], 
                            ecolor=colors[cat_idx % len(colors)], capthick=2, capsize=5, 
                            label=f'{category} (95% CI)', linestyle='None')
        
        plt.title(f'Layer {layer_idx + 1}')
        plt.xlabel('Head Index')
        plt.ylabel('Average Attention Weights Norm')
        plt.xticks(np.arange(num_heads))
        plt.legend()
    
    plt.suptitle("Average Attention Weights Norm Comparison by Prompt Category")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust subplot parameters to fit the title
    if save:
        if not base_plot_path:
            base_plot_path = constants.PLOTS_HEAD_ACTIVATIONS_COMPARISON + datetime.now().strftime("%Y%m%d_%H%M%S") + '/'
        
        plot_path = f"{base_plot_path}head_activations_comparison.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, bbox_inches='tight')
    else:
        plt.show()

def calculate_weighted_synergy_redundancy(all_attention_weights, synergy_redundancy_heads_averages, categories):
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
            synergy_means = synergy_redundancy_heads_averages["attention_weights"]["synergy"][layer_idx*num_heads:(layer_idx+1)*num_heads]
            redundancy_means = synergy_redundancy_heads_averages["attention_weights"]["redundancy"][layer_idx*num_heads:(layer_idx+1)*num_heads]

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

def plot_and_save_attention_analysis(prompts_dict, model, tokenizer, device, num_tokens_to_generate=128, save=True, synergy_redundancy_heads_averages=None):
    categories = list(prompts_dict.keys())
    all_attention_weights = {}
    
    # Create a base directory path with timestamp
    base_plot_path = constants.PLOTS_HEAD_ACTIVATIONS_ANALYSIS + datetime.now().strftime("%Y%m%d_%H%M%S") + '/'
    if save:
        os.makedirs(base_plot_path, exist_ok=True)

    # Generate and analyze prompts for each category only once
    for category in categories:
        print(f"Analyzing category: {category}")
        attention_weights = generate_and_analyze_prompts(prompts_dict[category], model, tokenizer, device, num_tokens_to_generate)
        all_attention_weights[category] = attention_weights

    # Plot category comparison using the collected attention weights
    plot_attention_weights_comparison(all_attention_weights, categories, save=save, base_plot_path=base_plot_path, synergy_redundancy_heads_averages=synergy_redundancy_heads_averages)
    weighted_scores = None
    if synergy_redundancy_heads_averages:
        weighted_scores = calculate_weighted_synergy_redundancy(all_attention_weights, synergy_redundancy_heads_averages, categories)
        plot_weighted_scores_line(weighted_scores, categories, save=save, base_plot_path=base_plot_path)

    # Plot individual heatmaps for each category
    for category, attention_weights in all_attention_weights.items():
        print(f"{category.capitalize()} Average Attention Heatmap:")
        compute_and_plot_attention_heatmap(attention_weights, plot_heatmap=True, save=save, base_plot_path=base_plot_path + f"{category}_heatmap")

    return weighted_scores
