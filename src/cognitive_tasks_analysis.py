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

def plot_attention_weights_comparison(all_attention_weights, categories, save=True, base_plot_path=None):
    num_layers = len(all_attention_weights[categories[0]])
    num_heads = len(all_attention_weights[categories[0]][0])
    num_categories = len(categories)
    
    # Create a large figure to accommodate all layer subplots
    plt.figure(figsize=(15, num_layers * 5))

    # Define colors/markers for different categories for visual distinction
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    markers = ['o', '^', 's', 'p', '*', '+', 'x']
    
    for layer_idx in range(num_layers):
        plt.subplot(num_layers, 1, layer_idx + 1)
        
        for cat_idx, category in enumerate(categories):
            # Extract attention weights for this category
            attention_weights = all_attention_weights[category]
            layer_attention = np.array(attention_weights[layer_idx])
            means = np.mean(layer_attention, axis=1).flatten()
            stds = np.std(layer_attention, axis=1).flatten()

            # Correct calculation of n as the number of prompts
            n = layer_attention.shape[1]  # Number of prompts is the size of the second dimension
            se = stds / np.sqrt(n)  # Standard error of the mean
            
            # t-critical value for 95% CI, degrees of freedom = n - 1
            t_critical = stats.t.ppf(1 - 0.025, n - 1)
            margins_of_error = t_critical * se  # Margins of error for 95% CI

            # Adjusted plot with error bars for 95% CI
            positions = np.arange(num_heads) + cat_idx * 0.1 - (num_categories - 1) * 0.05
            plt.errorbar(positions, means, yerr=margins_of_error, fmt=markers[cat_idx % len(markers)], 
                        ecolor=colors[cat_idx % len(colors)], capthick=2, capsize=5, 
                        label=f'{category} (95% CI)', linestyle='None')
            
            # # Plot with error bars: adjust positions to not overlap categories
            # positions = np.arange(num_heads) + cat_idx * 0.1 - (num_categories - 1) * 0.05
            # plt.errorbar(positions, means, yerr=stds, fmt=markers[cat_idx % len(markers)], ecolor=colors[cat_idx % len(colors)], 
            #              capthick=2, capsize=5, label=f'{category} (Mean ± SD)', linestyle='None')
        
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

def plot_and_save_attention_analysis(prompts_dict, model, tokenizer, device, num_tokens_to_generate=128, save=True):
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
    plot_attention_weights_comparison(all_attention_weights, categories, save=save, base_plot_path=base_plot_path + "category_comparison.png")

    # Plot individual heatmaps for each category
    for category, attention_weights in all_attention_weights.items():
        print(f"{category.capitalize()} Average Attention Heatmap:")
        compute_and_plot_attention_heatmap(attention_weights, plot_heatmap=True, save=save, base_plot_path=base_plot_path + f"{category}_heatmap")
