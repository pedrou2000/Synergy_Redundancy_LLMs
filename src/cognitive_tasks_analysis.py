from time_series_generation import *

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pickle

def generate_and_analyze_prompts(prompts, model, tokenizer, device, num_tokens_to_generate=128, aggregation_type='norm', temperature=0.7):
    # List to hold averaged attention weights for each prompt
    averaged_attention_weights_per_prompt = {}
    generated_texts = []
    for attention_measure in constants.METRICS_TRANSFORMER:
        averaged_attention_weights_per_prompt[attention_measure] = []

    for prompt in prompts:
        # Generate text and attention parameters for each prompt
        generated_text, attention_params = generate_text_with_attention(model, tokenizer, num_tokens_to_generate, 
                    device, prompt, temperature=temperature, modified_output_attentions=constants.MODIFIED_OUTPUT_ATTENTIONS)
        generated_texts.append(generated_text)

        # Compute attention metrics norms for each generated text
        time_series = compute_attention_metrics_norms(attention_params, constants.METRICS_TRANSFORMER, num_tokens_to_generate, aggregation_type=aggregation_type)

        for attention_measure in constants.METRICS_TRANSFORMER:
            # Average the attention weights across all timesteps for each prompt
            avg_attention_weights = torch.mean(time_series[attention_measure], axis=2)
            # Collect averaged attention weights
            averaged_attention_weights_per_prompt[attention_measure].append(avg_attention_weights)

    category_attention_weights_tensor, final_tensor = {}, {}
    for attention_measure in constants.METRICS_TRANSFORMER:
        # Stack the averaged attention weights for each prompt
        category_attention_weights_tensor[attention_measure] = torch.tensor(np.array(averaged_attention_weights_per_prompt[attention_measure]))
        # Since we want [constants.NUM_LAYERS, constants.NUM_HEADS_PER_LAYER, num_prompts], we transpose the axes
        final_tensor[attention_measure] = category_attention_weights_tensor[attention_measure].transpose(0, 1).transpose(1, 2)
    return final_tensor, generated_texts


def generate_and_analyze_rest(n_prompts, model, tokenizer, device, num_tokens_to_generate=128, aggregation_type='norm', 
                              random_input_length=10, temperature=2):
    # List to hold averaged attention weights for each prompt
    averaged_attention_weights_per_prompt = {}
    generated_texts = []
    for attention_measure in constants.METRICS_TRANSFORMER:
        averaged_attention_weights_per_prompt[attention_measure] = []

    for i in range(n_prompts):
        print(f"Generating resting state {i+1}/{n_prompts}")
        generated_text, attention_params = simulate_resting_state_attention(model, tokenizer, num_tokens_to_generate, device, temperature=temperature, random_input_length=random_input_length)
        generated_texts.append(generated_text)

        time_series = compute_attention_metrics_norms(attention_params, constants.METRICS_TRANSFORMER, num_tokens_to_generate, aggregation_type=aggregation_type)

        for attention_measure in constants.METRICS_TRANSFORMER:
            # Average the attention weights across all timesteps for each prompt
            avg_attention_weights = torch.mean(time_series[attention_measure], axis=2)
            # Move the averaged attention weights to the CPU
            avg_attention_weights = avg_attention_weights.cpu()
            # Collect averaged attention weights
            averaged_attention_weights_per_prompt[attention_measure].append(avg_attention_weights)
    category_attention_weights_tensor, final_tensor = {}, {}
    for attention_measure in constants.METRICS_TRANSFORMER:
        # Stack the averaged attention weights for each prompt
        category_attention_weights_tensor[attention_measure] = torch.tensor(np.array(averaged_attention_weights_per_prompt[attention_measure]))
        # Since we want [num_layers, num_heads, num_prompts], we transpose the axes
        final_tensor[attention_measure] = category_attention_weights_tensor[attention_measure].transpose(0, 1).transpose(1, 2)
    return final_tensor, generated_texts

def solve_prompts(prompts_dict, model, tokenizer, device, num_tokens_to_generate=128,temperature=0.7, resting_state="resting_state"):
    # Generate and analyze prompts for each category only once

    all_attention_weights, generated_text = {}, {}
    for metric in constants.METRICS_TRANSFORMER:
        all_attention_weights[metric] = {}

    for category in list(prompts_dict.keys()):
        print(f"Analyzing category: {category}")
        if category == resting_state:
            final_tensor, generated_texts = generate_and_analyze_rest(len(constants.PROMPTS[constants.PROMPT_CATEGORIES[0]]), model, tokenizer, device, num_tokens_to_generate, 
                                                                temperature=temperature)
        else:
            final_tensor, generated_texts = generate_and_analyze_prompts(prompts_dict[category], model, tokenizer, device, num_tokens_to_generate, 
                                                            temperature=temperature)
        for metric in constants.METRICS_TRANSFORMER:
            all_attention_weights[metric][category] = final_tensor[metric]
        generated_text[category] = generated_texts

    return all_attention_weights, generated_text

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
                base_plot_path = constants.PLOTS_ACTIVATION_HEATMAPS 
            
            plot_path = f"{base_plot_path}.png"
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()  # Close the plot to prevent it from displaying
        else:
            plt.show()

    return average_attention_transposed

def plot_attention_weights_comparison(all_attention_weights, save=True, base_plot_path=None, layer_indices=None):
    stats_dict = {}
    
    for metric in constants.METRICS_TRANSFORMER:
        # Initialize the dictionary to store means and stds for all layers
        stats_dict[metric] = {category: np.zeros((constants.NUM_LAYERS, constants.NUM_HEADS_PER_LAYER, 2)) for category in constants.PROMPT_CATEGORIES}

        # Compute stats for all layers
        for layer_idx in range(constants.NUM_LAYERS):
            for category in constants.PROMPT_CATEGORIES:
                attention_weights = all_attention_weights[metric][category]
                layer_attention = np.array(attention_weights[layer_idx])
                means = np.mean(layer_attention, axis=1).flatten()
                stds = np.std(layer_attention, axis=1).flatten()
                stats_dict[metric][category][layer_idx, :, 0] = means
                stats_dict[metric][category][layer_idx, :, 1] = stds

        if layer_indices:
            layer_start, layer_end = layer_indices
            layer_start = max(layer_start - 1, 0)
            layer_end = min(layer_end, constants.NUM_LAYERS)
        else:
            layer_start, layer_end = 0, constants.NUM_LAYERS

        # plt.figure()
        plt.figure(figsize=(20, max(2, (layer_end - layer_start) * 3 *(1/0.8))))
        colors = ['b', 'y', 'g', 'r', 'c', 'm', 'k']
        markers = ['o', '^', 's', 'p', '*', '+', 'x']

        # Collect handles and labels for a global legend
        handles, labels = [], []

        for layer_idx in range(layer_start, layer_end):
            ax = plt.subplot(layer_end - layer_start, 1, layer_idx - layer_start + 1)
            
            for cat_idx, category in enumerate(constants.PROMPT_CATEGORIES):
                means = stats_dict[metric][category][layer_idx, :, 0]
                stds = stats_dict[metric][category][layer_idx, :, 1] if not np.isnan(stats_dict[metric][category][layer_idx, 0, 1]) else None
                label = f'{category}' if stds is not None else category
                if stds is not None:
                    n = len(all_attention_weights[metric][category][layer_idx][0])
                    se = stds / np.sqrt(n)
                    t_critical = stats.t.ppf(1 - 0.025, n - 1)
                    margins_of_error = t_critical * se
                    positions = np.arange(constants.NUM_HEADS_PER_LAYER) + cat_idx * 0.1 - (len(constants.PROMPTS) - 1) * 0.05
                    line, caplines, errorlinecols = plt.errorbar(positions, means, yerr=margins_of_error, fmt=markers[cat_idx % len(markers)], 
                                                                ecolor=colors[cat_idx % len(colors)], capthick=2, capsize=5, 
                                                                label=label, linestyle='None')
                else:
                    line, = plt.plot(np.arange(constants.NUM_HEADS_PER_LAYER), means, marker=markers[cat_idx % len(markers)], 
                                    linestyle='-', color=colors[cat_idx % len(colors)], label=label)
                if layer_idx == layer_start:
                    handles.append(line)
                    labels.append(label)

            plt.title(f'Layer {layer_idx + 1}')
            if layer_idx == layer_end - 1:
                plt.xlabel('Head Index')
            plt.ylabel('Attention Weights')
            plt.xticks(np.arange(constants.NUM_HEADS_PER_LAYER))


        # Place the super title above everything, adjusting spacing as needed
        plt.suptitle("Average Attention Weights Norm Comparison by Prompt Category", y=0.98)  # Increased font size here

        # Add the figure-wide legend at the top, but below the super title
        plt.figlegend(handles, labels, loc='upper center', ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.97), fontsize='x-large')


        # Adjust layout to prevent overlap and make sure everything fits
        plt.subplots_adjust(top=0.915)  # You might need to adjust this value based on your specific plot configuration
        # plt.tight_layout(rect=[0, 0.03, 1, 0.98])




        if save:
            if not base_plot_path:
                base_plot_path = constants.PLOTS_HEAD_ACTIVATIONS_COMPARISON 
            plot_path = f"{base_plot_path}{metric}/2-Head_Activation_Comparison.png"
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()  # Close the plot to prevent it from displaying
        else:
            plt.show()

    return stats_dict

def plot_categories_comparison(all_attention_weights, save=False, base_plot_path=None, split_half=False, split_third=False):
    # Plot category comparison
    
    if base_plot_path is None:
        base_plot_path = constants.PLOTS_HEAD_ACTIVATIONS_COGNITIVE_TASKS 
        if save:
            os.makedirs(base_plot_path, exist_ok=True)

    if split_half:
        constants.NUM_LAYERS = len(all_attention_weights[list(all_attention_weights.keys())[0]])
        half_layers = constants.NUM_LAYERS // 2 # all_attention_weights, save=True, base_plot_path=None, layer_indices=None, filename='head_activations_comparison
        stats_dict = plot_attention_weights_comparison(all_attention_weights, save=save, base_plot_path=base_plot_path, 
                        layer_indices=(1,half_layers), filename='head_activations_comparison_1')
        stats_dict = plot_attention_weights_comparison(all_attention_weights, save=save, base_plot_path=base_plot_path, 
            layer_indices=(half_layers+1, constants.NUM_LAYERS), filename='head_activations_comparison_2')
    elif split_third:
        constants.NUM_LAYERS = len(all_attention_weights[list(all_attention_weights.keys())[0]])
        third_layers = constants.NUM_LAYERS // 3
        stats_dict = plot_attention_weights_comparison(all_attention_weights, save=save, base_plot_path=base_plot_path, 
                        layer_indices=(1,third_layers), filename='head_activations_comparison_1')
        stats_dict = plot_attention_weights_comparison(all_attention_weights, save=save, base_plot_path=base_plot_path, 
            layer_indices=(third_layers+1, 2*third_layers), filename='head_activations_comparison_2')
        stats_dict = plot_attention_weights_comparison(all_attention_weights, save=save, base_plot_path=base_plot_path, 
            layer_indices=(2*third_layers + 1, constants.NUM_LAYERS), filename='head_activations_comparison_3')
    else:
        stats_dict = plot_attention_weights_comparison(all_attention_weights, save=save, base_plot_path=base_plot_path)
    
    return stats_dict

def plot_all_heatmaps(all_attention_weights, save=False, base_plot_path=None):
    # Plot individual heatmaps for each category
    for metric in all_attention_weights.keys():

        if base_plot_path is None:
            base_plot_path = constants.PLOTS_HEAD_ACTIVATIONS_COGNITIVE_TASKS 
            if save:
                os.makedirs(base_plot_path, exist_ok=True)
        i = 3
        for category, attention_weights in all_attention_weights[metric].items():
            compute_and_plot_attention_heatmap(attention_weights, plot_heatmap=True, save=save, base_plot_path=base_plot_path + f"{metric}/1-Heatmaps/{i}-{category}")
            i+=1

def save_attention_weights(attention_weights_prompts, generated_text, base_save_path=None, merge=False):
    if not base_save_path:
        # Assuming 'constants.MATRICES_DIR' is defined and is a valid path
        base_save_path = constants.ATTENTION_WEIGHTS_DIR 
    
    # Extract directory from base_save_path
    dir_path = os.path.dirname(base_save_path)
    
    # Create the directory if it does not exist
    os.makedirs(dir_path, exist_ok=True)
    
    if merge:
        attention_weights_prompts_old = load_attention_weights()
        for metric in constants.METRICS_TRANSFORMER:
            for category in attention_weights_prompts[metric].keys():
                attention_weights_prompts_old[metric][category] = attention_weights_prompts[metric][category]
        attention_weights_prompts = attention_weights_prompts_old
    
    # Now save the file
    with open(base_save_path + 'attention_params.pkl', 'wb') as file:
        pickle.dump(attention_weights_prompts, file)
    # Save generated text as a txt file in the same directory
    for category, answers in generated_text.items():
        with open(base_save_path + 'answers-' + category + '.txt', 'w') as file:
            if category == "resting_state":
                prompts = ["Resting state"*len(answers)]
            else:
                prompts = constants.PROMPTS[category]
            
            for prompt, answer in zip(prompts, answers):
                file.write(f"--- Prompt ---\n{prompt}\n")
                file.write(f"--- Answer ---\n{answer}\n"+ "-"*50 + "\n")
 


def load_attention_weights(n=0, base_plot_path=None):
    if not base_plot_path:
        file_attention_weights = constants.ATTENTION_WEIGHTS_DIR + 'attention_params.pkl'
    else:
        file_attention_weights = base_plot_path + 'attention_params.pkl'

    with open(file_attention_weights, 'rb') as file:
        attention_weights_prompts = pickle.load(file)
    return attention_weights_prompts