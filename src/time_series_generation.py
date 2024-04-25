from huggingface_hub import login
import torch
import torch.nn.functional as F
import numpy as np
import os 
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GemmaForCausalLM
import constants
import gc  
import psutil  
from datetime import datetime



def sample_with_temperature(logits, temperature=1.0):
    # Sample an index from a logits array based on temperature.
    scaled_logits = logits / temperature
    probabilities = F.softmax(scaled_logits, dim=-1) # Convert logits to probabilities
    sampled_indices = torch.multinomial(probabilities, num_samples=1) # Sample from the probability distribution
    return sampled_indices

def generate_text_with_attention(model, tokenizer, num_tokens_to_generate: int, device: str, prompt=None, input_ids=None, temperature=0.1):
    # Autoregressively generates text from a given prompt while capturing all types of attention weights and other related tensors.

    # Encode the prompt and move to the specified device
    if prompt is not None and input_ids is None:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    elif input_ids is not None and prompt is None:
        input_ids = input_ids.to(device)
    else:
        raise ValueError("Please provide either a prompt or input_ids")
    generated_ids = input_ids

    # Initialize container for all tensors of each generation step
    attention_params = {}

    for t in range(num_tokens_to_generate):
        if t % 5 == 0:  # Monitor memory usage every 5 tokens
            # print(f"Generating token {t+1}/{num_tokens_to_generate}...")
            # print(f"RAM usage after generating {t+1} tokens: {psutil.virtual_memory().percent}%")
            gc.collect()  # Explicitly invoke garbage collection
        # Use torch.no_grad() to disable gradient calculations and reduce memory consumption
        with torch.no_grad():
            outputs = model(generated_ids, output_attentions=True)
        next_token_logits = outputs.logits[:, -1, :]  # Logits for the next token predictions
        next_token_id = sample_with_temperature(next_token_logits, temperature=temperature)

        generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

        # Only save the last timestep for memory efficiency
        if t == num_tokens_to_generate - 1:
            # Process and move attention outputs to CPU
            attentions_on_cpu = [{k: v.detach().to('cpu') for k, v in layer.items()} for layer in outputs.attentions]

            # Dynamically initialize and store all keys from attention outputs
            for idx, layer in enumerate(attentions_on_cpu):
                for key, value in layer.items():
                    if key not in attention_params:
                        attention_params[key] = []
                    attention_params[key].append(value[0]) # Remove the batch dimension

    # Convert time-step dictionaries into tensors where applicable
    for key in attention_params.keys():
        attention_params[key] = torch.stack(attention_params[key])

    # Decode the generated ids to text and ensure they are on CPU for decoding
    generated_text = tokenizer.decode(generated_ids[0].to('cpu'), skip_special_tokens=True)

    return generated_text, attention_params

def generate_random_token_input(length, tokenizer):
    # Generate a random input tensor of a specified length
    random_input_ids_np = np.random.randint(0, tokenizer.vocab_size, (1, length))
    random_input_ids = torch.tensor(random_input_ids_np, dtype=torch.long)
    return random_input_ids

def simulate_resting_state_attention(model, tokenizer, num_tokens_to_generate, device, temperature=3, random_input_length=10):
    # Simulate the resting state of the attention weights by generating text from random input tokens
    random_input_ids = generate_random_token_input(random_input_length, tokenizer).to(device)
    generated_text, attention_params = generate_text_with_attention(model, tokenizer, num_tokens_to_generate, device, input_ids=random_input_ids, temperature=temperature)
    return generated_text, attention_params

def compute_attention_metrics_norms_inefficient(attention_params, selected_metrics, num_tokens_to_generate):
    # Computes the norms of selected attention metrics across all layers and heads for each timestep.
    
    attention_metrics = [attention_params[metric] for metric in selected_metrics]
    num_layers = attention_metrics[0][0].shape[0]
    num_heads_per_layer = attention_metrics[0][0].shape[1]

    # Initialize the time series dictionary for storing the norms
    time_series = {metric: [[[[] for _ in range(num_tokens_to_generate)] for _ in range(num_heads_per_layer)] for _ in range(num_layers)] for metric in selected_metrics}

    for metric_index, selected_metric in enumerate(selected_metrics):
        for t in range(num_tokens_to_generate):
            for layer in range(num_layers):
                for head in range(num_heads_per_layer):
                    # Compute the norm for the specified token, layer, and head
                    query_norm = torch.norm(attention_metrics[metric_index][t][layer, head, -1])
                    # Store the computed norm
                    time_series[selected_metric][layer][head][t].append(query_norm.item())

    return time_series

def calculate_aggregation(vector, aggregation_type):
    if aggregation_type == 'norm':
        return torch.norm(vector)
    elif aggregation_type == 'mean':
        return torch.mean(vector)
    elif aggregation_type == 'entropy':
        # Make sure the vector is normalized and positive for entropy calculation
        vector_normalized = F.softmax(vector, dim=0)
        return -(vector_normalized * torch.log(vector_normalized + 1e-5)).sum()  # Adding epsilon to avoid log(0)
    elif aggregation_type == 'max':
        return torch.max(vector)
    else:
        raise ValueError("Unsupported aggregation type. Choose from 'norm', 'mean', 'entropy'.")


def compute_attention_metrics_norms(attention_params, selected_metrics, num_tokens_to_generate, aggregation_type='norm'):
    # Computes the norms of selected attention metrics across all layers and heads for each timestep.
    # print shape of attention_params
    attention_metrics = [attention_params[metric] for metric in selected_metrics]
    # print shape of attention_metrics
    num_layers, num_heads_per_layer, num_tokens, _ = attention_metrics[0].shape
    num_inputs = num_tokens - num_tokens_to_generate

    # Initialize the time series dictionary for storing the norms with a tensor of shape (num_layers, num_heads_per_layer, num_tokens_to_generate)
    time_series = {metric:torch.zeros((num_layers, num_heads_per_layer, num_tokens_to_generate)) for metric in selected_metrics}
    for metric_index, selected_metric in enumerate(selected_metrics):
        for t in range(num_tokens_to_generate):
            for layer in range(num_layers):
                for head in range(num_heads_per_layer):
                    # Compute the norm for the specified token, layer, and head
                    vector = attention_metrics[metric_index][layer, head, t + num_inputs]
                    # At time t, we only have t + num_inputs tokens, so attention weights are only available up to t + num_inputs
                    if selected_metric =="attention_weights":
                        vector = vector[: t + num_inputs + 1]
                    # print(f"Vector Mean Shape: {vector_mean.shape}, t + inputs: {t + num_inputs}")
                    aggregated_vector = calculate_aggregation(vector, aggregation_type)

                    # Store the computed norm
                    time_series[selected_metric][layer][head][t] = aggregated_vector.item()

    return time_series

def save_time_series(time_series, base_plot_path=None):
    if not base_plot_path:
        base_plot_path = constants.TIME_SERIES_DIR  + datetime.now().strftime("%Y%m%d_%H%M%S") + '.pt'
    torch.save(time_series, base_plot_path)

def load_time_series(time_series_number, base_plot_path=None):
    if not base_plot_path:
        base_plot_path = constants.TIME_SERIES_DIR  + datetime.now().strftime("%Y%m%d_%H%M%S") + '.pt'
    # Sort the time_series files by name and load the time_series_number-th file
    time_series_files = sorted(os.listdir(constants.TIME_SERIES_DIR))
    base_plot_path = constants.TIME_SERIES_DIR + time_series_files[time_series_number]

    return torch.load(base_plot_path)

def smooth_series(series, window_size=3):
    """Smooth the series with a simple moving average. window_size must be odd."""
    if window_size <= 1:
        return series
    # Flatten the series to ensure it's in the correct shape for convolution
    series_flattened = np.array(series).flatten()
    conv_window = np.ones(window_size) / window_size
    series_padded = np.pad(series_flattened, pad_width=(window_size//2, window_size//2), mode='edge')
    smoothed = np.convolve(series_padded, conv_window, mode='valid')
    return smoothed

def plot_attention_metrics_norms_over_time(time_series, metrics, num_heads_plot=8, base_plot_path=None, smoothing_window=1, save=False, layer_range=None):
    if not base_plot_path:
        base_plot_path = constants.PLOTS_TIME_SERIES_DIR + datetime.now().strftime("%Y%m%d_%H%M%S") + '/'

    if layer_range is None:
        layer_start = 0
        layer_end = len(time_series[next(iter(metrics))])  # Use the total number of layers if range is not provided
    else:
        layer_start, layer_end = layer_range
        layer_start = max(layer_start - 1, 0)  # Convert to zero-indexed
        layer_end = min(layer_end, len(time_series[next(iter(metrics))]))  # Ensure layer_end does not exceed number of layers

    for metric in metrics:
        num_layers = layer_end - layer_start
        num_tokens_to_generate = len(time_series[metric][0][0])  # Assuming uniform length across heads

        fig, axs = plt.subplots(nrows=num_layers, ncols=1, figsize=(11, 2 * num_layers), sharex=True)
        title = f'Norm of {metric} Over Time'
        if layer_range:
            title += f', Layers {layer_start+1} to {layer_end}'
        if smoothing_window > 1:
            title += f' (Smoothed with Window Size {smoothing_window})'
        fig.suptitle(title, y=0.99, fontsize='x-large')

        if num_layers == 1:
            axs = [axs]  # Ensure axs is iterable when only one plot

        tick_positions = np.arange(0, num_tokens_to_generate + 1, num_tokens_to_generate // 10)

        # Collect labels and handles for legend from the first subplot
        handles, labels = [], []
        for layer_idx in range(layer_start, layer_end):
            ax = axs[layer_idx - layer_start]
            for head_idx in range(min(num_heads_plot, len(time_series[metric][layer_idx]))):
                norms = [norm for norm in time_series[metric][layer_idx][head_idx]]
                if smoothing_window > 1:
                    norms = smooth_series(np.array(norms), window_size=smoothing_window)
                line, = ax.plot(norms, label=f'Head {head_idx+1}')
                if layer_idx == layer_start:  # Collect legend info only from the first layer
                    handles.append(line)
                    labels.append(f'Head {head_idx + 1}')

            ax.set_title(f'Layer {layer_idx+1}', pad=10)
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Norm')
            ax.set_xticks(tick_positions)

        # Place the legend at the top
        fig.legend(handles, labels, loc='upper center', ncol=min(num_heads_plot, len(time_series[metric][layer_start])), bbox_to_anchor=(0.5, 0.982), frameon=False)

        if layer_range is None:
            plt.tight_layout(rect=[0, 0.03, 1, 0.98])
            plt.subplots_adjust(top=0.965)  # Adjust this to fit both the title and the legend at the top
        else:
            plt.tight_layout(rect=[0, 0.03, 1, 0.98])
            plt.subplots_adjust(top=0.945)

        if save:
            layer_suffix = f'_layers_{layer_start+1}_to_{layer_end}' if layer_range else ''
            plot_path = f"{base_plot_path}{metric}{layer_suffix}_norms_over_time.png"
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path, bbox_inches='tight')
        else:
            plt.show()
        plt.close()



