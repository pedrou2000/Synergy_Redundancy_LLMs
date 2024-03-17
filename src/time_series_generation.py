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

def generate_text_with_attention(model, tokenizer, num_tokens_to_generate, device, prompt=None, input_ids=None, temperature=0.1):
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

        # Process and move attention outputs to CPU
        attentions_on_cpu = [{k: v.detach().to('cpu') for k, v in layer.items()} for layer in outputs.attentions]

        # Dynamically initialize and store all keys from attention outputs
        for idx, layer in enumerate(attentions_on_cpu):
            for key, value in layer.items():
                if key not in attention_params:
                    attention_params[key] = {}
                if t not in attention_params[key]:
                    attention_params[key][t] = []
                attention_params[key][t].append(value[0]) # Remove the batch dimension

    # Convert time-step dictionaries into tensors where applicable
    for key in attention_params.keys():
        for time_step in attention_params[key]:
            attention_params[key][time_step] = torch.stack(attention_params[key][time_step])

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

def compute_attention_metrics_norms(attention_params, selected_metrics, num_tokens_to_generate):
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

def save_time_series(time_series, base_plot_path=None):
    if not base_plot_path:
        base_plot_path = constants.TIME_SERIES_DIR  + datetime.now().strftime("%Y%m%d_%H%M%S") + '.pt'
    torch.save(time_series, base_plot_path)

def plot_attention_metrics_norms_over_time(time_series, metrics, num_heads_plot=5, base_plot_path=None):
    if not base_plot_path:
        base_plot_path = constants.PLOTS_TIME_SERIES_DIR + datetime.now().strftime("%Y%m%d_%H%M%S") + '/'
    for metric in metrics:
        num_layers = len(time_series[metric])  # Number of layers
        num_tokens_to_generate = len(time_series[metric][0][0])  # Assuming uniform length across heads

        fig, axs = plt.subplots(nrows=num_layers, ncols=1, figsize=(11, 2 * num_layers), sharex=True)
        fig.suptitle(f'Norm of {metric.capitalize()} Over Time')

        if num_layers == 1:
            axs = [axs]

        tick_positions = np.arange(0, num_tokens_to_generate + 1, 5)

        for layer_idx, ax in enumerate(np.atleast_1d(axs)):  # np.atleast_1d ensures axs is iterable
            for head_idx in range(min(num_heads_plot, len(time_series[metric][layer_idx]))):
                norms = [norm for norm in time_series[metric][layer_idx][head_idx]]
                ax.plot(norms, label=f'Head {head_idx}')

            ax.set_title(f'Layer {layer_idx}')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Norm')
            ax.legend(loc='upper right')
            ax.set_xticks(tick_positions)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Ensure the base plot path directory exists
        plot_path = f"{base_plot_path}{metric}_norms_over_time.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        plt.close()

