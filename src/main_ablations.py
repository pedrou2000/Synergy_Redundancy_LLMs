from time_series_generation import *
from phid import *
from graph_theoretical_analysis import *
from cognitive_tasks_analysis import *
from cognitive_tasks_vs_syn_red_analysis import *
from lda import *
from random_walk_time_series import *
from hf_token import TOKEN

from huggingface_hub import login
from transformers import AutoTokenizer, AutoConfig 
import seaborn as sns
import matplotlib.pyplot as plt

import ablation_studies
import utils
import random, os, json
import matplotlib.pyplot as plt

device = torch.device("cuda")
login(token = TOKEN)
attn_implementation="eager" # GEMMA_ATTENTION_CLASSES = {"eager": GemmaAttention, "flash_attention_2": GemmaFlashAttention2, "sdpa": GemmaSdpaAttention,}


# Load the configuration and modify it
model_config = AutoConfig.from_pretrained(constants.MODEL_NAME, cache_dir=constants.CACHE_DIR_BITBUCKET)
model_config._attn_implementation = attn_implementation  # Custom attention parameter

# Load the tokenizer and model with the modified configuration
tokenizer = AutoTokenizer.from_pretrained(constants.MODEL_NAME, cache_dir=constants.CACHE_DIR_BITBUCKET)
model = AutoModelForCausalLM.from_pretrained(
    constants.MODEL_NAME,
    cache_dir=constants.CACHE_DIR_BITBUCKET,
    device_map='auto',
    attn_implementation=attn_implementation, # Make sure to use the adequate attention layer in order to 
    config=model_config,  # Use the modified config
)

model.eval()
print("Loaded Model Name: ", model.config.name_or_path)
print("Model: ", model)
print("Attention Layers Implementation: ", model.config._attn_implementation)
print(f"Number of layers: {constants.NUM_LAYERS}")
print(f"Number of attention heads per layer: {constants.NUM_HEADS_PER_LAYER}")


##### Get the Ablation Order #####
prompt_category_name = 'average_prompts'
global_matrices, synergy_matrices, redundancy_matrices = load_matrices(base_save_path=constants.MATRICES_DIR + prompt_category_name + '/' + prompt_category_name + '.pt')
averages = calculate_average_synergy_redundancies_per_head(synergy_matrices, redundancy_matrices, within_layer=False)
gradient_ranks = compute_gradient_rank(averages)

### Gradient Ranks ###
gradient_ranks = gradient_ranks["attention_weights"]
sorted_heads_by_rank = sorted(gradient_ranks.items(), key=lambda x: x[1], reverse=True)
syn_minus_red_ranked_heads = [utils.get_layer_and_head(head_num) for head_num, _ in sorted_heads_by_rank]

### Synergy and Redundancy Ranks ###
synergies, redundancies = averages["attention_weights"]["synergy"], averages["attention_weights"]["redundancy"]

synergies_dict = {i+1: synergies[i] for i in range(len(synergies))}
redundancies_dict = {i+1: redundancies[i] for i in range(len(redundancies))}

synergy_ranks = {i: rank for rank, (i, _) in enumerate(sorted(synergies_dict.items(), key=lambda x: x[1], reverse=True))}
redundancy_ranks = {i: rank for rank, (i, _) in enumerate(sorted(redundancies_dict.items(), key=lambda x: x[1], reverse=True))}

sorted_heads_by_synergy = sorted(synergies_dict.items(), key=lambda x: x[1], reverse=True)
sorted_heads_by_redundancy = sorted(redundancies_dict.items(), key=lambda x: x[1], reverse=True)

synergy_ranked_heads = [utils.get_layer_and_head(head_num) for head_num, _ in sorted_heads_by_synergy]
redundancy_ranked_heads = [utils.get_layer_and_head(head_num) for head_num, _ in sorted_heads_by_redundancy]

head_rankings = {"synergy": synergy_ranked_heads, "redundancy": redundancy_ranked_heads, "syn_minus_red_temp_1": syn_minus_red_ranked_heads}
print(head_rankings)


##### Perform the Ablation Studies #####

import random, os, pickle
import matplotlib.pyplot as plt


num_tokens_to_generate = 50
ablation_ranking_method = constants.ABLATIONS_RANKING_METHOD
num_random_ablations = 5  # Number of iterations to repeat the random ablation process
num_heads_skip_per_iteration = 5  # The fixed number of heads to ablate in each iteration

# The resulting divergence trajectories for random and gradient rank-based ablations for each prompt
# The first key is the prompt category name, and the second key is the prompt number, third is whether it is random or gradient 
# The value is a list of divergence trajectories for each iteration
divergence_results = {
    "divergences": {prompt_category_name: {prompt_num: {'random': [], 'gradient': []} for prompt_num in range(len(constants.PROMPTS[prompt_category_name]))} for prompt_category_name in constants.PROMPTS.keys()},
    "list_heads_ablated": [i for i in range(0, constants.NUM_TOTAL_HEADS, num_heads_skip_per_iteration)],
}

n_prompts_per_category = len(constants.PROMPTS[list(constants.PROMPTS.keys())[0]])
for prompt_category_name, prompt_list in constants.PROMPTS.items():
    ##### Run the Non-Ablated Model #####
    model.reset_ablated_heads()
    model.print_ablated_heads()
    non_ablated_texts, non_ablated_logits, non_ablated_token_ids = ablation_studies.generate_text_with_logits_batch(
        model, 
        tokenizer, 
        prompts=prompt_list,
        num_tokens_to_generate=num_tokens_to_generate,
        device=device, 
        temperature=1.5
    )
    print(non_ablated_texts)

    ##### Run the Rank-Based Ablation #####
    # Loop over the number of steps required to ablate all heads
    for i in range(0, constants.NUM_TOTAL_HEADS, num_heads_skip_per_iteration):
        # Get the current list of heads to ablate
        current_heads_to_ablated = head_rankings[ablation_ranking_method][:i + num_heads_skip_per_iteration]

        # Initialize the ablated_attention_heads dictionary
        ablated_attention_heads = {}
        for layer, head in current_heads_to_ablated:
            if layer not in ablated_attention_heads:
                ablated_attention_heads[layer] = []
            ablated_attention_heads[layer].append(head)

        # Generate the output with the current ablated heads
        ablated_logits_tensor, ablated_divergence_tensor, ablated_texts = ablation_studies.generate_with_teacher_forcing_ablated_batch(
            model,
            tokenizer,
            prompts=prompt_list,
            non_ablated_token_ids=non_ablated_token_ids,
            non_ablated_logits=non_ablated_logits,
            device=device,
            temperature=1.5,
            ablated_attention_heads=ablated_attention_heads,
            verbose=False
        )

        # Record the sum of divergences
        for prompt_num, divergence in enumerate(ablated_divergence_tensor):
            divergence = divergence.mean().item()
            divergence_results["divergences"][prompt_category_name][prompt_num]['gradient'].append(divergence)

    ##### Run the Random Ablations #####
    for iteration in range(num_random_ablations):
        # Generate a list of all attention heads (layer, head)
        attention_heads_list = [(layer, head) for layer in range(constants.NUM_LAYERS) for head in range(constants.NUM_HEADS_PER_LAYER)]

        # Randomly shuffle the list to determine the ablation order
        random.shuffle(attention_heads_list)

        # Initialize list to record divergences for this iteration
        divergence_dict = {prompt_num: [] for prompt_num in range(len(prompt_list))}

        # Loop over the number of steps required to ablate all heads
        for i in range(0, constants.NUM_TOTAL_HEADS, num_heads_skip_per_iteration):
            # Get the current list of heads to ablate
            current_heads_to_ablated = attention_heads_list[:i + num_heads_skip_per_iteration]

            # Initialize the ablated_attention_heads dictionary
            ablated_attention_heads = {}
            for layer, head in current_heads_to_ablated:
                if layer not in ablated_attention_heads:
                    ablated_attention_heads[layer] = []
                ablated_attention_heads[layer].append(head)

            # Call your method to generate the output with the current ablated heads
            ablated_logits, ablated_divergences, generated_text = ablation_studies.generate_with_teacher_forcing_ablated_batch(
                model,
                tokenizer,
                prompts=prompt_list,
                non_ablated_token_ids=non_ablated_token_ids,
                non_ablated_logits=non_ablated_logits,
                device=device,
                temperature=1.5,
                ablated_attention_heads=ablated_attention_heads,
                verbose=False
            )

            for prompt_num, divergence in enumerate(ablated_divergences):
                divergence_dict[prompt_num].append(divergence.mean().item())
        
        for prompt_number in range(len(prompt_list)):
            divergence_results["divergences"][prompt_category_name][prompt_number]['random'].append(divergence_dict[prompt_number])

        # Optional: Print progress for each iteration
        print(f'Completed random ablation iteration {iteration + 1}.')

    
    ##### Save the results to a pickle file #####
    save_dir = constants.ABLATIONS_DIR + ablation_ranking_method + '/'
    os.makedirs(save_dir, exist_ok=True)
    # Save to a pickle file
    with open(save_dir + 'divergence_results.pkl', 'wb') as f:  
        pickle.dump(divergence_results, f)

    ##### PLOTING #####
    for prompt_num in range(len(prompt_list)):
        # Plot the divergence trajectories for random ablations and gradient rank-based ablation
        plt.figure(figsize=(10, 6))

        # Plot random ablations
        for idx, divergence_trajectory in enumerate(divergence_results["divergences"][prompt_category_name][prompt_num]['random']):
            plt.plot(divergence_results["list_heads_ablated"], divergence_trajectory, marker='o', color='blue', alpha=0.5,
                    label='Random Ablations' if idx == 0 else "")

        # Plot gradient rank-based ablation
        plt.plot(divergence_results["list_heads_ablated"], divergence_results["divergences"][prompt_category_name][prompt_num]['gradient'], marker='o', color='red', label='Gradient Rank Ablations')

        plt.xlabel('Number of Heads Ablated')
        plt.ylabel('Divergence Mean')
        plt.title('Divergence vs. Number of Ablated Attention Heads')
        plt.legend()
        plt.grid(True)
        save_dir = constants.PLOT_ABLATIONS + ablation_ranking_method + '/' + prompt_category_name + '/' + '1-Divergence_Plots/' 
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir + str(prompt_num) + '_random_vs_gradient_rank_ablations_divergence_trajectories.png')
        # plt.show()
        plt.close()