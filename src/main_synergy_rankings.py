from time_series_generation import *
from phid import *
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

models = ['Q25M-1', 'Q3-0', 'Q3-1', 'Q3-8']
all_head_rankings = {}
for model_code in models:
    constants.update_model_code(model_code)
    print(f"--- Running for model code: {model_code} ---")

    ##### Get the Ablation Order #####
    prompt_category_name = 'average_prompts'
    global_matrices, synergy_matrices, redundancy_matrices = load_matrices(base_save_path=constants.MATRICES_DIR + prompt_category_name + '/' + prompt_category_name + '.pt')
    averages = calculate_average_synergy_redundancies_per_head(synergy_matrices, redundancy_matrices, within_layer=False)
    gradient_ranks = compute_gradient_rank(averages)

    ### Gradient Ranks ###
    gradient_ranks = gradient_ranks[constants.ATTENTION_MEASURE]
    sorted_heads_by_rank = sorted(gradient_ranks.items(), key=lambda x: x[1], reverse=True)
    syn_minus_red_ranked_heads = [utils.get_layer_and_head(head_num) for head_num, _ in sorted_heads_by_rank]

    ### Synergy and Redundancy Ranks ###
    synergies, redundancies = averages[constants.ATTENTION_MEASURE]["synergy"], averages[constants.ATTENTION_MEASURE]["redundancy"]

    synergies_dict = {i+1: synergies[i] for i in range(len(synergies))}
    redundancies_dict = {i+1: redundancies[i] for i in range(len(redundancies))}

    synergy_ranks = {i: rank for rank, (i, _) in enumerate(sorted(synergies_dict.items(), key=lambda x: x[1], reverse=True))}
    redundancy_ranks = {i: rank for rank, (i, _) in enumerate(sorted(redundancies_dict.items(), key=lambda x: x[1], reverse=True))}

    sorted_heads_by_synergy = sorted(synergies_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_heads_by_redundancy = sorted(redundancies_dict.items(), key=lambda x: x[1], reverse=True)

    synergy_ranked_heads = [utils.get_layer_and_head(head_num) for head_num, _ in sorted_heads_by_synergy]
    redundancy_ranked_heads = [utils.get_layer_and_head(head_num) for head_num, _ in sorted_heads_by_redundancy]

    head_rankings = {"synergy": synergy_ranked_heads, "redundancy": redundancy_ranked_heads, "syn_minus_red": syn_minus_red_ranked_heads}
    all_head_rankings[constants.MODEL_NAME] = syn_minus_red_ranked_heads

# Save the head rankings to a JSON file in MODEL_COMPARISON_DIR
# Create a string with all the model codes
model_codes = "-".join(models)
file_name = f"head_rankings_{model_codes}.json"
if not os.path.exists(constants.MODEL_COMPARISON_SYN_MINUS_RED_RANKINGS_DIR):
    os.makedirs(constants.MODEL_COMPARISON_SYN_MINUS_RED_RANKINGS_DIR)
with open(os.path.join(constants.MODEL_COMPARISON_SYN_MINUS_RED_RANKINGS_DIR, file_name), 'w') as f:
    json.dump(all_head_rankings, f)
print(f"Head rankings saved to {os.path.join(constants.MODEL_COMPARISON_DIR, file_name)}")
print(all_head_rankings)

