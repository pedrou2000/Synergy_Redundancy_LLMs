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
# print(head_rankings)


