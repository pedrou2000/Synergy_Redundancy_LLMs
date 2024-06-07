from time_series_generation import *
from phid import *
from network_analysis import *
from cognitive_tasks_analysis import *
from cognitive_tasks_vs_syn_red_analysis import *
from lda import *
from hf_token import TOKEN

from huggingface_hub import login
from transformers import AutoTokenizer, BitsAndBytesConfig, GemmaForCausalLM
import seaborn as sns
import matplotlib.pyplot as plt

random_input_length, num_tokens_to_generate, temperature = 24, 100, 0.3
generated_text, attention_params, time_series = {}, {}, {}

for cognitive_task in constants.PROMPT_CATEGORIES:
    time_series[cognitive_task] = load_time_series(base_load_path=constants.TIME_SERIES_DIR+cognitive_task+".pt")


all_matrices, synergy_matrices, redundancy_matrices = {}, {}, {}
for cognitive_task in constants.PROMPT_CATEGORIES:
    print("\n--- Computing PhiID for task ", cognitive_task, " ---")
    all_matrices[cognitive_task], synergy_matrices[cognitive_task], redundancy_matrices[cognitive_task] = compute_PhiID(time_series[cognitive_task],
                save=True, kind="gaussian", base_save_path=constants.MATRICES_DIR+cognitive_task+'.pt')

# Compute and Save Average Prompt Matrices
all_matrices, synergy_matrices, redundancy_matrices = average_synergy_redundancies_matrices_cognitive_tasks(all_matrices, synergy_matrices, redundancy_matrices)
save_matrices(all_matrices, synergy_matrices, redundancy_matrices, base_save_path=constants.MATRICES_DIR + 'average_prompts' + '.pt')