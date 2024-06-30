from time_series_generation import *
from phid import *
from graph_theoretical_analysis import *
from cognitive_tasks_analysis import *
from cognitive_tasks_vs_syn_red_analysis import *
from lda import *
from hf_token import TOKEN

from huggingface_hub import login
from transformers import AutoTokenizer, BitsAndBytesConfig, GemmaForCausalLM
import seaborn as sns
import matplotlib.pyplot as plt

# Constants 
MODEL_NAMES = {
    1: {"HF_NAME": "google/gemma-2b-it", "FOLDER_NAME": "1-Gemma-2b-it"},
    2: {"HF_NAME": "google/gemma-1.1-2b-it", "FOLDER_NAME": "2-Gemma-1.1-2b-it"},
    3: {"HF_NAME": "google/gemma-1.1-7b-it", "FOLDER_NAME": "3-Gemma-1.1-7b-it"},
    4: {"HF_NAME": "meta-llama/Meta-Llama-3-8B-Instruct", "FOLDER_NAME": "4-Llama-3-8B-Instruct"},
    5: {"HF_NAME": "google/gemma-2-9b-it", "FOLDER_NAME": "5-Gemma-2-9B-Instruct"},
    6: {"HF_NAME": "meta-llama/Llama-2-13b-chat-hf", "FOLDER_NAME": "5-Llama-2-13b-chat-hf"},
    7: {"HF_NAME": "meta-llama/Llama-2-7b-chat-hf", "FOLDER_NAME": "6-Llama-2-7b-chat-hf"},
    8: {"HF_NAME": "meta-llama/Meta-Llama-3-70B-Instruct", "FOLDER_NAME": "7-Llama-3-70B-Instruct"},
}
MODEL_NUMBER = 2
MODEL_NAME = MODEL_NAMES[MODEL_NUMBER]["HF_NAME"]
FOLDER_MODEL_NAME = MODEL_NAMES[MODEL_NUMBER]["FOLDER_NAME"]

PLOTS_DIR = "../plots/" + FOLDER_MODEL_NAME + "/"
SAVED_DATA_DIR = "../data/" + FOLDER_MODEL_NAME + "/"
TIME_SERIES_DIR = SAVED_DATA_DIR + "2-Time_Series/"
MATRICES_DIR = SAVED_DATA_DIR + "3-Synergy_Redundancy_Matrices/"

print("--- Computing PhiID for model: ", MODEL_NAME, " ---\n")
print('Loading Time Series Data from ', TIME_SERIES_DIR)

# Compute PhiID for all cognitive tasks
random_input_length, num_tokens_to_generate, temperature = 24, 100, 0.3
generated_text = {cognitive_task: {} for cognitive_task in constants.PROMPT_CATEGORIES}
attention_params = {cognitive_task: {} for cognitive_task in constants.PROMPT_CATEGORIES}
time_series = {cognitive_task: {} for cognitive_task in constants.PROMPT_CATEGORIES}

categories = constants.PROMPT_CATEGORIES
for cognitive_task in categories:
    for n_prompt, prompt in enumerate(constants.PROMPTS[cognitive_task]):
        time_series[cognitive_task][n_prompt] = load_time_series(base_load_path=TIME_SERIES_DIR+cognitive_task+"/"+str(n_prompt) + ".pt")

print('Computing PhiID and saving matrices to ', MATRICES_DIR, '\n', '-'*50, '\n\n')
all_matrices = {task: {} for task in categories}
synergy_matrices = {task: {} for task in categories}
redundancy_matrices = {task: {} for task in categories}

for cognitive_task in categories:
    print("\n--- Computing PhiID for task ", cognitive_task, " ---")
    for n_prompt, prompt in enumerate(constants.PROMPTS[cognitive_task]):
        print("\nPrompt: ", n_prompt)
        all_matrices[cognitive_task][n_prompt], synergy_matrices[cognitive_task][n_prompt], redundancy_matrices[cognitive_task][n_prompt] = compute_PhiID(
            time_series[cognitive_task][n_prompt], save=True, kind="gaussian", base_save_path=MATRICES_DIR+cognitive_task+'/' + str(n_prompt) + '.pt')
        graph_theoretical_results = compare_synergy_redundancy(synergy_matrices[cognitive_task][n_prompt], redundancy_matrices[cognitive_task][n_prompt])
        save_graph_theoretical_results(graph_theoretical_results, file_name=str(n_prompt), base_save_path = constants.GRAPH_METRICS_DIR + cognitive_task + '/')

# Compute and Save Average Prompt Matrices
all_matrices, synergy_matrices, redundancy_matrices = average_synergy_redundancies_matrices_cognitive_tasks(all_matrices, synergy_matrices, redundancy_matrices)
save_matrices(all_matrices, synergy_matrices, redundancy_matrices, base_save_path=MATRICES_DIR + 'average_prompts' + '.pt')