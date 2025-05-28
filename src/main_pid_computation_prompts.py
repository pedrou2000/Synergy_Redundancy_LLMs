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
import concurrent.futures
import os

# Constants 
MODEL_NAMES = {
    # Qwen
    'Q3-0': {'hf_name': 'Qwen/Qwen3-0.6B-Base', 'company': 'alibaba', 'model_family': 'qwen-3', 'model_size': '0.6B', 'it': 'base', 'plot_name': 'Qwen 3 0.6B Base', 'color': '#8e9e00'},
    'Q3-1': {'hf_name': 'Qwen/Qwen3-1.7B-Base', 'company': 'alibaba', 'model_family': 'qwen-3', 'model_size': '1.7B', 'it': 'base', 'plot_name': 'Qwen 3 1.7B Base', 'color': '#a0ae00'},
    'Q3-4': {'hf_name': 'Qwen/Qwen3-4B-Base', 'company': 'alibaba', 'model_family': 'qwen-3', 'model_size': '4B', 'it': 'base', 'plot_name': 'Qwen 3 4B Base', 'color': '#b0be00'},
    'Q3-8': {'hf_name': 'Qwen/Qwen3-8B-Base', 'company': 'alibaba', 'model_family': 'qwen-3', 'model_size': '8B', 'it': 'base', 'plot_name': 'Qwen 3 8B Base', 'color': '#c0ce00'},
    'Q3-14': {'hf_name': 'Qwen/Qwen3-14B-Base', 'company': 'alibaba', 'model_family': 'qwen-3', 'model_size': '14B', 'it': 'base', 'plot_name': 'Qwen 3 14B Base', 'color': '#d0de00'},
    'Q3-30-A3': {'hf_name': 'Qwen/Qwen3-30B-A3B-Base', 'company': 'alibaba', 'model_family': 'qwen-3', 'model_size': '30B-A3B', 'it': 'base', 'plot_name': 'Qwen 3 30B A3B Base', 'color': '#e0ee00'},
    'Q25M-1': {'hf_name': 'Qwen/Qwen2.5-Math-1.5B', 'company': 'alibaba', 'model_family': 'qwen-2.5-math', 'model_size': '1.5B', 'it': 'base', 'plot_name': 'Qwen 2.5 Math 1.5B Base', 'color': '#8e9e00'},
    'Q25M-7': {'hf_name': 'Qwen/Qwen2.5-Math-7B', 'company': 'alibaba', 'model_family': 'qwen-2.5-math', 'model_size': '7B', 'it': 'base', 'plot_name': 'Qwen 2.5 Math 7B Base', 'color': '#a0ae00'},
    'Q25M-72': {'hf_name': 'Qwen/Qwen2.5-Math-72B', 'company': 'alibaba', 'model_family': 'qwen-2.5-math', 'model_size': '72B', 'it': 'base', 'plot_name': 'Qwen 2.5 Math 72B Base', 'color': '#b0be00'},
    
    # Gemma
    'G3-1': {'hf_name': 'google/gemma-3-1b-pt', 'company': 'google', 'model_family': 'gemma-3', 'model_size': '1B', 'it': 'base', 'plot_name': 'Gemma 3 1B Base', 'color': '#647c00'},
    'G3-4': {'hf_name': 'google/gemma-3-4b-pt', 'company': 'google', 'model_family': 'gemma-3', 'model_size': '4B', 'it': 'base', 'plot_name': 'Gemma 3 4B Base', 'color': '#7c8e00'},
    'G3-12': {'hf_name': 'google/gemma-3-12b-pt', 'company': 'google', 'model_family': 'gemma-3', 'model_size': '12B', 'it': 'base', 'plot_name': 'Gemma 3 12B Base', 'color': '#8e9e00'},
    'G3-27': {'hf_name': 'google/gemma-3-27b-pt', 'company': 'google', 'model_family': 'gemma-3', 'model_size': '27B', 'it': 'base', 'plot_name': 'Gemma 3 27B Base', 'color': '#9eae00'},

    # Llama
}
MODEL_CODE = os.getenv("MODEL_CODE", "Q25M-1")
MODEL_NAME = MODEL_NAMES[MODEL_CODE]["hf_name"]
FOLDER_MODEL_NAME = MODEL_NAMES[MODEL_CODE]["company"] + "/" + MODEL_NAMES[MODEL_CODE]["model_family"] + "/" + MODEL_NAMES[MODEL_CODE]["model_size"] + "/" + MODEL_NAMES[MODEL_CODE]["it"]
FINAL_MODELS = ['Q3-0', 'Q3-1', 'Q3-4', 'Q3-8', 'Q3-14', 'G3-1', 'G3-4', 'G3-12']

PLOTS_DIR = "../plots/" + FOLDER_MODEL_NAME + "/"
SAVED_DATA_DIR = "../data/" + FOLDER_MODEL_NAME + "/"
TIME_SERIES_DIR = SAVED_DATA_DIR + "2-Time_Series/"
MATRICES_DIR = SAVED_DATA_DIR + "3-Synergy_Redundancy_Matrices/"
GRAPH_METRICS_DIR = SAVED_DATA_DIR + "5-Graph_Theoretical_Properties/"

constants.update_model_code(MODEL_CODE)

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

# Function to compute PhiID and save results
def compute_and_save(task, n_prompt, prompt, time_series, save_path):
    print(f"\n--- Computing PhiID for task {task} ---")
    print(f"\nPrompt: {n_prompt}")
    result = compute_PhiID(time_series[task][n_prompt], save=True, kind="gaussian", base_save_path=save_path)
    all_matrices, synergy_matrices, redundancy_matrices = result
    all_matrices, synergy_matrices, redundancy_matrices = load_matrices(base_save_path=save_path)
    result = (all_matrices, synergy_matrices, redundancy_matrices)
    graph_theoretical_results = compare_synergy_redundancy(synergy_matrices, redundancy_matrices)
    save_graph_theoretical_results(graph_theoretical_results, file_name=str(n_prompt), base_save_path = GRAPH_METRICS_DIR + task + '/')
    return (task, n_prompt, result)

# Create a dictionary to store all matrices
all_matrices = {task: {} for task in categories}
synergy_matrices = {task: {} for task in categories}
redundancy_matrices = {task: {} for task in categories}

# List to keep track of futures
futures = []

# Create a thread pool executor
num_workers = os.cpu_count()  # or any number you find optimal
print(f"Using {num_workers} workers for parallel processing.")
with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:   
    for cognitive_task in categories:
        for n_prompt, prompt in enumerate(constants.PROMPTS[cognitive_task]):
            save_path = os.path.join(MATRICES_DIR, cognitive_task, f"{n_prompt}.pt")
            futures.append(
                executor.submit(compute_and_save, cognitive_task, n_prompt, prompt, time_series, save_path)
            )

    # Wait for all futures to complete and process the results
    for future in concurrent.futures.as_completed(futures):
        task, n_prompt, (all_matrix, synergy_matrix, redundancy_matrix) = future.result()
        all_matrices[task][n_prompt] = all_matrix
        synergy_matrices[task][n_prompt] = synergy_matrix
        redundancy_matrices[task][n_prompt] = redundancy_matrix

print("All tasks completed.")


# Compute and Save Average Prompt Matrices
cognitive_task = 'average_prompts'
all_matrices, synergy_matrices, redundancy_matrices = average_synergy_redundancies_matrices_cognitive_tasks(all_matrices, synergy_matrices, redundancy_matrices)
save_matrices(all_matrices, synergy_matrices, redundancy_matrices, base_save_path=MATRICES_DIR + cognitive_task + '/' + cognitive_task + '.pt')
graph_theoretical_results = compare_synergy_redundancy(synergy_matrices, redundancy_matrices)
save_graph_theoretical_results(graph_theoretical_results, file_name=cognitive_task, base_save_path=GRAPH_METRICS_DIR+cognitive_task+'/')
  