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
    'G1-2B': {"HF_NAME": "google/gemma-2b-it", "FOLDER_NAME": "1-Gemma-2b-it", "PLOT_NAME": "Gemma 1 2B"},
    'G1.1-2B': {"HF_NAME": "google/gemma-1.1-2b-it", "FOLDER_NAME": "2-Gemma-1.1-2b-it", "PLOT_NAME": "Gemma 1.1 2B"},
    'G1.1-7B': {"HF_NAME": "google/gemma-1.1-7b-it", "FOLDER_NAME": "3-Gemma-1.1-7b-it", "PLOT_NAME": "Gemma 1.1 7B"},
    'L3-8B': {"HF_NAME": "meta-llama/Meta-Llama-3-8B-Instruct", "FOLDER_NAME": "4-Llama-3-8B-Instruct", "PLOT_NAME": "Llama 3 8B"},
    'G2-2B': {"HF_NAME": "google/gemma-2-2b-it", "FOLDER_NAME": "5-Gemma-2-2B", "PLOT_NAME": "Gemma 2 2B", "COLOR": "#1f77b4"},
    'G2-9B': {"HF_NAME": "google/gemma-2-9b-it", "FOLDER_NAME": "6-Gemma-2-9B", "PLOT_NAME": "Gemma 2 9B", "COLOR": "#2ca02c"},
    'L3.2-3B': {"HF_NAME": "meta-llama/Llama-3.2-3B-Instruct", "FOLDER_NAME": "7-Llama-3.2-3B", "PLOT_NAME": "Llama 3.2 3B", "COLOR": "#ff7f0e"},
    'L3.1-8B': {"HF_NAME": "meta-llama/Llama-3.1-8B-Instruct", "FOLDER_NAME": "8-Llama-3.1-8B", "PLOT_NAME": "Llama 3.1 8B", "COLOR": "#9467bd"},
    'L3.1-8B-b': {"HF_NAME": "meta-llama/Llama-3.1-8B", "FOLDER_NAME": "9-Llama-3.1-8B-Base", "PLOT_NAME": "Llama 3.1 8B Base", "COLOR": "#9467ed"},
    'R1-L3.1-8B': {"HF_NAME": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "FOLDER_NAME": "10-R1-Distilled-Llama-3.1-8B", "PLOT_NAME": "R1 Distilled Llama 3.1 8B", "COLOR": "#9467ju"},
    'L3-1': {"HF_NAME": "meta-llama/Llama-3.2-1B", "FOLDER_NAME": "11-Llama-3.2-1B-Base", "PLOT_NAME": "Llama 3.2 1B", "COLOR": "#4d8e00"},
    'L3-3': {"HF_NAME": "meta-llama/Llama-3.2-3B", "FOLDER_NAME": "12-Llama-3.2-3B-Base", "PLOT_NAME": "Llama 3.2 3B", "COLOR": "#456e00"},
    'G3-1': {"HF_NAME": "google/gemma-3-1b-pt", "FOLDER_NAME": "13-Gemma-3-1B-Base", "PLOT_NAME": "Gemma 3 1B", "COLOR": "#647c00"},
    'G3-4': {"HF_NAME": "google/gemma-3-4b-pt", "FOLDER_NAME": "14-Gemma-3-4B-Base", "PLOT_NAME": "Gemma 3 4B", "COLOR": "#7c8e00"},
    'G3-12': {"HF_NAME": "google/gemma-3-12b-pt", "FOLDER_NAME": "15-Gemma-3-12B-Base", "PLOT_NAME": "Gemma 3 12B", "COLOR": "#8e9e00"},
    'Q3-0': {"HF_NAME": "Qwen/Qwen3-0.6B-Base", "FOLDER_NAME": "16-Qwen3-0.6B-Base", "PLOT_NAME": "Qwen 3 0.6B", "COLOR": "#8e9e00"},
    'Q3-1': {"HF_NAME": "Qwen/Qwen3-1.7B-Base", "FOLDER_NAME": "17-Qwen3-1.7B-Base", "PLOT_NAME": "Qwen 3 1.7B", "COLOR": "#a0ae00"},
    'Q3-4': {"HF_NAME": "Qwen/Qwen3-4B-Base", "FOLDER_NAME": "18-Qwen3-4B-Base", "PLOT_NAME": "Qwen 3 4B", "COLOR": "#b0be00"},
    'Q3-8': {"HF_NAME": "Qwen/Qwen3-8B-Base", "FOLDER_NAME": "19-Qwen3-8B-Base", "PLOT_NAME": "Qwen 3 8B", "COLOR": "#c0ce00"},
}
MODEL_CODE = 'G3-12'
MODEL_NAME = MODEL_NAMES[MODEL_CODE]["HF_NAME"]
FOLDER_MODEL_NAME = MODEL_NAMES[MODEL_CODE]["FOLDER_NAME"]

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

categories = constants.PROMPT_CATEGORIES[:-1][::-1]
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
  