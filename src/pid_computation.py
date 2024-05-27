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

device = torch.device("cuda" if constants.USE_GPU else "cpu")
login(token = TOKEN)
attn_implementation="eager" # GEMMA_ATTENTION_CLASSES = {"eager": GemmaAttention, "flash_attention_2": GemmaFlashAttention2, "sdpa": GemmaSdpaAttention,}


tokenizer = AutoTokenizer.from_pretrained(constants.MODEL_NAME, cache_dir=constants.CACHE_DIR_BITBUCKET)
model = AutoModelForCausalLM.from_pretrained(
    constants.MODEL_NAME, 
    cache_dir=constants.CACHE_DIR_BITBUCKET, 
    device_map='auto', 
    attn_implementation=attn_implementation, # Make sure to use the adequate attention layer in order to 
).to(device)
model.eval()
print("Attention Layers Implementation: ", model.config._attn_implementation)

random_input_length, num_tokens_to_generate, temperature = 24, 100, 0.3
time_series = {}

for cognitive_task in constants.prompts_2.keys():
    print("Cognitive Task: ", cognitive_task)
    prompt = constants.prompts_2[cognitive_task][0]
    generated_text, attention_params = generate_text_with_attention(model, tokenizer, num_tokens_to_generate, device, prompt=prompt, 
        temperature=temperature, modified_output_attentions=constants.MODIFIED_OUTPUT_ATTENTIONS)
    time_series[cognitive_task] = compute_attention_metrics_norms(attention_params, constants.METRICS_TRANSFORMER, num_tokens_to_generate)
    save_time_series(time_series[cognitive_task], base_save_path=constants.TIME_SERIES_DIR+cognitive_task+".pt")
    plot_attention_metrics_norms_over_time(time_series[cognitive_task], metrics=constants.METRICS_TRANSFORMER, num_heads_plot=8, smoothing_window=10, save=True,
                                           base_plot_path=constants.PLOTS_TIME_SERIES_DIR+cognitive_task+"/")
    print(f'Generated Text for {cognitive_task}: {generated_text}')

all_matrices, synergy_matrices, redundancy_matrices = {}, {}, {}
for cognitive_task in list(constants.prompts_2.keys()):
    print(cognitive_task)
    all_matrices[cognitive_task], synergy_matrices[cognitive_task], redundancy_matrices[cognitive_task] = compute_PhiID(time_series[cognitive_task],
                save=True, kind="gaussian", base_save_path=constants.MATRICES_DIR+cognitive_task+'.pt')
    plot_synergy_redundancy_PhiID(synergy_matrices[cognitive_task], redundancy_matrices[cognitive_task], save=True, base_plot_path=constants.PLOTS_SYNERGY_REDUNDANCY_DIR+cognitive_task+"/")