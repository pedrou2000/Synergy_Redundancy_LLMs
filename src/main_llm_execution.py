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


# Load the Model
device = torch.device("cuda")
login(token = TOKEN)
attn_implementation="eager" # GEMMA_ATTENTION_CLASSES = {"eager": GemmaAttention, "flash_attention_2": GemmaFlashAttention2, "sdpa": GemmaSdpaAttention,}


tokenizer = AutoTokenizer.from_pretrained(constants.MODEL_NAME, cache_dir=constants.CACHE_DIR_BITBUCKET)
model = AutoModelForCausalLM.from_pretrained(
    constants.MODEL_NAME, 
    cache_dir=constants.CACHE_DIR_BITBUCKET, 
    device_map='auto', 
    attn_implementation=attn_implementation, # Make sure to use the adequate attention layer in order to 
)
model.eval()
print("Loaded Model Name: ", model.config.name_or_path)
print("Model: ", model)
print("Attention Layers Implementation: ", model.config._attn_implementation)
print(f"Number of layers: {constants.NUM_LAYERS}")
print(f"Number of attention heads per layer: {constants.NUM_HEADS_PER_LAYER}")


# Generate Time Series
random_input_length, num_tokens_to_generate, temperature = 24, 100, 0.3
generated_text, attention_params, time_series = {}, {}, {}

for cognitive_task in constants.PROMPT_CATEGORIES:
    print("Cognitive Task: ", cognitive_task)
    prompt = constants.PROMPTS[cognitive_task][0]
    
    generated_text[cognitive_task], attention_params[cognitive_task] = generate_text_with_attention(model, tokenizer, 
        num_tokens_to_generate, device, prompt=prompt, temperature=temperature, modified_output_attentions=constants.MODIFIED_OUTPUT_ATTENTIONS)
    save_raw_attention(generated_text[cognitive_task], attention_params[cognitive_task],  base_save_path=constants.RAW_ATTENTION_DIR+cognitive_task+"/")
    
    time_series[cognitive_task] = compute_attention_metrics_norms(attention_params[cognitive_task], constants.METRICS_TRANSFORMER, num_tokens_to_generate, aggregation_type='norm')
    save_time_series(time_series[cognitive_task], base_save_path=constants.TIME_SERIES_DIR+cognitive_task+".pt")
    plot_attention_metrics_norms_over_time(time_series[cognitive_task], metrics=constants.METRICS_TRANSFORMER, num_heads_plot=8, 
        smoothing_window=10, save=True, base_plot_path=constants.PLOTS_TIME_SERIES_DIR+cognitive_task+"/")
    print(f'Generated Text for {cognitive_task}: {generated_text[cognitive_task]}')


# Attention Weights Average Activation per Task Category and Attention Head
base_plot_path = constants.PLOTS_HEAD_ACTIVATIONS_ANALYSIS + datetime.now().strftime("%Y%m%d_%H%M%S") + '/'
attention_weights_prompts, generated_text = solve_prompts(constants.PROMPTS, model, tokenizer, device, num_tokens_to_generate=64,temperature=0.7, attention_measure=constants.ATTENTION_MEASURE)
save_attention_weights(attention_weights_prompts, generated_text)
summary_stats_prompts = plot_categories_comparison(attention_weights_prompts, save=True, base_plot_path=base_plot_path, split_half=False, split_third=False)
plot_all_heatmaps(attention_weights_prompts, save=True, base_plot_path=base_plot_path)

