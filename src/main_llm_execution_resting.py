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

# Load the Model
print("--- Loading Model: ", constants.MODEL_NAME, " ---\n\n")
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

cognitive_task = "resting_state"

# Generate Time Series
print("\n\n\n--- Generating Time Series ---\n")
random_input_length, num_tokens_to_generate, temperature = 24, 1000, 3

generated_text, attention_params = simulate_resting_state_attention(model, tokenizer, num_tokens_to_generate, device, temperature=temperature, 
                                                                    random_input_length=random_input_length)
save_raw_attention(generated_text, attention_params,  base_save_path=constants.RAW_ATTENTION_DIR+cognitive_task+"/")

time_series = compute_attention_metrics_norms(attention_params, constants.METRICS_TRANSFORMER, num_tokens_to_generate, aggregation_type='norm')
save_time_series(time_series, base_save_path=constants.TIME_SERIES_DIR+cognitive_task+".pt")


# Attention Weights Average Activation per Task Category and Attention Head
print("\n\n\n--- Generating Attention Weights ---\n")
prompts_dict = {cognitive_task: ""}
attention_weights_prompts, generated_text = solve_prompts(prompts_dict, model, tokenizer, device, num_tokens_to_generate=128,
                                                          temperature=2)
save_attention_weights(attention_weights_prompts, generated_text, merge=True)

