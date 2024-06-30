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


# Generate Time Series
print("\n\n\n--- Generating Time Series ---\n")
random_input_length, num_tokens_to_generate, temperature = 24, 100, 0.3
generated_text, attention_params, time_series = {}, {}, {}

for cognitive_task in constants.PROMPT_CATEGORIES:
    print("Cognitive Task: ", cognitive_task)
    for n_prompt, prompt in enumerate(constants.PROMPTS[cognitive_task]):
        print("Prompt: ", n_prompt)
    
        generated_text[cognitive_task], attention_params[cognitive_task] = generate_text_with_attention(model, tokenizer, 
            num_tokens_to_generate, device, prompt=prompt, temperature=temperature, modified_output_attentions=constants.MODIFIED_OUTPUT_ATTENTIONS)
        save_raw_attention(generated_text[cognitive_task], attention_params[cognitive_task],  base_save_path=constants.RAW_ATTENTION_DIR+cognitive_task+"/"+ str(n_prompt) + "-")
        
        time_series[cognitive_task] = compute_attention_metrics_norms(attention_params[cognitive_task], constants.METRICS_TRANSFORMER, num_tokens_to_generate, aggregation_type='norm')
        save_time_series(time_series[cognitive_task], base_save_path=constants.TIME_SERIES_DIR+cognitive_task+"/"+str(n_prompt) + ".pt")


# Attention Weights Average Activation per Task Category and Attention Head
print("\n\n\n--- Generating Attention Weights ---\n")
attention_weights_prompts, generated_text = solve_prompts(constants.PROMPTS, model, tokenizer, device, num_tokens_to_generate=64,temperature=0.7)
save_attention_weights(attention_weights_prompts, generated_text)

