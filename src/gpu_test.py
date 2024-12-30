from huggingface_hub import login
from transformers import AutoTokenizer, AutoConfig 
import seaborn as sns
import matplotlib.pyplot as plt
import utils
import random, os, json, torch

device = torch.device("cuda")
login(token = "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")
attn_implementation="eager" # GEMMA_ATTENTION_CLASSES = {"eager": GemmaAttention, "flash_attention_2": GemmaFlashAttention2, "sdpa": GemmaSdpaAttention,}

print('Hello World!!!')


from time_series_generation import *
from phid import *
from graph_theoretical_analysis import *
from cognitive_tasks_analysis import *
from cognitive_tasks_vs_syn_red_analysis import *
from lda import *
from random_walk_time_series import *
from hf_token import TOKEN

from huggingface_hub import login
from transformers import AutoTokenizer, AutoConfig 
import seaborn as sns
import matplotlib.pyplot as plt

import subprocess

def get_gpu_memory_usage():
    try:
        # Use nvidia-smi to get the memory info
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            print("Error: Unable to run nvidia-smi. Make sure NVIDIA drivers are installed.")
            print(result.stderr)
            return
        
        # Parse the output
        memory_info = result.stdout.strip().split("\n")
        for i, line in enumerate(memory_info):
            used, total = map(int, line.split(","))
            available = total - used
            print(f"GPU {i}:")
            print(f"  Used Memory: {used} MB")
            print(f"  Total Memory: {total} MB")
            print(f"  Available Memory: {available} MB")
    except FileNotFoundError:
        print("Error: nvidia-smi not found. Ensure you have an NVIDIA GPU and drivers installed.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    get_gpu_memory_usage()

    if constants.LOAD_MODEL:
        device = torch.device("cuda")
        login(token = TOKEN)
        attn_implementation="eager" # GEMMA_ATTENTION_CLASSES = {"eager": GemmaAttention, "flash_attention_2": GemmaFlashAttention2, "sdpa": GemmaSdpaAttention,}


        # Load the configuration and modify it
        model_config = AutoConfig.from_pretrained(constants.MODEL_NAME, cache_dir=constants.CACHE_DIR_BITBUCKET)
        model_config._attn_implementation = attn_implementation  # Custom attention parameter

        # Load the tokenizer and model with the modified configuration
        tokenizer = AutoTokenizer.from_pretrained(constants.MODEL_NAME, cache_dir=constants.CACHE_DIR_BITBUCKET)
        model = AutoModelForCausalLM.from_pretrained(
            constants.MODEL_NAME,
            cache_dir=constants.CACHE_DIR_BITBUCKET,
            device_map='auto',
            attn_implementation=attn_implementation, # Make sure to use the adequate attention layer in order to 
            config=model_config,  # Use the modified config
        )

        model.eval()
        print("Loaded Model Name: ", model.config.name_or_path)
        print("Model: ", model)
        print("Attention Layers Implementation: ", model.config._attn_implementation)
        print(f"Number of layers: {constants.NUM_LAYERS}")
        print(f"Number of attention heads per layer: {constants.NUM_HEADS_PER_LAYER}")