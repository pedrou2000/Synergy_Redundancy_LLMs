import subprocess
import sys

MODEL_NAMES = {
    # Qwen
    'Q3-0': {'hf_name': 'Qwen/Qwen3-0.6B-Base', 'company': 'alibaba', 'model_family': 'qwen-3', 'model_size': '0.6B', 'it': 'base', 'plot_name': 'Qwen 3 0.6B Base', 'color': '#8e9e00', 'apply_chat_template': 'no'},
    'Q3-1': {'hf_name': 'Qwen/Qwen3-1.7B-Base', 'company': 'alibaba', 'model_family': 'qwen-3', 'model_size': '1.7B', 'it': 'base', 'plot_name': 'Qwen 3 1.7B Base', 'color': '#a0ae00', 'apply_chat_template': 'no'},
    'Q3-4': {'hf_name': 'Qwen/Qwen3-4B-Base', 'company': 'alibaba', 'model_family': 'qwen-3', 'model_size': '4B', 'it': 'base', 'plot_name': 'Qwen 3 4B Base', 'color': '#b0be00', 'apply_chat_template': 'no'},
    'Q3-8': {'hf_name': 'Qwen/Qwen3-8B-Base', 'company': 'alibaba', 'model_family': 'qwen-3', 'model_size': '8B', 'it': 'base', 'plot_name': 'Qwen 3 8B Base', 'color': '#c0ce00', 'apply_chat_template': 'no'},
    'Q3-14': {'hf_name': 'Qwen/Qwen3-14B-Base', 'company': 'alibaba', 'model_family': 'qwen-3', 'model_size': '14B', 'it': 'base', 'plot_name': 'Qwen 3 14B Base', 'color': '#d0de00', 'apply_chat_template': 'no'},
    'Q3-30-A3': {'hf_name': 'Qwen/Qwen3-30B-A3B-Base', 'company': 'alibaba', 'model_family': 'qwen-3', 'model_size': '30B-A3B', 'it': 'base', 'plot_name': 'Qwen 3 30B A3B Base', 'color': '#e0ee00', 'apply_chat_template': 'no'},
    'Q25M-1': {'hf_name': 'Qwen/Qwen2.5-Math-1.5B', 'company': 'alibaba', 'model_family': 'qwen-2.5-math', 'model_size': '1.5B', 'it': 'base', 'plot_name': 'Qwen 2.5 Math 1.5B Base', 'color': '#8e9e00', 'apply_chat_template': 'no'},
    'Q25M-7': {'hf_name': 'Qwen/Qwen2.5-Math-7B', 'company': 'alibaba', 'model_family': 'qwen-2.5-math', 'model_size': '7B', 'it': 'base', 'plot_name': 'Qwen 2.5 Math 7B Base', 'color': '#a0ae00', 'apply_chat_template': 'no'},
    'Q25M-72': {'hf_name': 'Qwen/Qwen2.5-Math-72B', 'company': 'alibaba', 'model_family': 'qwen-2.5-math', 'model_size': '72B', 'it': 'base', 'plot_name': 'Qwen 2.5 Math 72B Base', 'color': '#b0be00', 'apply_chat_template': 'no'},
    
    # Gemma
    'G3-1': {'hf_name': 'google/gemma-3-1b-pt', 'company': 'google', 'model_family': 'gemma-3', 'model_size': '1B', 'it': 'base', 'plot_name': 'Gemma 3 1B Base', 'color': '#647c00', 'apply_chat_template': 'base'},
    'G3-4': {'hf_name': 'google/gemma-3-4b-pt', 'company': 'google', 'model_family': 'gemma-3', 'model_size': '4B', 'it': 'base', 'plot_name': 'Gemma 3 4B Base', 'color': '#7c8e00', 'apply_chat_template': 'base'},
    'G3-12': {'hf_name': 'google/gemma-3-12b-pt', 'company': 'google', 'model_family': 'gemma-3', 'model_size': '12B', 'it': 'base', 'plot_name': 'Gemma 3 12B Base', 'color': '#8e9e00', 'apply_chat_template': 'base'},
    'G3-27': {'hf_name': 'google/gemma-3-27b-pt', 'company': 'google', 'model_family': 'gemma-3', 'model_size': '27B', 'it': 'base', 'plot_name': 'Gemma 3 27B Base', 'color': '#9eae00', 'apply_chat_template': 'base'},

    # Llama
}

# Hardcoded model codes (used if none provided via CLI)
HARDCODED_MODEL_CODES = ['Q25M-7', 'Q3-4', 'Q3-8', 'Q3-14', 'G3-12', 'G3-27']

def launch_screen_session(model_code):
    session_name = f"pid-{model_code}"
    log_file = f"log_pid_comp_{model_code}.txt"

    # Command to activate conda env and run the script
    command = (
        f"source ~/.bashrc && conda activate syn && "
        f"MODEL_CODE={model_code} python /home/p84400019/projects/consciousness-llms/Synergy_Redundancy_LLMs/src/main_pid_computation_prompts.py > {log_file} 2>&1"
    )

    screen_command = ["screen", "-dmS", session_name, "bash", "-c", command]

    try:
        subprocess.run(screen_command, check=True)
        print(f"Launched screen session '{session_name}' for model '{model_code}'")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch screen session for model {model_code}: {e}")

if __name__ == "__main__":
    # Use CLI model codes if provided, otherwise use hardcoded
    model_codes = sys.argv[1:] if len(sys.argv) > 1 else HARDCODED_MODEL_CODES

    print(f"Launching PID jobs for models: {', '.join(model_codes)}\n")
    for code in model_codes:
        launch_screen_session(code)
