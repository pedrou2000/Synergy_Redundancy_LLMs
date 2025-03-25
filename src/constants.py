from transformers import AutoConfig 

# CACHE_DIR_BITBUCKET = "/vol/bitbucket/pu22/Transformers/" # Bitbucket cache directory
# CACHE_DIR_LOCAL = "/homes/pu22/.cache/huggingface/hub" # Local cache directory

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
}
MODEL_CODE = 'L3.1-8B-b'
MODEL_NAME = MODEL_NAMES[MODEL_CODE]["HF_NAME"]
FOLDER_MODEL_NAME = MODEL_NAMES[MODEL_CODE]["FOLDER_NAME"]
FINAL_MODELS = ['G2-2B', 'G2-9B', 'L3.2-3B', 'L3.1-8B']

config = AutoConfig.from_pretrained(MODEL_NAME)
NUM_LAYERS = config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else config.n_layer
NUM_HEADS_PER_LAYER = config.num_attention_heads if hasattr(config, 'num_attention_heads') else config.n_head
NUM_TOTAL_HEADS = NUM_LAYERS * NUM_HEADS_PER_LAYER
MODIFIED_OUTPUT_ATTENTIONS = False
USING_REST_STATE = False

GENERATE_RAW_ATTENTION_AND_TIME_SERIES = True
COMPUTE_PID = False
COMPUTE_GRAPH_THEORETICAL_PROPERTIES = False
GENERATE_ATTENTION_WEIGHTS = False
LOAD_MODEL = GENERATE_ATTENTION_WEIGHTS or COMPUTE_PID or GENERATE_RAW_ATTENTION_AND_TIME_SERIES
SAVE_PLOTS = True

ABLATIONS_RANKING_METHOD = 'syn_minus_red' # 'syn_minus_red', 'synergy', 'redundancy', 'deepest_layers'
METRICS_TRANSFORMER = ['attention_outputs'] if not MODIFIED_OUTPUT_ATTENTIONS else ['queries', 'attention_weights', 'attention_outputs']
if len(METRICS_TRANSFORMER) > 1 and MODEL_CODE in ['G1-2B', 'G1.1-2B', 'G1.1-7B', 'L3-8B']:
    METRICS_TRANSFORMER[0] = 'projected_Q'
AGGREGRATION_METHODS = ['norm', 'mean', 'entropy', 'max']
ATTENTION_MEASURE = METRICS_TRANSFORMER[2 if MODIFIED_OUTPUT_ATTENTIONS else 0]

ATOMS_AVERAGE_VERTICALLY = ['rty', 'sty', 'xty', 'ytr', 'yts', 'yty']
INFORMATION_DYNAMICS = {
    "storage": ["rtr", "xtx", "yty", "sts"],
    "copy": ["xtr", "ytr"],
    "transfer": ["xty", "ytx"],
    "erasure": ["rtx", "rty"],
    "downward_causation": ["stx", "sty", "str"],
    "upward_causation": ["xts", "yts", "rts"],
    "information_storage": ["rtr", "rtx", "xtr", "xtx"],
    "transfer_entropy_x_t_->_y_t+1": ["str", "sty", "xtr", "xty"],
    "causal_density": [(2,"str"), "sty", "xtr", "xty", "stx", "ytx", "ytr"],
    "integrated_information": ["sts", "xts", "yts", "stx", "sty", "xty", "ytx", "rts", "str", (-1, "rtr")], 
    "mutual_information": [
        "rtr", "rtx", "rty", "rts", 
        "str", "stx", "sty", "sts", 
        "xtr", "xtx", "xty", "xts", 
        "ytr", "ytx", "yty", "yts", 
    ]
}

ATOM_NAMES = {
    "rtr": "Red → Red",
    "rtx": "Red → Un1",
    "rty": "Red → Un2",
    "rts": "Red → Syn",
    "str": "Syn → Red",
    "stx": "Syn → Un1",
    "sty": "Syn → Un2",
    "sts": "Syn → Syn",
    "xtr": "Un1 → Red",
    "xtx": "Un1 → Un1",
    "xty": "Un1 → Un2",
    "xts": "Un1 → Syn",
    "ytr": "Un2 → Red",
    "ytx": "Un2 → Un1",
    "yty": "Un2 → Un2",
    "yts": "Un2 → Syn",
}


# Directories 
PLOTS_DIR = "../plots/" + FOLDER_MODEL_NAME + "/"
SAVED_DATA_DIR = "../data/" + FOLDER_MODEL_NAME + "/"

RAW_ATTENTION_DIR = SAVED_DATA_DIR + "1-Raw_Attention/"
TIME_SERIES_DIR = SAVED_DATA_DIR + "2-Time_Series/"
MATRICES_DIR = SAVED_DATA_DIR + "3-Synergy_Redundancy_Matrices/"
ATTENTION_WEIGHTS_DIR = SAVED_DATA_DIR + "4-Attention_Weights_Prompts/"
GRAPH_METRICS_DIR = SAVED_DATA_DIR + "5-Graph_Theoretical_Properties/"
ABLATIONS_DIR = SAVED_DATA_DIR + "6-Ablations/"

PLOTS_TIME_SERIES_DIR = PLOTS_DIR + "1-Time_Series/"
PLOTS_SYNERGY_REDUNDANCY_DIR = PLOTS_DIR + "2-Synergy_Redundancy/"
PLOTS_HEAD_ACTIVATIONS_COGNITIVE_TASKS = PLOTS_DIR + "3-Head_Activations_Cognitive_Tasks/"
PLOT_SYNERGY_REDUNDANCY_TASK_CORRELATIONS = PLOTS_DIR + "4-Synergy-Redundancy_vs_Cognitive_Task_Correlations/"
PLOT_ABLATIONS = PLOTS_DIR + "5-Ablations/" + ATTENTION_MEASURE + "/"

MODEL_COMPARISON_DIR = "../plots/" + "0-Model_Comparison/"
MODEL_COMPARISON_GRAPH_THEORETICAL_DIR = MODEL_COMPARISON_DIR + "1-Graph_Theoretical_Analysis/"
MODEL_COMPARISON_GRADIENT_RANK_DIR = MODEL_COMPARISON_DIR + "2-Gradient_Ranks/"
MODEL_COMPARISON_ABLATIONS_DIR = MODEL_COMPARISON_DIR + "3-Ablations/"

# Prompts
PROMPTS = {
    "simple_maths": [
        "If you have 15 apples and you give away 5, how many do you have left?",
        "A rectangle's length is twice its width. If the rectangle's perimeter is 36 meters, what are its length and width?",
        "You read 45 pages of a book each day. How many pages will you have read after 7 days?",
        "If a train travels 60 miles in 1 hour, how far will it travel in 3 hours?",
        "There are 8 slices in a pizza. If you eat 2 slices, what fraction of the pizza is left?",
        "If one pencil costs 50 cents, how much do 12 pencils cost?",
        "You have a 2-liter bottle of soda. If you pour out 500 milliliters, how much soda is left?",
        "A marathon is 42 kilometers long. If you have run 10 kilometers, how much further do you have to run?",
        "If you divide 24 by 3, then multiply by 2, what is the result?",
        "A car travels 150 miles on 10 gallons of gas. How many miles per gallon does the car get?"
    ],
    "syntax_and_grammar_correction": [
        "Correct the error: He go to school every day.",
        "Correct the error: She have two cats and a dogs.",
        "Correct the error: I eats breakfast at 8:00 in the morning.",
        "Correct the error: Every students in the classroom has their own laptop.",
        "Correct the error: She don't like going to the park on weekends.",
        "Correct the error: We was happy to see the rainbow after the storm.",
        "Correct the error: There is many reasons to celebrate today.",
        "Correct the error: Him and I went to the market yesterday.",
        "Correct the error: The books is on the table.",
        "Correct the error: They walks to school together every morning."
    ],
    "part_of_speech_tagging": [
        "Identify the parts of speech in the sentence: Quickly, the agile cat climbed the tall tree.",
        "Identify the parts of speech in the sentence: She whispered a secret to her friend during the boring lecture.",
        "Identify the parts of speech in the sentence: The sun sets in the west.",
        "Identify the parts of speech in the sentence: Can you believe this amazing view?",
        "Identify the parts of speech in the sentence: He quickly finished his homework.",
        "Identify the parts of speech in the sentence: The beautifully decorated cake was a sight to behold.",
        "Identify the parts of speech in the sentence: They will travel to Japan next month.",
        "Identify the parts of speech in the sentence: My favorite book was lost.",
        "Identify the parts of speech in the sentence: The loud music could be heard from miles away.",
        "Identify the parts of speech in the sentence: She sold all of her paintings at the art show."
    ], 
    "basic_common_sense_reasoning": [
        "If it starts raining while the sun is shining, what weather phenomenon might you expect to see?",
        "Why do people wear sunglasses?",
        "What might you use to write on a chalkboard?",
        "Why would you put a letter in an envelope?",
        "If you're cold, what might you do to get warm?",
        "What is the purpose of a refrigerator?",
        "Why might someone plant a tree?",
        "What happens to ice when it's left out in the sun?",
        "Why do people shake hands when they meet?",
        "What can you use to measure the length of a desk?"
    ],
    "abstract_reasoning_and_creative_thinking": [
        "Imagine a future where humans have evolved to live underwater. Describe the adaptations they might develop.",
        "Invent a sport that could be played on Mars considering its lower gravity compared to Earth. Describe the rules.",
        "Describe a world where water is scarce, and every drop counts.",
        "Write a story about a child who discovers they can speak to animals.",
        "Imagine a city that floats in the sky. What does it look like, and how do people live?",
        "Create a dialogue between a human and an alien meeting for the first time.",
        "Design a vehicle that can travel on land, water, and air. Describe its features.",
        "Imagine a new holiday and explain how people celebrate it.",
        "Write a poem about a journey through a desert.",
        "Describe a device that allows you to experience other people's dreams."
    ],
    "emotional_intelligence_and_social_cognition": [
        "Write a dialogue between two characters where one comforts the other after a loss, demonstrating empathy.",
        "Describe a situation where someone misinterprets a friend's actions as hostile, and how they resolve the misunderstanding.",
        "Compose a letter from a character apologizing for a mistake they made.",
        "Describe a scene where a character realizes they are in love.",
        "Write a conversation between two old friends who haven't seen each other in years.",
        "Imagine a character facing a moral dilemma. What do they choose and why?",
        "Describe a character who is trying to make amends for past actions.",
        "Write about a character who overcomes a fear with the help of a friend.",
        "Create a story about a misunderstanding between characters from different cultures.",
        "Imagine a scenario where a character has to forgive someone who wronged them."
    ]
}

PROMPT_CATEGORIES = list(PROMPTS.keys())
RESTING_STATE_CATEGORY = "resting_state"
PROMPT_CATEGORIES.append(RESTING_STATE_CATEGORY) if USING_REST_STATE else None


def update_model_code(new_model_code):
    """Update MODEL_CODE and all dependent variables."""
    global MODEL_CODE, MODEL_NAME, FOLDER_MODEL_NAME, NUM_LAYERS, NUM_HEADS_PER_LAYER, NUM_TOTAL_HEADS
    global PLOTS_DIR, SAVED_DATA_DIR, RAW_ATTENTION_DIR, TIME_SERIES_DIR, MATRICES_DIR, ATTENTION_WEIGHTS_DIR
    global GRAPH_METRICS_DIR, ABLATIONS_DIR, PLOTS_TIME_SERIES_DIR, PLOTS_SYNERGY_REDUNDANCY_DIR
    global PLOTS_HEAD_ACTIVATIONS_COGNITIVE_TASKS, PLOT_SYNERGY_REDUNDANCY_TASK_CORRELATIONS, PLOT_ABLATIONS

    if new_model_code not in MODEL_NAMES:
        raise ValueError(f"Invalid model code: {new_model_code}. Available options: {list(MODEL_NAMES.keys())}")

    MODEL_CODE = new_model_code
    MODEL_NAME = MODEL_NAMES[MODEL_CODE]["HF_NAME"]
    FOLDER_MODEL_NAME = MODEL_NAMES[MODEL_CODE]["FOLDER_NAME"]

    config = AutoConfig.from_pretrained(MODEL_NAME)
    NUM_LAYERS = config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else config.n_layer
    NUM_HEADS_PER_LAYER = config.num_attention_heads if hasattr(config, 'num_attention_heads') else config.n_head
    NUM_TOTAL_HEADS = NUM_LAYERS * NUM_HEADS_PER_LAYER

    # Update directories
    PLOTS_DIR = "../plots/" + FOLDER_MODEL_NAME + "/"
    SAVED_DATA_DIR = "../data/" + FOLDER_MODEL_NAME + "/"
    
    RAW_ATTENTION_DIR = SAVED_DATA_DIR + "1-Raw_Attention/"
    TIME_SERIES_DIR = SAVED_DATA_DIR + "2-Time_Series/"
    MATRICES_DIR = SAVED_DATA_DIR + "3-Synergy_Redundancy_Matrices/"
    ATTENTION_WEIGHTS_DIR = SAVED_DATA_DIR + "4-Attention_Weights_Prompts/"
    GRAPH_METRICS_DIR = SAVED_DATA_DIR + "5-Graph_Theoretical_Properties/"
    ABLATIONS_DIR = SAVED_DATA_DIR + "6-Ablations/"

    PLOTS_TIME_SERIES_DIR = PLOTS_DIR + "1-Time_Series/"
    PLOTS_SYNERGY_REDUNDANCY_DIR = PLOTS_DIR + "2-Synergy_Redundancy/"
    PLOTS_HEAD_ACTIVATIONS_COGNITIVE_TASKS = PLOTS_DIR + "3-Head_Activations_Cognitive_Tasks/"
    PLOT_SYNERGY_REDUNDANCY_TASK_CORRELATIONS = PLOTS_DIR + "4-Synergy-Redundancy_vs_Cognitive_Task_Correlations/"
    PLOT_ABLATIONS = PLOTS_DIR + "5-Ablations/" + ATTENTION_MEASURE + "/"

    print(f"Updated to model: {MODEL_CODE}")

