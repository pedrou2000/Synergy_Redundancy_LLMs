CACHE_DIR_BITBUCKET = "/vol/bitbucket/pu22/Transformers/" # Bitbucket cache directory
CACHE_DIR_LOCAL = "/homes/pu22/.cache/huggingface/hub" # Local cache directory
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
# MODEL_NAME = "google/gemma-1.1-7b-it"
USE_GPU = True
NUM_HEADS_PER_LAYER = 32

MODIFIED_OUTPUT_ATTENTIONS=False
METRICS_TRANSFORMER = ['attention_weights'] if not MODIFIED_OUTPUT_ATTENTIONS else ['projected_Q', 'attention_weights', 'attention_outputs']
AGGREGRATION_METHODS = ['norm', 'mean', 'entropy', 'max']
ATTENTION_MEASURE = METRICS_TRANSFORMER[0]

# Directories 
PLOTS_DIR = "../plots/"
SAVED_DATA_DIR = "../data/"
TIME_SERIES_DIR = SAVED_DATA_DIR + "time_series/" + MODEL_NAME + "/"
MATRICES_DIR = SAVED_DATA_DIR + "syn_red_matrices/" + MODEL_NAME + "/"

PLOTS_TIME_SERIES_DIR = PLOTS_DIR + "1-Time_Series/" + MODEL_NAME + "/"
PLOTS_SYNERGY_REDUNDANCY_DIR = PLOTS_DIR + "2-Redundancy_Synergy_Matrices/" + MODEL_NAME + "/"
PLOTS_ALL_PHID_DIR = PLOTS_DIR + "3-All_PhiID_Matrices/" + MODEL_NAME + "/"
PLOTS_SYNERGY_REDUNDANCY_GRADIENTS = PLOTS_DIR + "4-Redundancy_Synergy_Gradients/" + MODEL_NAME + "/"
PLOTS_SYNERGY_REDUNDANCY_PER_HEAD = PLOTS_SYNERGY_REDUNDANCY_GRADIENTS + "1-Synergy_Redundancy_per_Head/"
PLOTS_SYNERGY_REDUNDANCY_RANK_GRADIENT = PLOTS_SYNERGY_REDUNDANCY_GRADIENTS + "2-Synergy_Redundancy_Rank_Gradient/"
PLOTS_GRADIENT_PERCENTILE = PLOTS_SYNERGY_REDUNDANCY_GRADIENTS + "3-Gradient_Percentile/"
PLOTS_HEAD_ACTIVATIONS_ANALYSIS = PLOTS_DIR + "5-Head_Activations_Analysis/" + MODEL_NAME + "/"
PLOTS_LDA = PLOTS_DIR + "6-LDA_Head_Activations/" + MODEL_NAME + "/"
PLOTS_ACTIVATIONS_SYN_RED_GRAD = PLOTS_DIR + "7-Activations_vs_Synergy-Redundancy_Rank_Gradient/" + MODEL_NAME + "/"

# Prompts
prompts = {
    "math_operations": [
        "What is the sum of 457 and 674?",
        "Calculate the product of 23 and 89.",
        "If you divide 144 by 12, what do you get?",
        "Subtract 321 from 789 and provide the result.",
        "What is the square of 16?",
        "Find the square root of 256.",
        "If a rectangle has a length of 14 and a width of 5, what is its area?",
        "What is 15 per cent of 200?",
        "How many prime numbers are there between 1 and 20?",
        "If I save $50 each month, how much will I have saved after one year?"
    ],
    "creative_writing": [
        "Describe a world where water is scarce, and every drop counts.",
        "Write a story about a child who discovers they can speak to animals.",
        "Imagine a city that floats in the sky. What does it look like, and how do people live?",
        "Create a tale of a lost civilization hidden beneath the ocean.",
        "Envision a future where humans have merged with technology.",
        "Tell the story of a mystical forest protected by ancient spirits.",
        "Describe a journey to a planet with two suns and three moons.",
        "Write about a character who can manipulate time, but with consequences.",
        "Imagine a world where dreams are currency. How does society function?",
        "Craft a story around a magical artifact that can change the seasons at will."
    ],
    "grammatical_errors": [
        "Correct the error: He go to school every day.",
        "Correct the error: She have two cats and a dogs.",
        "Correct the error: I eats breakfast at 8:00 in the morning.",
        "Correct the error: They is planning a trip to Paris next month.",
        "Correct the error: He don't like going to the gym.",
        "Correct the error: It's raining, but he forget his umbrella at home.",
        "Correct the error: Me and my friend are going to the beach tomorrow.",
        "Correct the error: She's happy because she passed she's exams.",
        "Correct the error: There's many reasons why I didn't attend the meeting.",
        "Correct the error: I can't find my keys. Have you saw them?"
    ]
}

prompts_2 = {
    "syntax_and_grammar_recognition": [
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
    "basic_numerical_reasoning": [
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