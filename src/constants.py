CACHE_DIR = "/vol/bitbucket/pu22/Transformers/"
MODEL_NAME = "google/gemma-2b-it"
USE_GPU = True

METRICS_TRANSFORMER = ['projected_Q', 'attention_weights', 'attention_outputs']
AGGREGRATION_METHODS = ['norm', 'mean', 'entropy', 'max']

# Directories 
PLOTS_DIR = "../plots/"
TIME_SERIES_DIR = "../time_series/"

PLOTS_TIME_SERIES_DIR = PLOTS_DIR + "1-Time_Series/"
PLOTS_SYNERGY_REDUNDANCY_DIR = PLOTS_DIR + "2-Redundancy_Synergy_Matrices/"
PLOTS_ALL_PHID_DIR = PLOTS_DIR + "3-All_PhiID_Matrices/"


# Prompts
# Simple Mathematical Operations Prompts
math_operations_prompts = [
    "What is the sum of 457 and 674?",
    "Calculate the product of 23 and 89.",
    "If you divide 144 by 12, what do you get?",
    "Subtract 321 from 789 and provide the result.",
    "What is the square of 16?",
    "Find the square root of 256.",
    "If a rectangle has a length of 14 and a width of 5, what is its area?",
    "What is 15% of 200?",
    "How many prime numbers are there between 1 and 20?",
    "If I save $50 each month, how much will I have saved after one year?"
]

# Very Creative Writing Prompts
creative_writing_prompts = [
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
]

grammatical_error_prompts = [
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

