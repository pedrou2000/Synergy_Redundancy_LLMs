from time_series_generation import *
from phid import *
from cognitive_tasks_analysis import *
from cognitive_tasks_vs_syn_red_analysis import *
from lda import *
from random_walk_time_series import *
from hf_token import TOKEN

from huggingface_hub import login
from transformers import AutoTokenizer, AutoConfig 
import seaborn as sns
import matplotlib.pyplot as plt

import ablation_studies
import utils
import random, os, json
import matplotlib.pyplot as plt



device = torch.device("cuda")
login(token = TOKEN)
attn_implementation="eager" # GEMMA_ATTENTION_CLASSES = {"eager": GemmaAttention, "flash_attention_2": GemmaFlashAttention2, "sdpa": GemmaSdpaAttention,}


print(f'Status: Starting main ablations...')
print(f'GPU memory allocated at the beginning of main ablations: {torch.cuda.memory_allocated(device)}')

# Load the configuration and modify it
model_config = AutoConfig.from_pretrained(constants.MODEL_NAME)#, cache_dir=constants.CACHE_DIR_BITBUCKET)
model_config._attn_implementation = attn_implementation  # Custom attention parameter

# Load the tokenizer and model with the modified configuration
tokenizer = AutoTokenizer.from_pretrained(constants.MODEL_NAME)#, cache_dir=constants.CACHE_DIR_BITBUCKET)
model = AutoModelForCausalLM.from_pretrained(
    constants.MODEL_NAME,
    # cache_dir=constants.CACHE_DIR_BITBUCKET,
    device_map='auto',
    attn_implementation=attn_implementation, # Make sure to use the adequate attention layer in order to 
    config=model_config,  # Use the modified config
)
print(f'GPU memory allocated after loading the model: {torch.cuda.memory_allocated(device)}')
print(f'GPU memory remaining after loading the model: {torch.cuda.memory_reserved(device)}')

model.eval()
print("Attention Measure: ", constants.ATTENTION_MEASURE)
print("Loaded Model Name: ", model.config.name_or_path)
print("Model: ", model)
print("Attention Layers Implementation: ", model.config._attn_implementation)
print(f"Number of layers: {constants.NUM_LAYERS}")
print(f"Number of attention heads per layer: {constants.NUM_HEADS_PER_LAYER}")
print(f'GPU memory allocated after loading the model: {torch.cuda.memory_allocated(device)}')


##### Get the Ablation Order #####
prompt_category_name = 'average_prompts'
global_matrices, synergy_matrices, redundancy_matrices = load_matrices(base_save_path=constants.MATRICES_DIR + prompt_category_name + '/' + prompt_category_name + '.pt')
averages = calculate_average_synergy_redundancies_per_head(synergy_matrices, redundancy_matrices, within_layer=False)
gradient_ranks = compute_gradient_rank(averages)

### Gradient Ranks ###
gradient_ranks = gradient_ranks[constants.ATTENTION_MEASURE]
sorted_heads_by_rank = sorted(gradient_ranks.items(), key=lambda x: x[1], reverse=True)
syn_minus_red_ranked_heads = [utils.get_layer_and_head(head_num) for head_num, _ in sorted_heads_by_rank]

### Synergy and Redundancy Ranks ###
synergies, redundancies = averages[constants.ATTENTION_MEASURE]["synergy"], averages[constants.ATTENTION_MEASURE]["redundancy"]

synergies_dict = {i+1: synergies[i] for i in range(len(synergies))}
redundancies_dict = {i+1: redundancies[i] for i in range(len(redundancies))}

synergy_ranks = {i: rank for rank, (i, _) in enumerate(sorted(synergies_dict.items(), key=lambda x: x[1], reverse=True))}
redundancy_ranks = {i: rank for rank, (i, _) in enumerate(sorted(redundancies_dict.items(), key=lambda x: x[1], reverse=True))}

sorted_heads_by_synergy = sorted(synergies_dict.items(), key=lambda x: x[1], reverse=True)
sorted_heads_by_redundancy = sorted(redundancies_dict.items(), key=lambda x: x[1], reverse=True)

synergy_ranked_heads = [utils.get_layer_and_head(head_num) for head_num, _ in sorted_heads_by_synergy]
redundancy_ranked_heads = [utils.get_layer_and_head(head_num) for head_num, _ in sorted_heads_by_redundancy]

head_rankings = {"synergy": synergy_ranked_heads, "redundancy": redundancy_ranked_heads, "syn_minus_red": syn_minus_red_ranked_heads}
# print(head_rankings)




##### Perform the Ablation Studies #####

# Print gpu memory
print(f'GPU memory allocated at the beginning of ablation studies: {torch.cuda.memory_allocated(device)}')

import random, os, pickle
import matplotlib.pyplot as plt
import torch  # Ensure torch is imported

num_tokens_to_generate = 100
ablation_ranking_method = constants.ABLATIONS_RANKING_METHOD
num_random_ablations = 5  # Number of iterations to repeat the random ablation process
num_total_ablation_steps = 30 # Total number of ablation data points to collect
num_heads_skip_per_iteration = constants.NUM_TOTAL_HEADS // num_total_ablation_steps # Number of heads to ablate per iteration
temperature = 0

# The resulting divergence trajectories for random and gradient rank-based ablations for each prompt
divergence_results = {
    "divergences": {
        prompt_category_name: {
            prompt_num: {'random': [], 'gradient': [], 'random_texts': [], 'gradient_text': []}
            for prompt_num in range(len(constants.PROMPTS[prompt_category_name]))
        }
        for prompt_category_name in constants.PROMPTS.keys()
    },
    "list_heads_ablated": [i for i in range(0, constants.NUM_TOTAL_HEADS, num_heads_skip_per_iteration)],
}

print(f'GPU memory allocated just before loop ablation studies: {torch.cuda.memory_allocated(device)}')
n_prompts_per_category = len(constants.PROMPTS[list(constants.PROMPTS.keys())[0]])
for prompt_category_name, prompt_list in constants.PROMPTS.items():#[('simple_maths', constants.PROMPTS['simple_maths'])]:
    print(f"\n---\nStarting ablation studies for prompt category: {prompt_category_name}\n---\n")

    ##### Run the Non-Ablated Model #####
    print(f'Running the non-ablated model for prompt category: {prompt_category_name}')
    model.reset_ablated_heads()
    model.print_ablated_heads()

    non_ablated_texts, non_ablated_logits, non_ablated_token_ids = ablation_studies.generate_text_with_logits_batch(
        model, 
        tokenizer, 
        prompts=prompt_list,
        num_tokens_to_generate=num_tokens_to_generate,
        device=device, 
        temperature=temperature
    )

    # Move tensors to CPU if not needed on GPU
    non_ablated_logits = non_ablated_logits.cpu()
    non_ablated_token_ids = non_ablated_token_ids.cpu()
    torch.cuda.empty_cache()

    print(f'Generated non-ablated text for prompt category: {prompt_category_name}: \n{non_ablated_texts[0]}')

    ##### Run the Rank-Based Ablation #####
    # Loop over the number of steps required to ablate all heads
    for i in range(0, constants.NUM_TOTAL_HEADS, num_heads_skip_per_iteration):
        print(f'Running gradient rank-based ablation for prompt category: {prompt_category_name}, number of ablated heads: {i}')
        # Get the current list of heads to ablate
        current_heads_to_ablated = head_rankings[ablation_ranking_method][:i + num_heads_skip_per_iteration]

        # Initialize the ablated_attention_heads dictionary
        ablated_attention_heads = {}
        for layer, head in current_heads_to_ablated:
            if layer not in ablated_attention_heads:
                ablated_attention_heads[layer] = []
            ablated_attention_heads[layer].append(head)

        # Generate the output with the current ablated heads
        ablated_logits_tensor, ablated_divergence_tensor, ablated_texts = ablation_studies.generate_with_teacher_forcing_ablated_batch(
            model,
            tokenizer,
            prompts=prompt_list,
            non_ablated_token_ids=non_ablated_token_ids.to(device),
            non_ablated_logits=non_ablated_logits.to(device),
            device=device,
            temperature=temperature,
            ablated_attention_heads=ablated_attention_heads,
            verbose=False
        )


        # Move divergence tensor to CPU
        ablated_divergence_tensor = ablated_divergence_tensor.cpu()
        torch.cuda.empty_cache()

        print(f'Generated ablated text for prompt category: {prompt_category_name}, number of ablated heads: {i}: \n{ablated_texts[0]}')

        # Record the sum of divergences
        for prompt_num, divergence in enumerate(ablated_divergence_tensor):
            divergence = divergence.mean().item()
            divergence_results["divergences"][prompt_category_name][prompt_num]['gradient'].append(divergence)
            divergence_results["divergences"][prompt_category_name][prompt_num]['gradient_text'].append(ablated_texts[prompt_num])

        # Delete tensors and clear cache
        del ablated_logits_tensor, ablated_divergence_tensor, ablated_texts
        torch.cuda.empty_cache()

    ##### Run the Random Ablations #####
    for iteration in range(num_random_ablations):
        print(f'Running random ablation iteration {iteration + 1} for prompt category: {prompt_category_name}')
        # Generate a list of all attention heads (layer, head)
        attention_heads_list = [(layer, head) for layer in range(constants.NUM_LAYERS) for head in range(constants.NUM_HEADS_PER_LAYER)]

        # Randomly shuffle the list to determine the ablation order
        random.shuffle(attention_heads_list)

        # Initialize list to record divergences for this iteration
        divergence_dict = {prompt_num: [] for prompt_num in range(len(prompt_list))}
        text_dict = {prompt_num: [] for prompt_num in range(len(prompt_list))}

        # Loop over the number of steps required to ablate all heads
        for i in range(0, constants.NUM_TOTAL_HEADS, num_heads_skip_per_iteration):
            print(f'Running random ablation iteration {iteration + 1}, number of ablated heads: {i} for prompt category: {prompt_category_name}')
            # Get the current list of heads to ablate
            current_heads_to_ablated = attention_heads_list[:i + num_heads_skip_per_iteration]

            # Initialize the ablated_attention_heads dictionary
            ablated_attention_heads = {}
            for layer, head in current_heads_to_ablated:
                if layer not in ablated_attention_heads:
                    ablated_attention_heads[layer] = []
                ablated_attention_heads[layer].append(head)

            # Generate the output with the current ablated heads
            ablated_logits, ablated_divergences, generated_text = ablation_studies.generate_with_teacher_forcing_ablated_batch(
                model,
                tokenizer,
                prompts=prompt_list,
                non_ablated_token_ids=non_ablated_token_ids.to(device),
                non_ablated_logits=non_ablated_logits.to(device),
                device=device,
                temperature=temperature,
                ablated_attention_heads=ablated_attention_heads,
                verbose=False
            )

            # Move divergence tensor to CPU
            ablated_divergences = ablated_divergences.cpu()
            torch.cuda.empty_cache()

            for prompt_num, divergence in enumerate(ablated_divergences):
                divergence_dict[prompt_num].append(divergence.mean().item())
                text_dict[prompt_num].append(generated_text[prompt_num])

            # Delete tensors and clear cache
            del ablated_logits, ablated_divergences, generated_text
            torch.cuda.empty_cache()

        for prompt_number in range(len(prompt_list)):
            divergence_results["divergences"][prompt_category_name][prompt_number]['random'].append(divergence_dict[prompt_number])
            divergence_results["divergences"][prompt_category_name][prompt_number]['random_texts'].append(text_dict[prompt_number])

    
    print(f'Completed ablation studies for prompt category: {prompt_category_name}. Saving results...')
    ##### Save the results to a pickle file #####
    save_dir = constants.ABLATIONS_DIR + ablation_ranking_method + '/'
    os.makedirs(save_dir, exist_ok=True)
    # Save to a pickle file
    with open(save_dir + 'divergence_results.pkl', 'wb') as f:  
        pickle.dump(divergence_results, f)

    ##### PLOTTING #####
    for prompt_num in range(len(prompt_list)):
        # Plot the divergence trajectories for random ablations and gradient rank-based ablation
        plt.figure(figsize=(10, 6))

        # Plot random ablations
        for idx, divergence_trajectory in enumerate(divergence_results["divergences"][prompt_category_name][prompt_num]['random']):
            plt.plot(divergence_results["list_heads_ablated"], divergence_trajectory, marker='o', color='blue', alpha=0.5,
                    label='Random Ablations' if idx == 0 else "")

        # Plot gradient rank-based ablation
        plt.plot(divergence_results["list_heads_ablated"], divergence_results["divergences"][prompt_category_name][prompt_num]['gradient'], marker='o', color='red', label='Gradient Rank Ablations')

        plt.xlabel('Number of Heads Ablated')
        plt.ylabel('Divergence Mean')
        plt.title('Divergence vs. Number of Ablated Attention Heads')
        plt.legend()
        plt.grid(True)
        save_dir = constants.PLOT_ABLATIONS + ablation_ranking_method + '/' +  '1-Divergence_Plots/' + prompt_category_name + '/' 
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir + str(prompt_num) + '_random_vs_gradient_rank_ablations_divergence_trajectories.png')
        plt.close()  # Ensure that the plot is closed to free memory

    # Delete variables and clear cache at the end of each prompt category
    del non_ablated_texts, non_ablated_logits, non_ablated_token_ids
    torch.cuda.empty_cache()

print(f'Completed all ablation studies. Results saved to {save_dir}')
print(f'Plotting divergence AUCs and statistical tests...')

##### Plot Ablation Studies #####
saved_divergence_results = pickle.load(open(constants.ABLATIONS_DIR + constants.ABLATIONS_RANKING_METHOD + '/divergence_results.pkl', 'rb'))
from scipy.integrate import trapezoid


### Compute AUCs ###
# Initialize dictionaries to store AUCs
auc_results = {}

# Iterate over each prompt category
for prompt_category, prompts in saved_divergence_results["divergences"].items():
    auc_results[prompt_category] = {
        'synergy': [],
        'random': []
    }
    x_values = saved_divergence_results["list_heads_ablated"]

    # Iterate over each prompt in the category
    for prompt_num, prompt_data in prompts.items():
        # Number of heads ablated at each step
        y_synergy = prompt_data["gradient"]

        # Compute AUC for synergy
        auc_synergy = trapezoid(y_synergy, x_values)
        auc_results[prompt_category]['synergy'].append(auc_synergy)

        # Random ablations (5 trials)
        auc_random_trials = []
        for random_trial in prompt_data["random"]:
            y_random = random_trial
            # Compute AUC for each random trial
            auc_random = trapezoid(y_random, x_values)
            auc_random_trials.append(auc_random)
        auc_results[prompt_category]['random'].append(auc_random_trials)
mean_auc_results = {}

for prompt_category, data in auc_results.items():
    # Synergy AUCs
    synergy_aucs = data['synergy']
    mean_synergy_auc = np.mean(synergy_aucs)

    # Random AUCs (list of lists)
    random_aucs = data['random']
    # Flatten the list of lists and compute mean per trial
    random_aucs_flat = [auc for sublist in random_aucs for auc in sublist]
    mean_random_auc = np.mean(random_aucs_flat)

    mean_auc_results[prompt_category] = {
        'mean_synergy_auc': mean_synergy_auc,
        'mean_random_auc': mean_random_auc,
        'synergy_aucs': synergy_aucs,
        'random_aucs': random_aucs_flat
    }


### Statistical Tests ###
# t-test for each prompt category
from scipy.stats import ttest_rel

p_values = {}

for prompt_category, data in auc_results.items():
    synergy_aucs = data['synergy']
    random_aucs = data['random']  # List of lists

    # Compute mean random AUC for each prompt
    mean_random_aucs_per_prompt = [np.mean(trials) for trials in random_aucs]

    # Perform paired t-test
    t_stat, p_value = ttest_rel(synergy_aucs, mean_random_aucs_per_prompt)

    p_values[prompt_category] = {
        't_stat': t_stat,
        'p_value': p_value
    }
    print(f"Prompt Category: {prompt_category}")
    print(f"t-statistic: {t_stat:.4f}, p-value: {p_value}\n")

# Effect Size (Cohen's d)
def cohen_d(x, y):
    # Compute Cohen's d for paired samples
    diff = np.array(x) - np.array(y)
    return np.mean(diff) / np.std(diff, ddof=1)

effect_sizes = {}

for prompt_category, data in auc_results.items():
    synergy_aucs = data['synergy']
    mean_random_aucs_per_prompt = [np.mean(trials) for trials in data['random']]

    d = cohen_d(synergy_aucs, mean_random_aucs_per_prompt)
    effect_sizes[prompt_category] = d
    print(f"Prompt Category: {prompt_category}")
    print(f"Cohen's d: {d:.4f}\n")

#### Are Prompt Categories Meaningful? ANOVA and Tukey's Honest Significance Test

import pandas as pd

# Prepare data for ANOVA
anova_data = []

for prompt_category, data in auc_results.items():
    # Synergy data
    for auc in data['synergy']:
        anova_data.append({
            'Prompt_Category': prompt_category,
            'Ablation_Type': 'Synergy',
            'AUC': auc
        })
    # Random data
    for auc_list in data['random']:
        for auc in auc_list:
            anova_data.append({
                'Prompt_Category': prompt_category,
                'Ablation_Type': 'Random',
                'AUC': auc
            })

df_anova = pd.DataFrame(anova_data)

import statsmodels.api as sm
from statsmodels.formula.api import ols

# Build the model
model = ols('AUC ~ C(Ablation_Type) * C(Prompt_Category)', data=df_anova).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np

# Assuming df_anova is your dataframe with 'Prompt_Category' and 'AUC'
# Perform Tukey's HSD test
tukey = pairwise_tukeyhsd(endog=df_anova['AUC'], groups=df_anova['Prompt_Category'], alpha=0.05)

# Convert the Tukey results into a DataFrame for easier plotting
tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])

# Rename columns for convenience
tukey_df.columns = ['group1', 'group2', 'meandiff', 'p-adj', 'lower', 'upper', 'reject']

# Sort the DataFrame by mean difference for better plotting
tukey_df = tukey_df.sort_values('meandiff')

# Create a new column to combine group1 and group2 for labeling
tukey_df['Comparison'] = tukey_df['group1'] + ' vs ' + tukey_df['group2']

# --- Part 1: Tukey HSD Bar Plot Visualization ---

# Set up the figure for the bar plot
plt.figure(figsize=(10, 6))

# Create a horizontal bar plot for mean differences
sns.barplot(x='meandiff', y='Comparison', data=tukey_df, palette='vlag')

# Add the confidence intervals as error bars
for i in range(tukey_df.shape[0]):
    plt.plot([tukey_df['lower'].iloc[i], tukey_df['upper'].iloc[i]], [i, i], color='black', lw=1)

# Mark significant comparisons
for i in range(tukey_df.shape[0]):
    if tukey_df['reject'].iloc[i]:
        plt.scatter(tukey_df['meandiff'].iloc[i], i, color='red', s=50, zorder=3)

# Customize the plot
plt.axvline(x=0, color='grey', linestyle='--', label='No Difference')
plt.title('Tukey HSD Post-Hoc Test Results', fontsize=16)
plt.xlabel('Mean Difference in AUC', fontsize=12)
plt.ylabel('Prompt Category Comparisons', fontsize=12)
plt.tight_layout()


# Save the plot into a file
save_dir = constants.PLOT_ABLATIONS + constants.ABLATIONS_RANKING_METHOD + '/' + '4-Meaningfulness_of_Prompt_Categories/'
os.makedirs(save_dir, exist_ok=True)
plt.savefig(save_dir + '1-Turkey_HSD_Bar_Plot.png')

# Show the plot
plt.show()

# --- Part 2: Significance Matrix (Heatmap) ---

# Get unique prompt categories
categories = df_anova['Prompt_Category'].unique()

# Create an empty matrix to store the significance values (True or False)
significance_matrix = np.zeros((len(categories), len(categories)), dtype=bool)

# Populate the matrix based on Tukey HSD results
for i, group1 in enumerate(categories):
    for j, group2 in enumerate(categories):
        if group1 != group2:
            # Check if group1 vs group2 exists in Tukey results (in either order)
            row = tukey_df[(tukey_df['group1'] == group1) & (tukey_df['group2'] == group2) |
                           (tukey_df['group1'] == group2) & (tukey_df['group2'] == group1)]
            if not row.empty:
                significance_matrix[i, j] = row['reject'].values[0]

# Create a heatmap to visualize the significance matrix
plt.figure(figsize=(8, 6))
sns.heatmap(significance_matrix, annot=True, fmt="d", cmap="coolwarm", xticklabels=categories, yticklabels=categories, cbar=False)

# Customize the heatmap
plt.title('Significance Matrix (Tukey HSD)', fontsize=16)
plt.xlabel('Prompt Category')
plt.ylabel('Prompt Category')

# Rotate the x labels to 45 degrees (you can adjust this angle)
# plt.xticks(rotation=45)

plt.tight_layout()

# Save the figure
plt.savefig(save_dir + '2-Significance_Matrix_Heatmap.png')

# Show the plot
plt.show()



### Bootstrap Resampling for Confidence Intervals ###
def bootstrap_mean_diff(x, y, n_bootstrap=1000):
    observed_diff = np.mean(x) - np.mean(y)
    boot_diffs = []
    combined = np.concatenate([x, y])
    n = len(x)
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(combined, size=2*n, replace=True)
        boot_x = boot_sample[:n]
        boot_y = boot_sample[n:]
        boot_diff = np.mean(boot_x) - np.mean(boot_y)
        boot_diffs.append(boot_diff)
    return observed_diff, np.percentile(boot_diffs, [2.5, 97.5])

bootstrap_results = {}

for prompt_category, data in auc_results.items():
    synergy_aucs = data['synergy']
    # Flatten random AUCs
    random_aucs = [auc for sublist in data['random'] for auc in sublist]

    observed_diff, ci = bootstrap_mean_diff(synergy_aucs, random_aucs)
    bootstrap_results[prompt_category] = {
        'observed_diff': observed_diff,
        'confidence_interval': ci
    }
    print(f"Prompt Category: {prompt_category}")
    print(f"Observed Difference in Mean AUC: {observed_diff:.4f}")
    print(f"95% Confidence Interval: [{ci[0]:.4f}, {ci[1]:.4f}]\n")

import numpy as np
import matplotlib.pyplot as plt

# Create a larger plot to compare all prompt categories
prompt_categories = list(mean_auc_results.keys())
num_categories = len(prompt_categories)

# Create arrays for mean AUC values and standard deviations
mean_synergy_aucs = [mean_auc_results[cat]['mean_synergy_auc'] for cat in prompt_categories]
mean_random_aucs = [mean_auc_results[cat]['mean_random_auc'] for cat in prompt_categories]
std_synergy_aucs = [np.std(mean_auc_results[cat]['synergy_aucs']) for cat in prompt_categories]
std_random_aucs = [np.std(mean_auc_results[cat]['random_aucs']) for cat in prompt_categories]

# Bar width
bar_width = 0.35
index = np.arange(num_categories)

# Create a larger figure
plt.figure(figsize=(12, 6))

# Plot bars for synergy and random ablations
bars_synergy = plt.bar(index, mean_synergy_aucs, bar_width, yerr=std_synergy_aucs,
                       label='Gradient Rank', capsize=5, color='blue')

bars_random = plt.bar(index + bar_width, mean_random_aucs, bar_width, yerr=std_random_aucs,
                      label='Random', capsize=5, color='green')

# Add labels and title
plt.xlabel('Prompt Category')
plt.ylabel('Mean AUC of KL Divergence')
plt.title('KL Divergence AUC Comparison Across Prompt Categories')

# Set x-axis tick labels
plt.xticks(index + bar_width / 2, prompt_categories, rotation=10)

# Add legend
plt.legend()

# Annotate p-values above the bars
for i, prompt_category in enumerate(prompt_categories):
    p_value = p_values[prompt_category]['p_value']
    p_value_smaller_than_alpha = p_value < 0.001
    p_value_print = f"p-value < 0.001" if p_value_smaller_than_alpha else f"p-value = {p_value:.4f}"
    
    # Add p-value annotations above the bars for synergy
    # plt.text(i, mean_synergy_aucs[i] + std_synergy_aucs[i] + 0.02, p_value_print,
    #          ha='center', fontsize=10, color='red')

# Show the final plot
plt.tight_layout()

# Save the plot into a file
save_dir = constants.PLOT_ABLATIONS + constants.ABLATIONS_RANKING_METHOD + '/' + '2-KL_Divergence_AUC_Comparison_Across_Prompt_Categories/' 
os.makedirs(save_dir, exist_ok=True)
plt.savefig(save_dir + 'KL_Divergence_AUC_Comparison_Across_Prompt_Categories.png')

plt.show()
import matplotlib.pyplot as plt
import os

# Assuming we already have `mean_auc_results` and `p_values` from previous steps
for prompt_category, data in mean_auc_results.items():
    mean_synergy_auc = data['mean_synergy_auc']
    mean_random_auc = data['mean_random_auc']
    std_synergy = np.std(data['synergy_aucs'])
    std_random = np.std(data['random_aucs'])

    # Get p-value from t-test results
    p_value = p_values[prompt_category]['p_value']

    # Create bar plot
    plt.figure(figsize=(6, 4))
    plt.bar(['Synergy', 'Random'], [mean_synergy_auc, mean_random_auc],
            yerr=[std_synergy, std_random], capsize=5, color=['blue', 'green'])
    
    # Add labels and title
    plt.ylabel('Mean AUC of KL Divergence')
    
    # Add the p-value in the title or as an annotation
    p_value_smaller_than_alpha = p_value < 0.001
    p_value_print = f"p-value < 0.001" if p_value_smaller_than_alpha else f"p-value = {p_value: .4f}"
    plt.title(f'AUC Comparison for {prompt_category} ({p_value_print})', pad=20)  # Increase padding for the title
    
    # Adjust layout to prevent cutoff of titles, labels, etc.
    plt.tight_layout()

    # Save the plot with bbox_inches='tight' to ensure all content is included
    save_dir = constants.PLOT_ABLATIONS + constants.ABLATIONS_RANKING_METHOD + '/' + '3-KL_Divergence_AUC_Comparison_Per_Prompt_Category/' 
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_dir + prompt_category + '.png', bbox_inches='tight')  # Use bbox_inches='tight' to ensure nothing is cut off

    # Show plot
    plt.show()


print(f'Completed plotting divergence AUCs and statistical tests. Results saved to {save_dir}')