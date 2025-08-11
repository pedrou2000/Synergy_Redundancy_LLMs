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


# ### Random Walk Time Series ###

# n_steps = 100  # Number of time steps
# n_dim = (constants.NUM_LAYERS, constants.NUM_HEADS_PER_LAYER)  # Shape of the state vector
# scale_factor = 0.9  # Scaling factor for stability
# wishart_df = np.prod(n_dim) + 1  # Degrees of freedom for the Wishart distribution
# seed = None  # Seed for reproducibility
# value_range = (0, 1)  # Range to scale the time series values
# correlation_strength = 0.001  # Strength of the correlation between components

# random_time_series = generate_time_series(n_steps, n_dim, scale_factor, wishart_df, seed, value_range, correlation_strength)
# print(random_time_series.shape)

# random_time_series = {"random_walk_time_series": random_time_series}
# plot_attention_metrics_norms_over_time(random_time_series, metrics=["random_walk_time_series"], num_heads_plot=8, 
#     save=True, base_plot_path=constants.PLOTS_TIME_SERIES_DIR+'random_walk_time_series'+"/")

# save_time_series(random_time_series, base_save_path=constants.TIME_SERIES_DIR+"random_walk_time_series.pt")

# cognitive_task = 'random_walk_time_series'
# print(f"\n--- Computing PhiID for task {cognitive_task} ---")
# save_path = os.path.join(constants.MATRICES_DIR, cognitive_task, f"random_walk_time_series.pt")
# compute_PhiID(random_time_series, save=True, kind="gaussian", base_save_path=save_path)
# all_matrices, synergy_matrices, redundancy_matrices = load_matrices(base_save_path=save_path)
# graph_theoretical_results = compare_synergy_redundancy(synergy_matrices, redundancy_matrices, selected_metrics=['random_walk_time_series'])
# save_graph_theoretical_results(graph_theoretical_results, file_name=str(cognitive_task), base_save_path = constants.GRAPH_METRICS_DIR + cognitive_task + '/')



for model_code in constants.FINAL_MODELS:
    print(f"\n--- Processing Model: {model_code} ---")
    constants.update_model_code(model_code)
    
    ### Time Series Plots ###

    time_series = {cognitive_task: {} for cognitive_task in constants.PROMPT_CATEGORIES}

    # print("Loading Raw Attention and Time Series")
    # for cognitive_task in constants.PROMPT_CATEGORIES:
    #     print("Loading Cognitive Task: ", cognitive_task)
    #     for n_prompt, prompt in enumerate(constants.PROMPTS[cognitive_task]):
    #         time_series[cognitive_task][n_prompt] = load_time_series(base_load_path=constants.TIME_SERIES_DIR+cognitive_task+"/"+str(n_prompt) + ".pt")
    #         plot_attention_metrics_norms_over_time(time_series[cognitive_task][n_prompt], metrics=constants.METRICS_TRANSFORMER, num_heads_plot=8, smoothing_window=0, 
    #             save=constants.SAVE_PLOTS, base_plot_path=constants.PLOTS_TIME_SERIES_DIR+cognitive_task+"/"+str(n_prompt)+"/")



    ### Plotting PhiID for Average Prompts and Random Walk Time Series ###
    prompt_category_names = ['average_prompts']

    for prompt_category_name in prompt_category_names:
        print("\n--- Plotting Prompt Category: ", prompt_category_name, " ---")

        global_matrices, synergy_matrices, redundancy_matrices = load_matrices(base_save_path=constants.MATRICES_DIR + prompt_category_name + '/' + prompt_category_name + '.pt')
        # plot_synergy_redundancy_PhiID(synergy_matrices, redundancy_matrices, save=constants.SAVE_PLOTS, base_plot_path=constants.PLOTS_SYNERGY_REDUNDANCY_DIR + prompt_category_name + '/')
        # plot_all_PhiID(global_matrices, save=constants.SAVE_PLOTS, base_plot_path=constants.PLOTS_SYNERGY_REDUNDANCY_DIR + prompt_category_name + '/')
        # results_all_phid = plot_all_PhiID_separately({"attention_outputs": global_matrices["attention_outputs"]}, save=constants.SAVE_PLOTS, base_plot_path=constants.PLOTS_SYNERGY_REDUNDANCY_DIR + prompt_category_name + '/')
        
        # plot_box_plot_information_dynamics(results_all_phid, atom_or_dynamics="dynamics", save=constants.SAVE_PLOTS, base_plot_path=constants.PLOTS_SYNERGY_REDUNDANCY_DIR + prompt_category_name + '/')
        # plot_box_plot_information_dynamics(results_all_phid, atom_or_dynamics="atoms", save=constants.SAVE_PLOTS, base_plot_path=constants.PLOTS_SYNERGY_REDUNDANCY_DIR + prompt_category_name + '/')

        base_plot_path = constants.PLOTS_SYNERGY_REDUNDANCY_DIR + prompt_category_name + '/'
        averages = calculate_average_synergy_redundancies_per_head(synergy_matrices, redundancy_matrices, within_layer=False)
        plot_averages_per_head(averages, save=constants.SAVE_PLOTS, use_heatmap=True, num_heads_per_layer=constants.NUM_HEADS_PER_LAYER, base_plot_path=base_plot_path)
        plot_averages_per_layer(averages, num_heads_per_layer=constants.NUM_HEADS_PER_LAYER)
        # gradient_ranks = compute_gradient_rank(averages)
        gradient_ranks = compute_gradient_rank_first(averages)
        plot_gradient_rank(gradient_ranks, base_plot_path=base_plot_path, save=constants.SAVE_PLOTS, use_heatmap=True, num_heads_per_layer=constants.NUM_HEADS_PER_LAYER)
        ranks_per_layer_mean, ranks_per_layer_std = plot_average_ranks_per_layer(gradient_ranks, save=constants.SAVE_PLOTS, num_heads_per_layer=constants.NUM_HEADS_PER_LAYER, base_plot_path=base_plot_path)
    
        # Graph Theoretical Analysis
        # graph_theoretical_results = load_graph_theoretical_results(base_save_path=constants.GRAPH_METRICS_DIR + prompt_category_name + '/', file_name=prompt_category_name)
        # plot_graph_theoretical_results(graph_theoretical_results, save=constants.SAVE_PLOTS, base_plot_path=base_plot_path)


# ### Plotting PhiID for All Cognitive Tasks ###
# ranks_per_layer_mean = {cognitive_task: {} for cognitive_task in constants.PROMPT_CATEGORIES}
# ranks_per_layer_std = {cognitive_task: {} for cognitive_task in constants.PROMPT_CATEGORIES}
# global_matrices = {cognitive_task: {} for cognitive_task in constants.PROMPT_CATEGORIES}
# synergy_matrices = {cognitive_task: {} for cognitive_task in constants.PROMPT_CATEGORIES}
# redundancy_matrices = {cognitive_task: {} for cognitive_task in constants.PROMPT_CATEGORIES}

# for prompt_category_name in constants.PROMPT_CATEGORIES:
#     print("Plotting Prompt Category: ", prompt_category_name,)
#     for n_prompt, prompt in enumerate(constants.PROMPTS[prompt_category_name]):
#         print("Prompt Number: ", n_prompt)

    
#         global_matrices[prompt_category_name][n_prompt], synergy_matrices[prompt_category_name][n_prompt], redundancy_matrices[prompt_category_name][n_prompt] = load_matrices(base_save_path=constants.MATRICES_DIR + prompt_category_name + '/' + str(n_prompt) + '.pt')
        # plot_synergy_redundancy_PhiID( synergy_matrices[prompt_category_name][n_prompt], redundancy_matrices[prompt_category_name][n_prompt], 
        #                               save=constants.SAVE_PLOTS, base_plot_path=constants.PLOTS_SYNERGY_REDUNDANCY_DIR + prompt_category_name + '/' + str(n_prompt) + '/')
        # plot_all_PhiID(global_matrices[prompt_category_name][n_prompt], save=constants.SAVE_PLOTS, base_plot_path=constants.PLOTS_SYNERGY_REDUNDANCY_DIR + prompt_category_name + '/' + str(n_prompt) + '/')
        # plot_all_PhiID_separately(global_matrices[prompt_category_name][n_prompt], save=constants.SAVE_PLOTS, base_plot_path=constants.PLOTS_SYNERGY_REDUNDANCY_DIR + prompt_category_name + '/' + str(n_prompt) + '/')

        # base_plot_path = constants.PLOTS_SYNERGY_REDUNDANCY_DIR + prompt_category_name + '/' + str(n_prompt) + '/'
        # averages = calculate_average_synergy_redundancies_per_head(synergy_matrices[prompt_category_name][n_prompt], redundancy_matrices[prompt_category_name][n_prompt], 
        #                                                            within_layer=False)
        # plot_averages_per_head(averages, save=constants.SAVE_PLOTS, use_heatmap=True, num_heads_per_layer=constants.NUM_HEADS_PER_LAYER, base_plot_path=base_plot_path)
        # plot_averages_per_layer(averages, num_heads_per_layer=constants.NUM_HEADS_PER_LAYER)
        # gradient_ranks = compute_gradient_rank(averages)
        # plot_gradient_rank(gradient_ranks, base_plot_path=base_plot_path, save=constants.SAVE_PLOTS, use_heatmap=True, num_heads_per_layer=constants.NUM_HEADS_PER_LAYER)
        # ranks_per_layer_mean[prompt_category_name], ranks_per_layer_std[prompt_category_name] = plot_average_ranks_per_layer(gradient_ranks, save=constants.SAVE_PLOTS, num_heads_per_layer=constants.NUM_HEADS_PER_LAYER, base_plot_path=base_plot_path)

        # # Graph Theoretical Analysis
        # graph_theoretical_results = load_graph_theoretical_results(base_save_path=constants.GRAPH_METRICS_DIR + prompt_category_name + '/', file_name=str(n_prompt))
        # plot_graph_theoretical_results(graph_theoretical_results, save=constants.SAVE_PLOTS, base_plot_path=base_plot_path)


# ### Different Attention Heads for Different Cognitive Tasks ###
# base_plot_path = constants.PLOTS_HEAD_ACTIVATIONS_COGNITIVE_TASKS

# print("Loading Attention Weights")
# attention_weights_prompts =  load_attention_weights()

# print("Plotting Attention Weights")
# summary_stats_prompts = plot_categories_comparison(attention_weights_prompts, save=constants.SAVE_PLOTS, base_plot_path=base_plot_path, split_half=False, split_third=False)
# plot_all_heatmaps(attention_weights_prompts, save=constants.SAVE_PLOTS, base_plot_path=base_plot_path)
# perform_lda_analysis(attention_weights_prompts, save=constants.SAVE_PLOTS, base_plot_path=base_plot_path)