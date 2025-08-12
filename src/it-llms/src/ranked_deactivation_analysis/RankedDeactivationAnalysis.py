import pickle
import os
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional, Tuple, Union, Dict
import xarray as xr
import torch
from dataclasses import dataclass
import matplotlib.pyplot as plt
from functools import cached_property

from src.utils import template_tokenize_prompts, get_tokens_and_probs, invert_node_ranking, get_teacher_forcing_tokens_and_probs
from src.ranked_deactivation_analysis.deactivate_model_parts import deactivate_model_parts

@dataclass
class PerformanceDivergenceResult:
    kl_xr: xr.DataArray                     # dims: ("category", "prompt", "time")
    num_nodes_deactivated: int
    deactivated_nodes: List[Tuple[int, int]]

    # ---------- derived metrics -----------------------------------------
    @cached_property
    def overall_performance_divergence(self) -> float:
        return float(self.kl_xr.mean(skipna=True).item())

    @cached_property
    def divergence_per_prompt(self) -> xr.DataArray:
        # dims -> ("category", "prompt")
        return self.kl_xr.mean(dim="time", skipna=True)

    @cached_property
    def divergence_per_category(self) -> xr.DataArray:
        # dims -> ("category",)
        return self.kl_xr.mean(dim=("prompt", "time"), skipna=True)

@dataclass
class RankedDeactivationResults:
    deactivation_results: List[PerformanceDivergenceResult]  # One per iteration
    deactivation_schedule: List[int]  # Number of nodes deactivated at each iteration
    
    def save(self, dir_path: str):
        """ Save the results to a file. """

        dir_path = file_path.rsplit('/', 1)[0]
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Results saved to {file_path}")

    @classmethod
    def load(cls, file_path: str) -> 'RankedDeactivationResults':
        """ Load the results from a file. """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")

        with open(file_path, 'rb') as f:
            results = pickle.load(f)

        if not isinstance(results, RankedDeactivationResults):
            raise ValueError("Loaded data is not of type RankedDeactivationResults.")

        return cls(**results.__dict__)
    
    def plot_overall_performance_divergence(self):
        """
        Plot the performance divergence as a function of number of deactivated nodes.
        """
        x = self.deactivation_schedule
        y = [result.overall_performance_divergence for result in self.deactivation_results]

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'b-o', linewidth=2, markersize=6)
        plt.xlabel('Number of Deactivated Nodes')
        plt.ylabel('Overall Performance Divergence (KL)')
        plt.title('Performance Divergence vs Number of Deactivated Nodes')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_performance_divergence_per_category(self):
        """
        Plot the performance divergence per category as a function of number of deactivated nodes.
        """
        x = self.deactivation_schedule
        categories = self.deactivation_results[0].kl_xr.coords['category'].values

        plt.figure(figsize=(12, 8))
        for i, category in enumerate(categories):
            y = [result.divergence_per_category.sel(category=category).item() for result in self.deactivation_results]
            plt.plot(x, y, label=category, marker='o')

        plt.xlabel('Number of Deactivated Nodes')
        plt.ylabel('Performance Divergence (KL)')
        plt.title('Performance Divergence per Category vs Number of Deactivated Nodes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


class RankedDeactivationAnalysis:
    def __init__(
        self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompts: Union[List[str], List[List[str]]], chat_template: str,
        node_ranking: xr.DataArray, # dims ('source_layer', 'source_node'), values are node ranks
        max_new_tokens: int = 128
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.node_ranking = invert_node_ranking(node_ranking)
        print(f"Node ranking loaded with {len(self.node_ranking)} nodes: {self.node_ranking[:5]}{'...' if len(self.node_ranking) > 5 else ''}")
        # Randomly shuffle the node ranking list several times to ensure randomness
        # random.shuffle(self.node_ranking)

        self.max_new_tokens = max_new_tokens

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

        self.tokenized_prompts = template_tokenize_prompts(
            prompts,
            tokenizer,
            prompt_template=chat_template,
            tokenize_kwargs={
                "padding": "longest",
                "truncation": True,
            },
        )
    
    def reverse_node_ranking(self):
        """
        Reverse the node ranking to get the original order.
        This is useful for comparing with the original model performance.
        """
        self.node_ranking = self.node_ranking[::-1]
        print(f"Node ranking reversed: {self.node_ranking[:10]}{'...' if len(self.node_ranking) > 5 else ''}")

    def randomize_node_ranking(self):
        """
        Randomly shuffle the node ranking list to ensure randomness in deactivation order.
        """
        random.shuffle(self.node_ranking)
        print(f"Node ranking randomized: {self.node_ranking[:10]}{'...' if len(self.node_ranking) > 5 else ''}")

    
    def compute_kl_divergence(
        self,
        non_deactivated_results: Dict[str, List[tuple]],
        deactivated_results: Dict[str, List[tuple]],
        num_nodes_deactivated: int,
        deactivated_nodes: List[Tuple[int, int]],
        reverse_kl: bool = False
    ) -> PerformanceDivergenceResult:
        """
        Compute a single 3D xarray.DataArray of KL divergences:
            dims = (category, prompt, time)
        NaN padding is used for variable-length continuations.
        """
        eps = 1e-10
        categories = list(non_deactivated_results.keys())
        n_cat = len(categories)

        # ── Determine max prompt count and max generation length ─────────────
        max_prompts = max(len(v) for v in non_deactivated_results.values())
        max_T = max(
            probs.shape[0]
            for prompts in non_deactivated_results.values()
            for (_, probs, _) in prompts
        )

        # ── Allocate output array (with padding) ─────────────────────────────
        kl_tensor = torch.full((n_cat, max_prompts, max_T), float("nan"))

        # ── Fill it directly ─────────────────────────────────────────────────
        for c_idx, cat in enumerate(categories):
            na_list = non_deactivated_results[cat]
            a_list  = deactivated_results[cat]
            for p_idx, ((_, p_na, _), (_, p_a, _)) in enumerate(zip(na_list, a_list)):
                assert p_na.shape == p_a.shape, f"Shape mismatch for category '{cat}', prompt {p_idx}: {p_na.shape} vs {p_a.shape}"
                p = p_na.float().clamp_min(eps)
                q = p_a.float().clamp_min(eps)
                if reverse_kl:
                    # KL(q || p)
                    kl_t = torch.sum(q * torch.log(q / p), dim=-1)
                else:
                    # KL(p || q)
                    kl_t = torch.sum(p * torch.log(p / q), dim=-1)  # (T,)
                kl_tensor[c_idx, p_idx, :kl_t.shape[0]] = kl_t.cpu()

        # ── Wrap in xarray for labeled indexing ──────────────────────────────
        kl_xr = xr.DataArray(
            kl_tensor,
            dims=["category", "prompt", "time"],
            coords={
                "category": categories,
                "prompt": list(range(max_prompts)),
                "time": list(range(max_T)),
            },
        )

        return PerformanceDivergenceResult(
            kl_xr=kl_xr,
            num_nodes_deactivated=num_nodes_deactivated,
            deactivated_nodes=deactivated_nodes,
        )
  
    def run(
        self, 
        deactivate_k_nodes_per_iteration: int, 
        max_deactivated_nodes: Union[int, None] = None, 
        micro_batch_size: int = 32, 
        save_file_path: str = None,
        reverse_kl: bool = False,
        noise_std: Optional[float] = None
    ) -> RankedDeactivationResults:
        """
        Run the ranked deactivation analysis on the model with the given prompts.
        
        :param deactivate_k_nodes_per_iteration: Number of additional nodes to deactivate in each iteration.
        :param max_deactivated_nodes: Maximum number of nodes to deactivate in total. If None, deactivate all.
        """
        print("Getting non-deactivated model results...")
        # Dict[str, List[GenResult]] where GenResult = (tokens, probs, decoded_text)
        non_deactivated_token_and_logits_generate = get_tokens_and_probs( 
            model=self.model, 
            tokenizer=self.tokenizer,
            tokenized_prompts=self.tokenized_prompts, 
            max_new_tokens=self.max_new_tokens, 
            micro_batch_size=micro_batch_size
        )
        # Re-run with the teacher forcing tokens and logits to avoid numerical differences between underlying 'generate' and 'forward' methods
        non_deactivated_token_and_logits = get_teacher_forcing_tokens_and_probs(
            model=self.model,
            tokenizer=self.tokenizer,
            non_deactivated_token_and_logits=non_deactivated_token_and_logits_generate,
            micro_batch_size=micro_batch_size,
        )

        deactivation_results = []
        deactivation_schedule = []

        # Iterate over the node ranking and deactivate nodes
        total_nodes = len(self.node_ranking)
        max_deactivated_nodes = min(total_nodes, max_deactivated_nodes) if max_deactivated_nodes is not None else total_nodes
        
        print(f"Starting deactivation analysis: {max_deactivated_nodes} max nodes, {deactivate_k_nodes_per_iteration} per iteration")
        
        for iteration, last_deactivated_node in enumerate(range(0, max_deactivated_nodes + deactivate_k_nodes_per_iteration, deactivate_k_nodes_per_iteration)):
            nodes_to_deactivate = self.node_ranking[:last_deactivated_node]
            num_nodes_to_deactivate = len(nodes_to_deactivate)
            deactivated_nodes_list = [(int(layer), int(node)) for layer, node in nodes_to_deactivate]
            
            print(f"\nIteration {iteration + 1}: Deactivating {num_nodes_to_deactivate} nodes")
            print(f"Sample deactivated nodes: {deactivated_nodes_list}{'...' if len(deactivated_nodes_list) > 5 else ''}")

            # Deactivate the nodes in the model temporarily only for this iteration
            with deactivate_model_parts(
                model=self.model,
                nodes_to_deactivate=nodes_to_deactivate,
                module_name="self_attn",  # "self_attn", "mlp", etc.
                noise_std=noise_std # Optional noise standard deviation for deactivation

            ) as deactivated_model:
                # Re-run the generation with the deactivated nodes
                # Dict[str, List[GenResult]] where GenResult = (tokens, probs, decoded_text)
                print(f"Getting deactivated model results for {num_nodes_to_deactivate} deactivated nodes...")
                deactivated_token_and_logits = get_teacher_forcing_tokens_and_probs( 
                    model=deactivated_model, 
                    tokenizer=self.tokenizer,
                    non_deactivated_token_and_logits=non_deactivated_token_and_logits_generate,
                    micro_batch_size=micro_batch_size,
                )
                
                # Compute KL divergence
                print("Computing KL divergence...")
                divergence_result = self.compute_kl_divergence(
                    non_deactivated_results=non_deactivated_token_and_logits,
                    deactivated_results=deactivated_token_and_logits,
                    num_nodes_deactivated=num_nodes_to_deactivate,
                    deactivated_nodes=deactivated_nodes_list,
                    reverse_kl=reverse_kl
                )
                
                print(f"Overall performance divergence: {divergence_result.overall_performance_divergence:.6f}")
                
                deactivation_results.append(divergence_result)
                deactivation_schedule.append(num_nodes_to_deactivate)
                
                # Clear GPU memory
                del deactivated_token_and_logits
                torch.cuda.empty_cache()
        

        results = RankedDeactivationResults(
            deactivation_results=deactivation_results,
            deactivation_schedule=deactivation_schedule
        )

        if save_file_path:
            results.save(save_file_path)

        return results
    