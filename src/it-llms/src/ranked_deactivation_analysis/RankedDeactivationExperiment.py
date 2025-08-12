from __future__ import annotations
import os
import numpy as np

"""ranked_deactivation_experiment.py

Thin wrapper around `RankedDeactivationAnalysis` that makes it easy to run and
compare multiple deactivation schedules (e.g. original vs. shuffled ranking)
and to visualise both overall and per‑category performance divergence.

This module is intentionally lightweight—the heavy lifting lives inside
`RankedDeactivationAnalysis`.  The wrapper just instantiates the analysis
object as needed, keeps track of the results, and offers a few convenience
plotting helpers.  Adding new experimental runs is as simple as calling
`experiment.run(...)` with a new `name`.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt

from src.ranked_deactivation_analysis import (
    RankedDeactivationAnalysis,  # noqa: F401 – imported for side effects / typing
    RankedDeactivationResults,
)

__all__ = ["RankedDeactivationExperiment"]


@dataclass
class RankedDeactivationExperiment:
    """Run one or more node‑deactivation experiments and plot the results."""

    # Arguments required to build a `RankedDeactivationAnalysis`.
    analysis_kwargs: Dict[str, Any]

    # Container for the results of each run: ``run_name -> results``.
    runs: Dict[str, RankedDeactivationResults] = field(default_factory=dict)

    # ---------------------------------------------------------------------
    # Core API
    # ---------------------------------------------------------------------
    def run(
        self,
        *,
        name: str,
        deactivate_k_nodes_per_iteration: int,
        max_deactivated_nodes: Optional[int] = None,
        micro_batch_size: int = 32,
        randomise_order: bool = False,
        reverse_order: bool = False,
        save_file_path: Optional[str] = None,
        reverse_kl: bool = False,
        noise_std: Optional[float] = None,
        data_deactivation_dir: Optional[str] = None,
    ) -> RankedDeactivationResults:
        """Run a single deactivation experiment.

        Parameters
        ----------
        name
            A unique key that will identify this run (e.g. "original",
            "shuffled").  Used as the label in plots.
        randomise_order
            If *True*, shuffle the node‑ranking before running so the nodes are
            deactivated in a random order.  The shuffling is *in‑place* on a
            copy of the ranking inside the analysis instance, so subsequent
            runs are unaffected unless `randomise_order` is re‑used.
        All other arguments are forwarded verbatim to ``RankedDeactivationAnalysis.run``.
        """

        # Build analysis instance.
        analysis = RankedDeactivationAnalysis(**self.analysis_kwargs)

        # Optionally shuffle the node ranking.
        if randomise_order:
            analysis.randomize_node_ranking()
        
        if reverse_order:
            analysis.reverse_node_ranking()

        # Execute.
        results = analysis.run(
            deactivate_k_nodes_per_iteration=deactivate_k_nodes_per_iteration,
            max_deactivated_nodes=max_deactivated_nodes,
            micro_batch_size=micro_batch_size,
            save_file_path=save_file_path,
            reverse_kl=reverse_kl,
            noise_std=noise_std,
        )

        if data_deactivation_dir is not None:
            file_path = os.path.join(data_deactivation_dir, f"{name}.pkl")
            results.save(file_path)

        # Stash for later comparison.
        if name in self.runs:
            raise ValueError(f"Run name '{name}' already exists – choose a new one.")
        self.runs[name] = results
        return results

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def run_default_and_random(
        self,
        *,
        deactivate_k_nodes_per_iteration: int,
        max_deactivated_nodes: Optional[int] = None,
        micro_batch_size: int = 32,
        n_randomised_runs: int = 1,
        reverse_kl: bool = False,
        run_reverse_ranking: bool = False,
        noise_std: Optional[float] = None,
        data_deactivation_dir: Optional[str] = None,
    ) -> Dict[str, RankedDeactivationResults]:
        """Run the *original* and a *randomised* deactivation order back‑to‑back."""

        # Synergistic Order
        self.run(
            name="syn_minus_red_order",
            deactivate_k_nodes_per_iteration=deactivate_k_nodes_per_iteration,
            max_deactivated_nodes=max_deactivated_nodes,
            micro_batch_size=micro_batch_size,
            randomise_order=False,
            reverse_kl=reverse_kl,
            noise_std=noise_std,
            data_deactivation_dir=data_deactivation_dir,
        )

        # Reverse Synergistic Order
        if run_reverse_ranking:
            self.run(
                name="reverse_syn_minus_red_order",
                deactivate_k_nodes_per_iteration=deactivate_k_nodes_per_iteration,
                max_deactivated_nodes=max_deactivated_nodes,
                micro_batch_size=micro_batch_size,
                randomise_order=False,
                reverse_order=True,
                reverse_kl=reverse_kl,
                noise_std=noise_std,
                data_deactivation_dir=data_deactivation_dir,
            )

        # Run the randomised order multiple times if requested.
        for i in range(n_randomised_runs):
            self.run(
                name=f"random_order_{i + 1}",
                deactivate_k_nodes_per_iteration=deactivate_k_nodes_per_iteration,
                max_deactivated_nodes=max_deactivated_nodes,
                micro_batch_size=micro_batch_size,
                randomise_order=True,
                reverse_kl=reverse_kl,
                noise_std=noise_std,
                data_deactivation_dir=data_deactivation_dir,
            )
        return self.runs

    # ------------------------------------------------------------------
    # Plotting utilities
    # ------------------------------------------------------------------

    def _is_random(self, name: str) -> bool:
        return name.startswith("random_order_")

    def _fraction_mask(self, x, fraction: float):
        if not (0 < fraction <= 1):
            raise ValueError("fraction must be in (0, 1].")
        x_arr = np.asarray(x, dtype=float)
        cutoff = fraction * float(x_arr.max())
        mask = x_arr <= cutoff
        # ensure we show at least the first point
        if not mask.any():
            mask = np.zeros_like(x_arr, dtype=bool)
            mask[0] = True
        return mask

    
    def plot_overall(
        self,
        plot_dir: Optional[str] = None,
        *,
        aggregate_random: bool = False,
        fraction: float = 1.0,
    ) -> None:
        """Plot *overall* divergence curves for all stored runs.

        If `aggregate_random=True`, collapse runs named `random_order_*` into a single
        mean curve with a shaded ±1 std band.

        `fraction` ∈ (0,1] limits the plot to the first fraction of the deactivation
        schedule (by x-axis value, i.e., number of deactivated nodes).
        """
        if not self.runs:
            raise RuntimeError("No runs available – call .run() first.")

        random_items = [(n, r) for n, r in self.runs.items() if self._is_random(n)]
        nonrandom_items = [(n, r) for n, r in self.runs.items() if not self._is_random(n)]

        plt.figure(figsize=(10, 6))

        # Plot non-random runs as-is (subject to fraction)
        for run_name, res in nonrandom_items:
            x = np.asarray(res.deactivation_schedule, dtype=float)
            y = np.asarray([r.overall_performance_divergence for r in res.deactivation_results], dtype=float)
            mask = self._fraction_mask(x, fraction)
            plt.plot(x[mask], y[mask], marker="o", label=run_name)

        # Aggregate the random runs if requested
        if aggregate_random and random_items:
            x = np.asarray(random_items[0][1].deactivation_schedule, dtype=float)
            mask = self._fraction_mask(x, fraction)

            y_stack = []
            for _, res in random_items:
                y = np.asarray([r.overall_performance_divergence for r in res.deactivation_results], dtype=float)
                y_stack.append(y[mask])
            Y = np.vstack(y_stack)  # (n_runs, n_sel_steps)

            mean = Y.mean(axis=0)
            band = Y.std(axis=0, ddof=0)

            line, = plt.plot(x[mask], mean, marker="o", label="random_order")
            plt.fill_between(x[mask], mean - band, mean + band, color=line.get_color(), alpha=0.2)

        # Otherwise, show each random run (subject to fraction)
        if not aggregate_random:
            for run_name, res in random_items:
                x = np.asarray(res.deactivation_schedule, dtype=float)
                y = np.asarray([r.overall_performance_divergence for r in res.deactivation_results], dtype=float)
                mask = self._fraction_mask(x, fraction)
                plt.plot(x[mask], y[mask], marker="o", label=run_name)

        plt.xlabel("Number of Deactivated Nodes")
        plt.ylabel("Overall Performance Divergence (KL)")
        plt.title("Overall Performance Divergence Comparison")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        if plot_dir:
            plot_dir += 'fraction_' + str(fraction).replace('.', '_') + '/'
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir, exist_ok=True)
            save_file = os.path.join(plot_dir, "overall_divergence_plot.png")
            plt.savefig(save_file, dpi=300)
            plt.close()
            print(f"Plot saved to {save_file}")
        else:
            plt.show()
  
    
    def plot_per_category(
        self,
        plot_dir: Optional[str] = None,
        *,
        aggregate_random: bool = False,
        fraction: float = 1.0,
    ) -> None:
        """Plot per-category divergence curves for every stored run.

        If `aggregate_random=True`, collapse runs named `random_order_*` into a single
        mean curve per category with a shaded ±1 std band.

        `fraction` ∈ (0,1] limits the plot to the first fraction of the deactivation
        schedule (by x-axis value).
        """
        if not self.runs:
            raise RuntimeError("No runs available – call .run() first.")

        random_items = [(n, r) for n, r in self.runs.items() if self._is_random(n)]
        nonrandom_items = [(n, r) for n, r in self.runs.items() if not self._is_random(n)]

        # Assume categories are identical across runs – grab from the first.
        first_res = next(iter(self.runs.values()))
        categories = first_res.deactivation_results[0].kl_xr.coords["category"].values

        n_categories = len(categories)
        n_cols = 2
        n_rows = (n_categories + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), sharex="all", sharey="all")
        axes = axes.flatten()

        for idx, category in enumerate(categories):
            ax = axes[idx]

            # Plot non-random runs per category (subject to fraction)
            for run_name, res in nonrandom_items:
                x = np.asarray(res.deactivation_schedule, dtype=float)
                y = np.asarray(
                    [r.divergence_per_category.sel(category=category).item() for r in res.deactivation_results],
                    dtype=float,
                )
                mask = self._fraction_mask(x, fraction)
                ax.plot(x[mask], y[mask], marker="o", label=run_name)

            # Aggregate randoms if requested
            if aggregate_random and random_items:
                x = np.asarray(random_items[0][1].deactivation_schedule, dtype=float)
                mask = self._fraction_mask(x, fraction)

                y_stack = []
                for _, res in random_items:
                    y = np.asarray(
                        [r.divergence_per_category.sel(category=category).item() for r in res.deactivation_results],
                        dtype=float,
                    )
                    y_stack.append(y[mask])
                Y = np.vstack(y_stack)

                mean = Y.mean(axis=0)
                band = Y.std(axis=0, ddof=0)

                line, = ax.plot(x[mask], mean, marker="o", label="random_order")
                ax.fill_between(x[mask], mean - band, mean + band, color=line.get_color(), alpha=0.2)

            # Otherwise, show each random run (subject to fraction)
            if not aggregate_random:
                for run_name, res in random_items:
                    x = np.asarray(res.deactivation_schedule, dtype=float)
                    y = np.asarray(
                        [r.divergence_per_category.sel(category=category).item() for r in res.deactivation_results],
                        dtype=float,
                    )
                    mask = self._fraction_mask(x, fraction)
                    ax.plot(x[mask], y[mask], marker="o", label=run_name)

            ax.set_title(str(category))
            ax.set_xlabel("# Deactivated Nodes")
            ax.set_ylabel("KL")
            ax.grid(alpha=0.3)
            if idx == 0:
                ax.legend()

        # Hide any unused subplots.
        for unused_ax in axes[n_categories:]:
            unused_ax.set_visible(False)

        fig.suptitle("Per-Category Performance Divergence Comparison", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if plot_dir:
            plot_dir += 'fraction_' + str(fraction).replace('.', '_') + '/'
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir, exist_ok=True)
            save_file = os.path.join(plot_dir, "per_category_divergence_plot.png")
            plt.savefig(save_file, dpi=300)
            plt.close()
            print(f"Plot saved to {save_file}")
        else:
            plt.show()



    # ------------------------------------------------------------------
    # Convenience dunder methods
    # ------------------------------------------------------------------
    def __getitem__(self, run_name: str) -> RankedDeactivationResults:
        return self.runs[run_name]

    def __iter__(self):
        return iter(self.runs.items())
