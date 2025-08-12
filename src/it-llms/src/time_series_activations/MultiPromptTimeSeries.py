"""Time‑series utilities for analysing per‑node activations across multiple prompts."""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Sequence, Union, Optional, Iterable, Tuple
import os

import numpy as np
import matplotlib.pyplot as plt

from src.utils import ModelInformation
from src.activation_recorder import MultiPromptActivations, PromptActivations, ModelActivations


Projection = Literal["norm", "mean", "max"]


@dataclass
class NodeTimeSeries:
    """Stores the scalar time‑series for a single node (e.g. an attention head)."""

    node_index: int
    model_info: ModelInformation
    _buffer: List[float] = field(default_factory=list, init=False, repr=False)

    def add_timestep(self, value: float) -> None:  # O(1)
        """Append a new scalar value (cheap list append)."""
        self._buffer.append(float(value))
    
    @property
    def time_series(self) -> np.ndarray:
        """Return the time_series as a *materialised* float32 array (lazy conversion)."""
        if isinstance(self._buffer, list):  # first access ➜ convert & cache
            self._buffer = np.asarray(self._buffer, dtype=np.float32)
        return self._buffer  # type: ignore[return-value]


@dataclass
class LayerTimeSeries:
    """Aggregates ``NodeTimeSeries`` for every node in a layer."""

    layer_index: int
    model_info: ModelInformation
    nodes: Dict[int, NodeTimeSeries] = field(default_factory=dict, init=False)

    def get_or_create_node(self, node_index: int) -> NodeTimeSeries:
        series = self.nodes.get(node_index)
        if series is None:
            series = self.nodes[node_index] = NodeTimeSeries(node_index, self.model_info)
        return series


@dataclass
class PromptTimeSeries:
    """Holds per‑layer time‑series for a single prompt."""

    prompt_index: int
    model_info: ModelInformation
    layers: Dict[int, LayerTimeSeries] = field(default_factory=dict, init=False)
    generated_tokens: Sequence[str] = field(default_factory=list, init=False)

    @classmethod
    def from_activations(
        cls,
        prompt_index: int,
        activations: PromptActivations,
        model_info: ModelInformation,
        node_type: str,
        node_activation: str,
        projection_method: Projection = "norm",
        exclude_shared_expert_moe: bool = False,
    ) -> "PromptTimeSeries":
        """Build and populate a ``PromptTimeSeries`` from recorded activations.

        Parameters
        ----------
        prompt_index : int
            Index of the prompt.
        activations : dict[int, MultiPromptActivations]
            Activations for the prompt, indexed by layer.
        model_info : ModelInformation
            Model metadata.
        node_type : str
            Attribute name inside ``layer`` objects (e.g. ``"attention_head"``).
        node_activation : str
            Attribute name inside ``node`` objects whose tensor we will project.
        projection_method : {"norm", "mean", "max"}
            Scalar projection applied to the tensor.
        exclude_shared_expert_moe : bool
            Whether to exclude shared expert MOE nodes from the time series.
        """
        obj = cls(prompt_index, model_info)

        generated_tokens = activations.generated_tokens
        if node_type =="moe":
            activations.uncompress_moe_activations(node_activation=node_activation)

        for step_index, step_acts in activations.steps.items():
            del step_index  # step granularity handled implicitly by append order

            for layer_index, layer_acts in step_acts.layers.items():
                layer_ts = obj.get_or_create_layer(layer_index)

                node_layer = getattr(layer_acts, node_type, None)
                if layer_index == 0 and node_type == "moe":
                    continue # skip MoE layer at index 0 as it is an MLP layer
                
                if node_layer is None:
                    raise AttributeError(f"Layer {layer_index} lacks node type '{node_type}' for prompt {prompt_index}.")

                nodes = getattr(node_layer, "nodes", None)
                if nodes is None:
                    raise AttributeError(f"No 'nodes' attribute in {node_type} of layer {layer_index} (prompt {prompt_index}).")

                for node_index, node_acts in nodes.items():
                    if hasattr(node_acts, "is_shared") and exclude_shared_expert_moe and node_acts.is_shared:
                        # Skip shared expert nodes in MoE layers if requested
                        continue
                    node_ts = layer_ts.get_or_create_node(node_index)
                    activation = getattr(node_acts, node_activation, None)
                    
                    # Project the activation tensor to a scalar
                    value = obj._project(activation, projection_method)
                    node_ts.add_timestep(value)
        # Set the tokens for the prompt time-series
        obj.generated_tokens = generated_tokens if generated_tokens is not None else []


        return obj

    @staticmethod
    def _project(tensor, method: Projection) -> float:  # noqa: ANN001 – tensor type is backend‑dependent
        """Project the incoming tensor to a scalar."""
        if method == "norm":
            return float(tensor.norm())
        if method == "mean":
            return float(tensor.mean())
        if method == "max":
            return float(tensor.max())
        # The ``Literal`` type ensures we never reach here at type‑check time.
        raise ValueError(f"Unsupported projection method: {method}")

    def get_or_create_layer(self, layer_index: int) -> LayerTimeSeries:
        ts = self.layers.get(layer_index)
        if ts is None:
            ts = self.layers[layer_index] = LayerTimeSeries(layer_index, self.model_info)
        return ts

    def plot(
        self,
        *,
        token_x: bool | str = "auto",
        figsize_per_layer: float = 2.5,
        ticks_all_layers: bool = False,
        plot_dir: Union[str, None] = None,
    ) -> None:
        """Plot each node’s series (figure‑per‑prompt, subplot‑per‑layer).

        Parameters
        ----------
        token_x : bool | {'auto', True, False}
            * ``True``  – always use generated tokens as the x‑axis labels.
            * ``False`` – always use numeric indices (legacy behaviour).
            * ``'auto'`` – (default) use tokens *only* when they are present.
        figsize_per_layer : float
            Height in inches allocated to each layer subplot.
        show : bool
            Call ``plt.show()`` automatically.
        """
        # decide whether to show tokens on x‑axis
        labels = list(self.generated_tokens.values())
        labels = [t.replace("$", r"\$").replace("{", "").replace("}", "") for t in labels]
        print(f"Prompt {self.prompt_index} has {len(labels)} generated tokens: {labels}")
        use_tokens = (token_x is True) or (token_x == "auto" and labels)

        layer_ids = sorted(self.layers)
        if not layer_ids:
            return

        fig, axes = plt.subplots(
            len(layer_ids), 1, sharex=True,
            figsize=(10, figsize_per_layer * len(layer_ids)),
        )
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]

        # draw each layer
        for ax, layer_id in zip(axes, layer_ids):
            layer_ts = self.layers[layer_id]
            for node_id, node_ts in layer_ts.nodes.items():
                label = f"N{node_id}"
                ax.plot(range(len(node_ts.time_series)), node_ts.time_series, label=label)
            ax.set_ylabel(f"Layer {layer_id}")
            ax.margins(x=0)

            if ticks_all_layers:
                for ax in axes:
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels, rotation=90, fontsize="small")
                    ax.tick_params(labelbottom=True)  # 👈 force showing labels

        # configure shared x‑axis
        if use_tokens:
            axes[-1].set_xticks(range(len(labels)))
            axes[-1].set_xticklabels(labels, rotation=0, fontsize="small")
            axes[-1].set_xlabel("Token")
        else:
            axes[-1].set_xlabel("Timestep")


        handles_by_label: dict[str, matplotlib.artist.Artist] = {}

        for ax in axes:                                   # every subplot
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l and l not in handles_by_label:       # first time we see this label
                    handles_by_label[l] = h               # remember its handle

        if handles_by_label:                              # create a *deduplicated* legend
            fig.legend(
                handles_by_label.values(), handles_by_label.keys(),
                fontsize="small", ncol=12,
                loc="upper right", bbox_to_anchor=(1, 1)
            )


        fig.tight_layout(rect=[0, 0, 1, 0.98])   # leave 8 % of the height free on top

        if plot_dir:
            plot_dir = f"{plot_dir}/prompt_{self.prompt_index}"
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir, exist_ok=True)
            save_file = f"{plot_dir}/time_series.png"
            fig.savefig(save_file, dpi=300)
            plt.close(fig)
            print(f"Time-series plot saved to {save_file}")
        else:
            plt.show()

    def _get_token_labels(self) -> List[str]:
        """Return sanitized token labels (empty if unavailable)."""
        gt = self.generated_tokens
        if gt is None:
            return []
        # Accept either Sequence[str] or Dict[int, str]
        if isinstance(gt, dict):
            labels = list(gt.values())
        else:
            labels = list(gt)
        # Sanitize for matplotlib/TeX
        labels = [str(t).replace("$", r"\$").replace("{", "").replace("}", "") for t in labels]
        return labels
    

    def _get_series(self, layer_idx: int, node_idx: int) -> np.ndarray:
        if layer_idx not in self.layers:
            raise KeyError(f"Layer {layer_idx} not found for prompt {self.prompt_index}.")
        layer_ts = self.layers[layer_idx]
        if node_idx not in layer_ts.nodes:
            raise KeyError(f"Node {node_idx} not found in layer {layer_idx} for prompt {self.prompt_index}.")
        return layer_ts.nodes[node_idx].time_series

    def plot_two_series(
        self,
        series_a: Tuple[int, int],              # (layer_index, node_index)
        series_b: Tuple[int, int],              # (layer_index, node_index)
        *,
        token_x: bool | str = "auto",           # as in existing plot(): 'auto' | True | False
        tokens: Optional[Iterable[int] | slice | Tuple[int, int]] = None,
        figsize: Tuple[float, float] = (10.0, 5.0),
        plot_dir: Optional[str] = None,         # save as SVG if provided
        dpi: int = 300,
        show: bool = True,                      # call plt.show() (if you’re not saving-only)
    ) -> None:
        """Plot exactly two single-node time series (one per subplot).

        Parameters
        ----------
        series_a, series_b : (layer_index, node_index)
            Select which node to plot in each subplot.
        token_x : bool | {'auto', True, False}
            Whether to display token strings on the x-axis (falls back to indices).
        tokens : None | slice | (start, end) | Iterable[int]
            Restrict which timesteps to display. Examples:
              * None           → plot all timesteps
              * slice(10, 50)  → Python slicing semantics (end exclusive)
              * (10, 50)       → same as slice(10, 50)
              * [0, 2, 5, 13]  → explicit indices
        figsize : (width, height)
            Figure size in inches.
        plot_dir : str | None
            If given, saves the figure as an SVG to this path.
        dpi : int
            DPI used when saving raster preview (not relevant for SVG strokes).
        show : bool
            Whether to call plt.show(). Ignored if running in a headless batch environment.
        """

        la, na = series_a
        lb, nb = series_b
        ts_a = self._get_series(la, na)
        ts_b = self._get_series(lb, nb)

        # Ensure same length when plotting side-by-side (truncate to min)
        L = min(len(ts_a), len(ts_b))
        ts_a = ts_a[:L]
        ts_b = ts_b[:L]

        # ---- compute which indices to plot ----
        if tokens is None:
            idx = np.arange(L)
        elif isinstance(tokens, slice):
            idx = np.arange(L)[tokens]
        elif isinstance(tokens, tuple) and len(tokens) == 2:
            start, end = tokens
            idx = np.arange(L)[slice(start, end)]
        else:
            # assume iterable of indices
            idx = np.array(list(tokens), dtype=int)
            idx = idx[(idx >= 0) & (idx < L)]  # guard against out-of-range

        ts_a = ts_a[idx]
        ts_b = ts_b[idx]

        # ---- x-axis labels / ticks ----
        token_labels = self._get_token_labels()
        token_labels = token_labels[:L] if token_labels else token_labels
        token_labels = [token_labels[i] for i in idx] if token_labels else token_labels

        use_tokens = (token_x is True) or (token_x == "auto" and len(token_labels) > 0)

        # ---- plotting ----
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=figsize)
        ax_a, ax_b = axes

        ax_a.plot(np.arange(len(ts_a)), ts_a)
        ax_a.set_ylabel(f"Layer {la} • Head {na}")
        ax_a.margins(x=0)

        ax_b.plot(np.arange(len(ts_b)), ts_b)
        ax_b.set_ylabel(f"Layer {lb} • Head {nb}")
        ax_b.margins(x=0)
        fig.supylabel("Activation", x=0.02)

        # Configure shared x-axis
        if use_tokens:
            ax_b.set_xticks(np.arange(len(token_labels)))
            ax_b.set_xticklabels(token_labels, rotation=90, fontsize="small")
            ax_b.set_xlabel("Token")
        else:
            ax_b.set_xlabel("Timestep")

        fig.tight_layout()

        # ---- save/show ----
        if plot_dir:
            # Ensure directory exists
            os.makedirs(os.path.dirname(plot_dir) or ".", exist_ok=True)
            file_name = plot_dir + "two_series.svg"
            fig.savefig(file_name, format="svg", dpi=dpi)
            plt.close(fig)
            print(f"Two-series SVG saved to {plot_dir}")
        elif show:
            plt.show()




@dataclass
class MultiPromptTimeSeries:
    """Top‑level container mapping each prompt to a ``PromptTimeSeries``."""

    model_info: ModelInformation
    prompts: Dict[int, PromptTimeSeries] = field(default_factory=dict, init=False)

    @classmethod
    def from_activations(
        cls,
        activations: MultiPromptActivations,
        *,
        node_type: str,
        node_activation: str,
        projection_method: Projection = "norm",
        exclude_shared_expert_moe: bool = False,
    ) -> "MultiPromptTimeSeries":
        """Build and populate a ``MultiPromptTimeSeries`` from recorded activations.

        Parameters
        ----------
        activations : MultiPromptActivations
            Hierarchical activations container (prompts ➜ steps ➜ layers ➜ nodes).
        node_type : str
            Attribute name inside ``layer`` objects (e.g. ``"attention_head"``).
        node_activation : str
            Attribute name inside ``node`` objects whose tensor we will project.
        projection_method : {"norm", "mean"}
            Scalar projection applied to the tensor. Further methods can be added
            by extending the ``_project`` helper.
        """
        obj = cls(activations.model_info)

        # Populate the time-series for each prompt
        for prompt_index, prompt_acts in activations.prompts.items():
            obj.prompts[prompt_index] = PromptTimeSeries.from_activations(
                prompt_index, prompt_acts, obj.model_info, 
                node_type=node_type,
                node_activation=node_activation,
                projection_method=projection_method,
                exclude_shared_expert_moe=exclude_shared_expert_moe,
            )

        return obj

    def plot(
        self,
        *,
        token_x: bool | str = "auto",
        figsize_per_layer: float = 2.5,
        ticks_all_layers: bool = False,
        plot_dir: Union[str, None] = None,
    ) -> None:
        """Plot each node’s series (figure‑per‑prompt, subplot‑per‑layer).

        Parameters
        ----------
        token_x : bool | {'auto', True, False}
            * ``True``  – always use generated tokens as the x‑axis labels.
            * ``False`` – always use numeric indices (legacy behaviour).
            * ``'auto'`` – (default) use tokens *only* when they are present.
        figsize_per_layer : float
            Height in inches allocated to each layer subplot.
        show : bool
            Call ``plt.show()`` automatically.
        """
        for prompt_index, prompt_ts in self.prompts.items():
            prompt_ts.plot(
                token_x=token_x,
                figsize_per_layer=figsize_per_layer,
                ticks_all_layers=ticks_all_layers,
                plot_dir=plot_dir,
            )
                
