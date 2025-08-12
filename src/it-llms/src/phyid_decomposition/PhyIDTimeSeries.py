from dataclasses import dataclass, field
from typing import Dict, List, Literal, Sequence, Union, Tuple
import numpy as np
import xarray as xr

from phyid.calculate import calc_PhiID 

from src.utils import ModelInformation
from src.time_series_activations import NodeTimeSeries


INFORMATION_DYNAMICS = {
    "storage": ["rtr", "xtx", "yty", "sts"],
    "copy": ["xtr", "ytr"],
    "transfer": ["xty", "ytx"],
    "erasure": ["rtx", "rty"],
    "downward_causation": ["stx", "sty", "str_"],
    "upward_causation": ["xts", "yts", "rts"],
    "information_storage": ["rtr", "rtx", "xtr", "xtx"],
    "transfer_entropy_source_past_to_target_future": ["str_", "sty", "xtr", "xty"],
    "causal_density": [(2,"str_"), "sty", "xtr", "xty", "stx", "ytx", "ytr"],
    "integrated_information": ["sts", "xts", "yts", "stx", "sty", "xty", "ytx", "rts", "str_", (-1, "rtr")], 
    "mutual_information": [
        "rtr", "rtx", "rty", "rts", 
        "str_", "stx", "sty", "sts", 
        "xtr", "xtx", "xty", "xts", 
        "ytr", "ytx", "yty", "yts", 
    ]
}


@dataclass
class PhyIDTimeSeries:

    model_info: ModelInformation
    source_layer_index: int
    source_node_index: int
    target_layer_index: int
    target_node_index: int

    xtx: List[float] = field(default_factory=list, repr=False)
    xty: List[float] = field(default_factory=list, repr=False)
    xtr: List[float] = field(default_factory=list, repr=False)
    xts: List[float] = field(default_factory=list, repr=False)

    ytx: List[float] = field(default_factory=list, repr=False)
    yty: List[float] = field(default_factory=list, repr=False)
    ytr: List[float] = field(default_factory=list, repr=False)
    yts: List[float] = field(default_factory=list, repr=False)

    rtx: List[float] = field(default_factory=list, repr=False)
    rty: List[float] = field(default_factory=list, repr=False)
    rtr: List[float] = field(default_factory=list, repr=False)
    rts: List[float] = field(default_factory=list, repr=False)

    stx: List[float] = field(default_factory=list, repr=False)
    sty: List[float] = field(default_factory=list, repr=False)
    str_: List[float] = field(default_factory=list, repr=False)  # 'str' is a built-in, so use 'str_'
    sts: List[float] = field(default_factory=list, repr=False)

    @classmethod
    def from_time_series(
        cls,
        model_info: ModelInformation,
        source_layer_index: int,
        source_node_index: int,
        target_layer_index: int,
        target_node_index: int,
        source_time_series: NodeTimeSeries = None,
        target_time_series: NodeTimeSeries = None,
        phyid_tau: int = 1,
        phyid_kind: Literal["gaussian", "discrete"] = "gaussian",
        phyid_redundancy: Literal["MMI", "CCS"] = "MMI",
    ) -> "PhyIDTimeSeries":
        """ Compute the phyid decomposition for a given source and target nodes. """
        obj = cls(model_info, source_layer_index, source_node_index, target_layer_index, target_node_index)
        if source_time_series is not None and target_time_series is not None:
            obj._compute_phyid(source_time_series, target_time_series, phyid_tau=phyid_tau, phyid_kind=phyid_kind, phyid_redundancy=phyid_redundancy)
        return obj

    def _compute_phyid(
        self,
        source_time_series: NodeTimeSeries,
        target_time_series: NodeTimeSeries,
        phyid_tau: int = 1,
        phyid_kind: Literal["gaussian", "discrete"] = "gaussian",
        phyid_redundancy: Literal["MMI", "CCS"] = "MMI",    
    ) -> None:
        """Compute the phyid decomposition for the given source and target time-series."""
        atoms_res, calc_res = calc_PhiID(
            src=source_time_series.time_series, 
            trg=target_time_series.time_series, 
            tau=phyid_tau, 
            kind=phyid_kind, 
            redundancy=phyid_redundancy
        )
        self.fill_from_atoms(atoms_res)


    def fill_from_atoms(self, atoms_res: Dict[str, Union[float, np.ndarray]]) -> None:
        """Fill the PhyIDTimeSeries from the atoms of the decomposition."""
        self.xtx = np.asarray(atoms_res["xtx"], dtype=np.float32)
        self.xty = np.asarray(atoms_res["xty"], dtype=np.float32)
        self.xtr = np.asarray(atoms_res["xtr"], dtype=np.float32)
        self.xts = np.asarray(atoms_res["xts"], dtype=np.float32)
        self.ytx = np.asarray(atoms_res["ytx"], dtype=np.float32)
        self.yty = np.asarray(atoms_res["yty"], dtype=np.float32)
        self.ytr = np.asarray(atoms_res["ytr"], dtype=np.float32)
        self.yts = np.asarray(atoms_res["yts"], dtype=np.float32)
        self.rtx = np.asarray(atoms_res["rtx"], dtype=np.float32)
        self.rty = np.asarray(atoms_res["rty"], dtype=np.float32)
        self.rtr = np.asarray(atoms_res["rtr"], dtype=np.float32)
        self.rts = np.asarray(atoms_res["rts"], dtype=np.float32)
        self.stx = np.asarray(atoms_res["stx"], dtype=np.float32)
        self.sty = np.asarray(atoms_res["sty"], dtype=np.float32)
        self.str_ = np.asarray(atoms_res["str"], dtype=np.float32)
        self.sts = np.asarray(atoms_res["sts"], dtype=np.float32)
    
    def compute_extra_atoms(self) -> None:
        """ Compute additional atoms based on the existing ones. """
        # Information dynamics atoms
        for extra_atom, dependencies in INFORMATION_DYNAMICS.items():
            extra_ts = np.zeros_like(self.xtx, dtype=np.float32)
            for dep in dependencies:
                multiplier = 1 if isinstance(dep, str) else dep[0]
                dep_atom = dep if isinstance(dep, str) else dep[1]
                extra_ts += multiplier * getattr(self, dep_atom)
            setattr(self, extra_atom, extra_ts)
        
        # Mutual informaiton normalized atoms
        mi = self.mutual_information
        for atom in self.get_atoms_names():
            setattr(self, f"{atom}_normalized", getattr(self, atom) / mi)

        # Synergy minus redundancy ranks
        syn_minus_red = self.sts - self.rtr 
        setattr(self, "syn_minus_red", syn_minus_red)
        syn_minus_red_rank = np.argsort(syn_minus_red)[::-1]  # descending order
        setattr(self, "syn_minus_red_rank", syn_minus_red_rank)

    def get_atoms_names(self) -> List[str]:
        """Return the names of the atoms in this PhyIDTimeSeries."""
        return sorted(k for k, v in vars(self).items() if isinstance(v, (list, np.ndarray, float, xr.DataArray)))