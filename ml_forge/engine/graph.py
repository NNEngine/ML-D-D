"""
graph.py
Graph traversal and validation for all three pipeline tabs.

Data structures:

GraphNode  - a fully resolved node: its tag, block label, block def, param
             values read live from DearPyGui, and which pins are connected.

ValidationResult - list of Issue objects returned by validate_pipeline().
             Severity is "error" (blocks training) or "warning" (advisory).

Public API

  build_graph(tab)          → dict[ntag -> GraphNode]
  topological_sort(tab)     → list[GraphNode]   (raises CycleError if cyclic)
  validate_pipeline()       → ValidationResult
  get_tab_by_role(role)     → tab dict | None

Role expectations

  data_prep  : must have at least one DataLoader node
  model      : must have exactly one Input and one Output node,
               all layer nodes must be reachable from Input
  training   : must have exactly one Loss and one Optimizer node
"""

from __future__ import annotations

import dearpygui.dearpygui as dpg
from dataclasses import dataclass, field
from typing import Optional

import ml_forge.state as state
from ml_forge.engine.blocks import get_block_def


#  Data structures

@dataclass
class GraphNode:
    ntag:        str
    block_label: str
    params:      dict[str, str]   # param_name -> current field value
    inputs:      list[str]        # pin names defined by block
    outputs:     list[str]
    connected_inputs:  set[str]   # pin names that have an incoming link
    connected_outputs: set[str]   # pin names that have an outgoing link


@dataclass
class Issue:
    severity: str        # "error" | "warning"
    message:  str
    ntag:     Optional[str] = None   # node tag if applicable


@dataclass
class ValidationResult:
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, msg: str, ntag: str | None = None) -> None:
        self.issues.append(Issue("error", msg, ntag))

    def add_warning(self, msg: str, ntag: str | None = None) -> None:
        self.issues.append(Issue("warning", msg, ntag))


class CycleError(Exception):
    pass


#  Helpers

def get_tab_by_role(role: str) -> dict | None:
    """Return the first tab with the given role, or None."""
    for t in state.tabs.values():
        if t.get("role") == role:
            return t
    return None


def _read_params(ntag: str, block_label: str) -> dict[str, str]:
    """Read all param field values live from DearPyGui for a node."""
    block = get_block_def(block_label)
    if not block:
        return {}
    parts = ntag.split("_")          # node_{tid}_{nid}
    tid_s, nid_s = parts[1], parts[2]
    vals = {}
    for param in block["params"]:
        ftag = f"node_{tid_s}_{nid_s}_input_{param}"
        vals[param] = dpg.get_value(ftag).strip() if dpg.does_item_exist(ftag) else ""
    return vals


def _pin_owner(attr_tag: str) -> tuple[str, str] | None:
    """
    Given an attribute tag like node_{tid}_{nid}_in_{pin}
    return (node_tag, pin_name) or None if unparseable.
    """
    # format: node_<tid>_<nid>_in_<pin>  or  node_<tid>_<nid>_out_<pin>
    parts = attr_tag.split("_")
    if len(parts) < 5:
        return None
    # node_{tid}_{nid}_{direction}_{pin...}
    ntag = f"node_{parts[1]}_{parts[2]}"
    pin  = "_".join(parts[4:])
    return ntag, pin


#  Graph builder

def build_graph(tab: dict) -> dict[str, GraphNode]:
    """
    Build a dict of GraphNode objects from a tab's live node/link state.
    Reads current param field values and computes which pins are connected.
    """
    # First pass: which pins have connections?
    connected_in:  dict[str, set[str]] = {}   # ntag -> set of connected input pin names
    connected_out: dict[str, set[str]] = {}   # ntag -> set of connected output pin names

    for ntag in tab["nodes"]:
        connected_in[ntag]  = set()
        connected_out[ntag] = set()

    for a1, a2 in tab["links"].values():
        # DearPyGui may store endpoints as integer item IDs — resolve to aliases
        if isinstance(a1, int):
            a1 = dpg.get_item_alias(a1) or ""
        if isinstance(a2, int):
            a2 = dpg.get_item_alias(a2) or ""
        src = _pin_owner(a1)
        dst = _pin_owner(a2)
        if src:
            connected_out[src[0]].add(src[1])
        if dst:
            connected_in[dst[0]].add(dst[1])

    # Second pass: build GraphNode objects
    # node_info may be a plain string (legacy) or {"label": ..., "theme": ...}
    def _label(node_info) -> str:
        return node_info["label"] if isinstance(node_info, dict) else str(node_info)

    graph: dict[str, GraphNode] = {}
    for ntag, node_info in tab["nodes"].items():
        block_label = _label(node_info)
        block  = get_block_def(block_label)
        params = _read_params(ntag, block_label)
        graph[ntag] = GraphNode(
            ntag=ntag,
            block_label=block_label,
            params=params,
            inputs=block["inputs"]  if block else [],
            outputs=block["outputs"] if block else [],
            connected_inputs=connected_in.get(ntag,  set()),
            connected_outputs=connected_out.get(ntag, set()),
        )
    return graph


# topoligical sort - kahn's

def topological_sort(tab: dict) -> list[GraphNode]:
    """
    Return nodes in execution order (sources first).
    Raises CycleError if the graph contains a cycle.
    Only includes nodes that are part of the connected graph.
    Isolated nodes (no connections at all) are appended at the end.
    """
    graph = build_graph(tab)
    if not graph:
        return []

    # Build adjacency: ntag -> list[ntag] (successors)
    # and in-degree count
    successors:  dict[str, list[str]] = {n: [] for n in graph}
    in_degree:   dict[str, int]       = {n: 0  for n in graph}

    # Map from attr_tag -> ntag for quick lookup
    attr_to_node: dict[str, str] = {}
    for ntag in graph:
        parts = ntag.split("_")
        tid_s, nid_s = parts[1], parts[2]
        block = get_block_def(graph[ntag].block_label)
        if not block:
            continue
        for pin in block["inputs"]:
            attr_to_node[f"node_{tid_s}_{nid_s}_in_{pin}"] = ntag
        for pin in block["outputs"]:
            attr_to_node[f"node_{tid_s}_{nid_s}_out_{pin}"] = ntag

    for a1, a2 in tab["links"].values():
        if isinstance(a1, int):
            a1 = dpg.get_item_alias(a1) or ""
        if isinstance(a2, int):
            a2 = dpg.get_item_alias(a2) or ""
        src_node = attr_to_node.get(a1)
        dst_node = attr_to_node.get(a2)
        if src_node and dst_node and src_node != dst_node:
            successors[src_node].append(dst_node)
            in_degree[dst_node] += 1

    # Kahn's algorithm
    queue  = [n for n, d in in_degree.items() if d == 0]
    sorted_nodes: list[GraphNode] = []

    while queue:
        ntag = queue.pop(0)
        sorted_nodes.append(graph[ntag])
        for succ in successors[ntag]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    if len(sorted_nodes) != len(graph):
        raise CycleError("Graph contains a cycle — cannot sort.")

    return sorted_nodes

#  Validators per tab role

# Nodes that count as "required params" — any param that is not optional
# For now: any param field that exists should be filled (empty = warning)
_OPTIONAL_PARAMS: set[str] = {"padding", "bias", "eps", "momentum",
                               "weight_decay", "betas", "ignore_index", "weight"}


def _validate_params(graph: dict[str, GraphNode], result: ValidationResult) -> None:
    """Warn on empty required params, error on completely unfilled nodes."""
    for node in graph.values():
        if not node.params:
            continue
        empty = [p for p, v in node.params.items()
                 if not v and p not in _OPTIONAL_PARAMS]
        if len(empty) == len(node.params):
            result.add_error(
                f"{node.block_label}: all parameters are empty.",
                node.ntag,
            )
        elif empty:
            result.add_warning(
                f"{node.block_label}: params not filled — {', '.join(empty)}.",
                node.ntag,
            )


_DATALOADER_BLOCKS = {"DataLoader (train)", "DataLoader (val)", "DataLoader"}
_DATASET_BLOCKS    = {"MNIST","CIFAR10","CIFAR100","FashionMNIST","ImageFolder"}
_AUG_BLOCKS        = {"Resize","CenterCrop","RandomCrop","RandomHFlip","RandomVFlip",
                      "ColorJitter","RandomRotation","GaussianBlur","RandomErasing",
                      "Normalize","ToTensor","Grayscale"}


def _validate_data_prep(tab: dict, result: ValidationResult) -> None:
    graph = build_graph(tab)
    if not graph:
        result.add_error("Data Prep tab has no nodes.")
        return

    nodes = list(graph.values())

    # Must have exactly one dataset source per chain
    dataset_nodes = [n for n in nodes if n.block_label in _DATASET_BLOCKS]
    if len(dataset_nodes) == 0:
        result.add_error("Data Prep tab needs a Dataset node (e.g. CIFAR10, ImageFolder).")
        return

    # Must have at least a train DataLoader
    train_loaders = [n for n in nodes if n.block_label in ("DataLoader (train)", "DataLoader")]
    val_loaders   = [n for n in nodes if n.block_label == "DataLoader (val)"]
    all_loaders   = [n for n in nodes if n.block_label in _DATALOADER_BLOCKS]

    if not train_loaders:
        result.add_error("Data Prep tab needs at least a DataLoader (train) node.")
        return

    # Check each dataset output is connected
    for ds in dataset_nodes:
        if "img" not in ds.connected_outputs:
            result.add_warning(
                f"{ds.block_label}: output is not connected.",
                ds.ntag,
            )

    # Check all dataloaders have connected inputs
    for loader in all_loaders:
        if "img" not in loader.connected_inputs:
            result.add_warning(
                f"{loader.block_label}: input is not connected — wire the chain into it.",
                loader.ntag,
            )

    # Check augmentation nodes are not orphaned
    for node in nodes:
        if node.block_label not in _AUG_BLOCKS:
            continue
        if not node.connected_inputs and not node.connected_outputs:
            result.add_warning(
                f"{node.block_label}: node is not connected to anything.",
                node.ntag,
            )

    # Cycle check
    try:
        topological_sort(tab)
    except CycleError:
        result.add_error("Data Prep graph contains a cycle.")

    _validate_params(graph, result)


def _validate_model(tab: dict, result: ValidationResult) -> None:
    graph = build_graph(tab)
    if not graph:
        result.add_error("Model tab has no nodes.")
        return

    inputs  = [n for n in graph.values() if n.block_label == "Input"]
    outputs = [n for n in graph.values() if n.block_label == "Output"]

    if len(inputs) == 0:
        result.add_error("Model tab needs an Input node.")
    elif len(inputs) > 1:
        result.add_error(f"Model tab has {len(inputs)} Input nodes — only one allowed.")

    if len(outputs) == 0:
        result.add_error("Model tab needs an Output node.")
    elif len(outputs) > 1:
        result.add_error(f"Model tab has {len(outputs)} Output nodes — only one allowed.")

    # Check for orphaned nodes (no connections at all)
    for node in graph.values():
        if node.block_label in ("Input", "Output"):
            continue
        if not node.connected_inputs and not node.connected_outputs:
            result.add_warning(
                f"{node.block_label}: node is not connected to anything.",
                node.ntag,
            )

    # Check for disconnected inputs on non-source nodes
    for node in graph.values():
        if node.block_label == "Input":
            continue
        missing = [p for p in node.inputs if p not in node.connected_inputs]
        if missing:
            result.add_warning(
                f"{node.block_label}: input pin(s) not connected — {', '.join(missing)}.",
                node.ntag,
            )

    # Cycle check
    try:
        topological_sort(tab)
    except CycleError:
        result.add_error("Model graph contains a cycle.")

    _validate_params(graph, result)


_LOSS_BLOCKS      = {"CrossEntropyLoss","MSELoss","BCELoss","BCEWithLogits",
                     "NLLLoss","HuberLoss","KLDivLoss"}
_OPTIMIZER_BLOCKS = {"Adam","AdamW","SGD","RMSprop","Adagrad","LBFGS"}


def _validate_training(tab: dict, result: ValidationResult) -> None:
    graph = build_graph(tab)
    if not graph:
        result.add_error("Training tab has no nodes.")
        return

    nodes = list(graph.values())

    model_nodes  = [n for n in nodes if n.block_label == "ModelBlock"]
    loader_nodes = [n for n in nodes if n.block_label == "DataLoaderBlock"]
    losses       = [n for n in nodes if n.block_label in _LOSS_BLOCKS]
    optimizers   = [n for n in nodes if n.block_label in _OPTIMIZER_BLOCKS]

    if not model_nodes:
        result.add_error("Training tab needs a ModelBlock node (find it in the palette under Training).")
    if not loader_nodes:
        result.add_error("Training tab needs a DataLoaderBlock node (find it in the palette under Training).")

    if not losses:
        result.add_error("Training tab needs a Loss Function node.")
    elif len(losses) > 1:
        result.add_warning(f"{len(losses)} loss nodes found - only the first will be used.")

    if not optimizers:
        result.add_error("Training tab needs an Optimizer node.")
    elif len(optimizers) > 1:
        result.add_warning(f"{len(optimizers)} optimizer nodes found - only the first will be used.")

    model_node  = model_nodes[0]  if model_nodes  else None
    loader_node = loader_nodes[0] if loader_nodes else None
    loss_node   = losses[0]       if losses       else None
    optim_node  = optimizers[0]   if optimizers   else None

    if model_node and loader_node:
        if "images" not in loader_node.connected_outputs:
            result.add_error(
                "DataLoaderBlock: images output must be connected to ModelBlock images input.",
                loader_node.ntag,
            )
        if "images" not in model_node.connected_inputs:
            result.add_error(
                "ModelBlock: images input must be connected from DataLoaderBlock images output.",
                model_node.ntag,
            )

    if model_node and loss_node:
        if "predictions" not in model_node.connected_outputs:
            result.add_error(
                "ModelBlock: predictions output must be connected to Loss pred input.",
                model_node.ntag,
            )
        if "pred" not in loss_node.connected_inputs:
            result.add_error(
                f"{loss_node.block_label}: pred input must be connected from ModelBlock predictions.",
                loss_node.ntag,
            )

    if loader_node and loss_node:
        if "labels" not in loader_node.connected_outputs:
            result.add_error(
                "DataLoaderBlock: labels output must be connected to Loss target input.",
                loader_node.ntag,
            )
        if "target" not in loss_node.connected_inputs:
            result.add_error(
                f"{loss_node.block_label}: target input must be connected from DataLoaderBlock labels.",
                loss_node.ntag,
            )

    if loss_node and optim_node:
        if "loss" not in loss_node.connected_outputs:
            result.add_error(
                f"{loss_node.block_label}: loss output must be connected to Optimizer params input.",
                loss_node.ntag,
            )
        if "params" not in optim_node.connected_inputs:
            result.add_error(
                f"{optim_node.block_label}: params input must be connected from Loss loss output.",
                optim_node.ntag,
            )

    _validate_params(graph, result)


# Public: Validate Pipeline

def validate_pipeline() -> ValidationResult:
    """
    Validate all three pipeline tabs.
    Returns a ValidationResult — check .ok and .issues.
    """
    result = ValidationResult()

    for role, validator in (
        ("data_prep", _validate_data_prep),
        ("model",     _validate_model),
        ("training",  _validate_training),
    ):
        tab = get_tab_by_role(role)
        if tab is None:
            result.add_error(f"No tab assigned to the '{role}' role.")
        else:
            validator(tab, result)

    return result