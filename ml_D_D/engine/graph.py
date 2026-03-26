"""
graph.py
Graph traversal and validation for all three pipeline tabs.

This module provides the core graph infrastructure for the ML drag-and-drop
pipeline builder. It is responsible for constructing in-memory graph
representations from live DearPyGui state, performing topological ordering of
nodes, and running role-specific structural validation across all three pipeline
tabs (Data Prep, Model, and Training).

Data structures:

    GraphNode        - A fully resolved node: its tag, block label, block
                       definition, parameter values read live from DearPyGui,
                       and which pins are connected.

    ValidationResult - Aggregated list of Issue objects returned by
                       validate_pipeline(). Severity is "error" (blocks
                       training) or "warning" (advisory only).

Public API:

    build_graph(tab)          → dict[ntag -> GraphNode]
    topological_sort(tab)     → list[GraphNode]  (raises CycleError if cyclic)
    validate_pipeline()       → ValidationResult
    get_tab_by_role(role)     → tab dict | None

Role expectations:

    data_prep  : must have at least one Dataset source node and a train
                 DataLoader node; augmentation nodes must be connected.
    model      : must have exactly one Input and one Output node; all layer
                 nodes must be reachable from Input.
    training   : must have exactly one Loss and one Optimizer node, a
                 ModelBlock, and a DataLoaderBlock, all wired correctly.
"""

from __future__ import annotations

import dearpygui.dearpygui as dpg
from dataclasses import dataclass, field
from typing import Optional

import ml_D_D.state as state
from ml_D_D.engine.blocks import get_block_def


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    """
    An immutable snapshot of a single pipeline node and its connectivity.

    A ``GraphNode`` is built by :func:`build_graph` from the live DearPyGui
    item tree.  It captures both the *static* information stored in the tab
    state (block label, pin names) and the *dynamic* information polled from
    DearPyGui at construction time (current param field values, which pins
    currently have incoming/outgoing links).

    Attributes
    ----------
    ntag : str
        Unique DearPyGui alias for the node widget, formatted as
        ``"node_{tid}_{nid}"``.
    block_label : str
        Human-readable block name that maps to a block definition via
        :func:`~ml_D_D.engine.blocks.get_block_def`, e.g. ``"Conv2d"`` or
        ``"CrossEntropyLoss"``.
    params : dict[str, str]
        Mapping of *parameter name* → *current string value* for every param
        field defined by the block definition.  Values are stripped of
        surrounding whitespace.  An empty string indicates the field has not
        been filled in.
    inputs : list[str]
        Ordered list of input pin names declared by the block definition.
        These are the *possible* input pins; see ``connected_inputs`` for
        which ones are currently wired.
    outputs : list[str]
        Ordered list of output pin names declared by the block definition.
    connected_inputs : set[str]
        Subset of ``inputs`` whose corresponding DearPyGui attribute currently
        has at least one incoming link.
    connected_outputs : set[str]
        Subset of ``outputs`` whose corresponding DearPyGui attribute currently
        has at least one outgoing link.
    """

    ntag:        str
    block_label: str
    params:      dict[str, str]   # param_name -> current field value
    inputs:      list[str]        # pin names defined by block
    outputs:     list[str]
    connected_inputs:  set[str]   # pin names that have an incoming link
    connected_outputs: set[str]   # pin names that have an outgoing link


@dataclass
class Issue:
    """
    A single validation finding attached to the pipeline.

    Issues are collected into a :class:`ValidationResult` by the validator
    functions.  They carry a severity level, a human-readable message, and an
    optional reference to the offending node.

    Attributes
    ----------
    severity : str
        Either ``"error"`` (must be fixed before training can start) or
        ``"warning"`` (advisory; training may still proceed).
    message : str
        A concise, user-facing description of the problem.
    ntag : str or None
        DearPyGui node alias of the offending node, or ``None`` when the issue
        is tab-level rather than node-level.
    """

    severity: str        # "error" | "warning"
    message:  str
    ntag:     Optional[str] = None   # node tag if applicable


@dataclass
class ValidationResult:
    """
    Aggregated outcome of a full pipeline validation pass.

    Returned by :func:`validate_pipeline` after checking all three pipeline
    tabs.  Callers should inspect :attr:`ok` first; if ``False``, iterate
    :attr:`errors` to surface blocking issues.

    Attributes
    ----------
    issues : list[Issue]
        All issues found during validation, in the order they were added.
        Mix of errors and warnings.

    Properties
    ----------
    errors : list[Issue]
        Filtered view of :attr:`issues` containing only error-severity items.
    warnings : list[Issue]
        Filtered view of :attr:`issues` containing only warning-severity items.
    ok : bool
        ``True`` iff there are zero error-severity issues.  Warnings alone do
        not make ``ok`` return ``False``.
    """

    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        """Return only the error-severity issues."""
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        """Return only the warning-severity issues."""
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        """``True`` when no error-severity issues are present."""
        return len(self.errors) == 0

    def add_error(self, msg: str, ntag: str | None = None) -> None:
        """
        Append an error-severity issue.

        Parameters
        ----------
        msg : str
            Human-readable description of the error.
        ntag : str, optional
            DearPyGui alias of the node responsible for the error.
        """
        self.issues.append(Issue("error", msg, ntag))

    def add_warning(self, msg: str, ntag: str | None = None) -> None:
        """
        Append a warning-severity issue.

        Parameters
        ----------
        msg : str
            Human-readable description of the warning.
        ntag : str, optional
            DearPyGui alias of the node responsible for the warning.
        """
        self.issues.append(Issue("warning", msg, ntag))


class CycleError(Exception):
    """
    Raised by :func:`topological_sort` when the node graph contains a cycle.

    A cycle means at least one node is (directly or transitively) connected to
    itself, which makes forward execution order impossible to determine.
    """
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_tab_by_role(role: str) -> dict | None:
    """
    Look up the first pipeline tab that has been assigned a given role.

    Tabs are stored in ``state.tabs`` as a mapping of tab-id → tab-dict.
    Each tab-dict may contain a ``"role"`` key whose value is one of
    ``"data_prep"``, ``"model"``, or ``"training"``.

    Parameters
    ----------
    role : str
        The role string to search for (e.g. ``"model"``).

    Returns
    -------
    dict or None
        The tab dict for the first matching tab, or ``None`` if no tab carries
        the requested role.

    Examples
    --------
    >>> tab = get_tab_by_role("training")
    >>> if tab is None:
    ...     print("No training tab configured.")
    """
    for t in state.tabs.values():
        if t.get("role") == role:
            return t
    return None


def _read_params(ntag: str, block_label: str) -> dict[str, str]:
    """
    Read all parameter field values for a node directly from DearPyGui.

    The function derives the DearPyGui item aliases for every param input
    widget belonging to the node and polls their current values.  Fields that
    do not exist in DearPyGui (e.g. because the UI was not fully initialised)
    are returned as empty strings.

    Parameters
    ----------
    ntag : str
        Node alias, formatted as ``"node_{tid}_{nid}"``.
    block_label : str
        Block name used to look up the list of expected parameters via
        :func:`~ml_D_D.engine.blocks.get_block_def`.

    Returns
    -------
    dict[str, str]
        Mapping of *param_name* → *stripped string value*.  Returns an empty
        dict if the block definition cannot be found.

    Notes
    -----
    Parameter field aliases follow the convention::

        node_{tid}_{nid}_input_{param_name}
    """
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
    Decompose a pin attribute alias into its owning node tag and pin name.

    Pin attribute aliases follow the format::

        node_{tid}_{nid}_{direction}_{pin_name}

    where *direction* is either ``"in"`` or ``"out"``.

    Parameters
    ----------
    attr_tag : str
        Full DearPyGui alias of a node attribute (pin), e.g.
        ``"node_0_3_in_images"``.

    Returns
    -------
    tuple[str, str] or None
        A ``(node_tag, pin_name)`` pair where *node_tag* is
        ``"node_{tid}_{nid}"`` and *pin_name* is everything after the
        direction segment.  Returns ``None`` if the alias has fewer than five
        underscore-delimited parts and therefore cannot be parsed.

    Examples
    --------
    >>> _pin_owner("node_0_3_in_images")
    ('node_0_3', 'images')
    >>> _pin_owner("node_1_2_out_predictions")
    ('node_1_2', 'predictions')
    """
    # format: node_<tid>_<nid>_in_<pin>  or  node_<tid>_<nid>_out_<pin>
    parts = attr_tag.split("_")
    if len(parts) < 5:
        return None
    # node_{tid}_{nid}_{direction}_{pin...}
    ntag = f"node_{parts[1]}_{parts[2]}"
    pin  = "_".join(parts[4:])
    return ntag, pin


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(tab: dict) -> dict[str, GraphNode]:
    """
    Construct a fully resolved graph from a tab's live DearPyGui state.

    The function performs two passes over the tab data:

    1. **Connectivity pass** – iterates all links stored in ``tab["links"]``
       and records, for each node, which of its input and output pins are
       currently connected.
    2. **Node pass** – iterates all entries in ``tab["nodes"]``, resolves
       the block definition, reads live parameter values, and produces a
       :class:`GraphNode` snapshot.

    Integer DearPyGui item IDs are transparently resolved to string aliases
    using :func:`dpg.get_item_alias` so callers never need to handle both
    formats.

    Parameters
    ----------
    tab : dict
        A tab dict from ``state.tabs``.  Expected keys:

        * ``"nodes"`` – mapping of *ntag* → node info (string label or dict
          with ``"label"`` key).
        * ``"links"`` – mapping of *link_id* → ``(attr1_tag, attr2_tag)``
          where *attr1* is the source output pin and *attr2* is the
          destination input pin.

    Returns
    -------
    dict[str, GraphNode]
        Mapping of node alias → :class:`GraphNode` for every node currently
        present in the tab.  An empty dict is returned if the tab has no
        nodes.

    Notes
    -----
    The returned snapshot reflects DearPyGui state at the **moment of the
    call**.  It becomes stale as soon as the user modifies the canvas.
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
        """Extract the block label string from a node info entry."""
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


# ---------------------------------------------------------------------------
# Topological sort — Kahn's algorithm
# ---------------------------------------------------------------------------

def topological_sort(tab: dict) -> list[GraphNode]:
    """
    Return the nodes of a tab's graph in a valid execution order.

    Uses **Kahn's algorithm** (iterative in-degree reduction) to produce a
    linear ordering in which every node appears before all nodes that depend
    on it (i.e. sources before sinks).

    Isolated nodes (no incoming *or* outgoing links) are naturally appended
    at the end of the sorted list, since they start with an in-degree of zero
    and are processed when the queue first empties.

    Parameters
    ----------
    tab : dict
        A tab dict from ``state.tabs``.  See :func:`build_graph` for the
        expected structure.

    Returns
    -------
    list[GraphNode]
        All nodes in topological order.  An empty list is returned for a tab
        with no nodes.

    Raises
    ------
    CycleError
        If the graph contains one or more directed cycles, making a linear
        execution order impossible.

    Notes
    -----
    The adjacency map is built directly from ``tab["links"]``, so only
    *explicit* DearPyGui links create ordering constraints.  Nodes whose
    block definitions cannot be resolved (``get_block_def`` returns ``None``)
    are included in the sort but do not contribute edges.

    Algorithm complexity is O(V + E) where V is the number of nodes and E is
    the number of links.

    Examples
    --------
    >>> try:
    ...     order = topological_sort(model_tab)
    ... except CycleError:
    ...     print("Fix the cycle before running.")
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


# ---------------------------------------------------------------------------
# Validators per tab role
# ---------------------------------------------------------------------------

# Params that are considered optional and will not generate an "unfilled" warning.
_OPTIONAL_PARAMS: set[str] = {"padding", "bias", "eps", "momentum",
                               "weight_decay", "betas", "ignore_index", "weight"}

_DATALOADER_BLOCKS = {"DataLoader (train)", "DataLoader (val)", "DataLoader"}
_DATASET_BLOCKS    = {"MNIST","CIFAR10","CIFAR100","FashionMNIST","ImageFolder"}
_AUG_BLOCKS        = {"Resize","CenterCrop","RandomCrop","RandomHFlip","RandomVFlip",
                      "ColorJitter","RandomRotation","GaussianBlur","RandomErasing",
                      "Normalize","ToTensor","Grayscale"}
_LOSS_BLOCKS       = {"CrossEntropyLoss","MSELoss","BCELoss","BCEWithLogits",
                      "NLLLoss","HuberLoss","KLDivLoss"}
_OPTIMIZER_BLOCKS  = {"Adam","AdamW","SGD","RMSprop","Adagrad","LBFGS"}


def _validate_params(graph: dict[str, GraphNode], result: ValidationResult) -> None:
    """
    Check parameter fields on every node and emit warnings or errors.

    Applies two tiers of reporting:

    * **Error** – *all* non-optional parameters of a node are empty, which
      means the node is completely unconfigured and would certainly fail at
      runtime.
    * **Warning** – *some* (but not all) non-optional parameters are empty,
      indicating a partially configured node that may still fail.

    Parameters that appear in :data:`_OPTIONAL_PARAMS` are silently skipped
    because they have sensible defaults defined by the underlying PyTorch
    layer or transform.

    Parameters
    ----------
    graph : dict[str, GraphNode]
        Node graph as returned by :func:`build_graph`.
    result : ValidationResult
        Accumulator to which issues are appended in-place.

    Notes
    -----
    Nodes whose block definitions have no parameters (``node.params`` is
    empty) are skipped entirely.
    """
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


def _validate_data_prep(tab: dict, result: ValidationResult) -> None:
    """
    Validate the structural correctness of the Data Prep pipeline tab.

    Checks (in order):

    1. The tab has at least one node.
    2. At least one Dataset source node is present
       (MNIST, CIFAR10, CIFAR100, FashionMNIST, or ImageFolder).
    3. At least one ``DataLoader (train)`` (or generic ``DataLoader``) node
       is present — a val loader alone is insufficient.
    4. Every Dataset node's ``img`` output pin is connected downstream.
    5. Every DataLoader node's ``img`` input pin is connected — the chain must
       reach the loader.
    6. Augmentation nodes that have no connections at all (orphaned) are
       flagged as warnings.
    7. The graph is acyclic (uses :func:`topological_sort`).
    8. All required parameter fields are filled
       (delegated to :func:`_validate_params`).

    Parameters
    ----------
    tab : dict
        Tab dict with ``role == "data_prep"``.
    result : ValidationResult
        Accumulator to which issues are appended in-place.
    """
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
    """
    Validate the structural correctness of the Model architecture tab.

    Checks (in order):

    1. The tab has at least one node.
    2. Exactly one ``Input`` node is present (zero or more than one → error).
    3. Exactly one ``Output`` node is present (zero or more than one → error).
    4. Any non-Input / non-Output node with *no* connections at all is flagged
       as an orphaned warning.
    5. Any non-Input node that has unconnected input pins is flagged as a
       warning (each missing pin is listed).
    6. The graph is acyclic (uses :func:`topological_sort`).
    7. All required parameter fields are filled
       (delegated to :func:`_validate_params`).

    Parameters
    ----------
    tab : dict
        Tab dict with ``role == "model"``.
    result : ValidationResult
        Accumulator to which issues are appended in-place.
    """
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


def _validate_training(tab: dict, result: ValidationResult) -> None:
    """
    Validate the structural correctness of the Training configuration tab.

    The training graph must contain exactly the following four block types,
    wired together in the canonical data-flow pattern:

    .. code-block:: text

        DataLoaderBlock ──images──► ModelBlock ──predictions──► LossNode ──loss──► Optimizer
        DataLoaderBlock ──labels─────────────────────────────► LossNode (target)

    Checks performed:

    1. A ``ModelBlock`` node is present.
    2. A ``DataLoaderBlock`` node is present.
    3. At least one Loss Function node is present
       (warning if more than one — only the first is used).
    4. At least one Optimizer node is present
       (warning if more than one — only the first is used).
    5. ``DataLoaderBlock.images`` output → ``ModelBlock.images`` input.
    6. ``ModelBlock.predictions`` output → ``LossNode.pred`` input.
    7. ``DataLoaderBlock.labels`` output → ``LossNode.target`` input.
    8. ``LossNode.loss`` output → ``Optimizer.params`` input.
    9. All required parameter fields are filled
       (delegated to :func:`_validate_params`).

    Parameters
    ----------
    tab : dict
        Tab dict with ``role == "training"``.
    result : ValidationResult
        Accumulator to which issues are appended in-place.

    Notes
    -----
    Connectivity checks are performed only when the relevant nodes actually
    exist, so the function produces at most one error per missing block rather
    than cascading failures.
    """
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


# ---------------------------------------------------------------------------
# Public: Validate Pipeline
# ---------------------------------------------------------------------------

def validate_pipeline() -> ValidationResult:
    """
    Run a full structural validation pass across all three pipeline tabs.

    Iterates the three required pipeline roles — ``"data_prep"``,
    ``"model"``, and ``"training"`` — and delegates to the corresponding
    role-specific validator.  If a tab for a role cannot be found in
    ``state.tabs``, a top-level error is added for that role.

    Returns
    -------
    ValidationResult
        Aggregated result containing every issue found across all tabs.
        Check :attr:`~ValidationResult.ok` to determine whether training can
        proceed; inspect :attr:`~ValidationResult.errors` and
        :attr:`~ValidationResult.warnings` for actionable feedback.

    Examples
    --------
    >>> result = validate_pipeline()
    >>> if not result.ok:
    ...     for issue in result.errors:
    ...         print(f"[ERROR] {issue.message}")
    ... else:
    ...     print("Pipeline is valid — ready to train.")

    See Also
    --------
    _validate_data_prep : Data Prep tab validator.
    _validate_model     : Model tab validator.
    _validate_training  : Training tab validator.
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