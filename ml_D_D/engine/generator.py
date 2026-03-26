"""
generator.py
PyTorch training script generator.

This module is responsible for reading the live graph state from all three
pipeline tabs (Data Prep, Model, and Training) and emitting a complete,
self-contained, runnable ``train.py`` file.

The generated script includes:

* A ``Model`` class derived from ``nn.Module`` whose layers and forward pass
  mirror the node graph on the Model tab.
* A data pipeline section with ``transforms.Compose`` definitions, dataset
  instantiation, and ``DataLoader`` construction — supporting both a
  single-chain layout (one DataLoader) and a dual-chain layout (separate
  train and val DataLoaders with independent augmentation pipelines).
* Loss function and optimizer instantiation pulled from the Training tab.
* Reusable ``train`` and ``evaluate`` helper functions.
* A ``__main__`` training loop with per-epoch logging and final checkpoint
  saving.

Public API:

    generate_pytorch() -> str
        Assemble and return the full script as a string.

    export_pytorch()
        Open a DearPyGui file-dialog and write the script to a user-chosen
        ``.py`` file.  Wired to *File > Export > Python > PyTorch*.
"""

from __future__ import annotations
import textwrap
from ml_D_D.engine.graph import build_graph, topological_sort, get_tab_by_role, GraphNode


# ---------------------------------------------------------------------------
# PyTorch module name mapping
# ---------------------------------------------------------------------------
# Each entry maps a block label to a (torch_module, arg_template) pair.
# Curly-brace tokens in arg_template (e.g. ``{in_features}``) are replaced
# with the corresponding parameter value read from the node's param fields
# via :func:`_fill`.  Missing values fall back to :data:`_PARAM_DEFAULTS`.

_LAYER_MAP: dict[str, tuple[str, str]] = {
    # Linear layers
    "Linear":            ("nn.Linear",             "{in_features}, {out_features}"),
    "Conv2D":            ("nn.Conv2d",              "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}"),
    "ConvTranspose2D":   ("nn.ConvTranspose2d",     "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}"),
    "Flatten":           ("nn.Flatten",             ""),
    # Activation functions
    "ReLU":              ("nn.ReLU",                ""),
    "Sigmoid":           ("nn.Sigmoid",             ""),
    "Tanh":              ("nn.Tanh",                ""),
    "Softmax":           ("nn.Softmax",             "dim={dim}"),
    "GELU":              ("nn.GELU",                ""),
    "LeakyReLU":         ("nn.LeakyReLU",           "negative_slope={negative_slope}"),
    # Normalisation
    "BatchNorm2D":       ("nn.BatchNorm2d",         "{num_features}"),
    "LayerNorm":         ("nn.LayerNorm",           "{normalized_shape}"),
    "GroupNorm":         ("nn.GroupNorm",           "{num_groups}, {num_channels}"),
    "Dropout":           ("nn.Dropout",             "p={p}"),
    # Pooling
    "MaxPool2D":         ("nn.MaxPool2d",           "kernel_size={kernel_size}, stride={stride}, padding={padding}"),
    "AvgPool2D":         ("nn.AvgPool2d",           "kernel_size={kernel_size}, stride={stride}, padding={padding}"),
    "AdaptiveAvgPool2D": ("nn.AdaptiveAvgPool2d",   "{output_size}"),
}

_LOSS_MAP: dict[str, tuple[str, str]] = {
    "CrossEntropyLoss": ("nn.CrossEntropyLoss",     ""),
    "MSELoss":          ("nn.MSELoss",              "reduction='{reduction}'"),
    "BCELoss":          ("nn.BCELoss",              "reduction='{reduction}'"),
    "BCEWithLogits":    ("nn.BCEWithLogitsLoss",    "reduction='{reduction}'"),
    "NLLLoss":          ("nn.NLLLoss",              "reduction='{reduction}'"),
    "HuberLoss":        ("nn.HuberLoss",            "delta={delta}"),
    "KLDivLoss":        ("nn.KLDivLoss",            "reduction='{reduction}'"),
}

_OPTIM_MAP: dict[str, tuple[str, str]] = {
    "Adam":    ("optim.Adam",    "lr={lr}"),
    "AdamW":   ("optim.AdamW",   "lr={lr}, weight_decay={weight_decay}"),
    "SGD":     ("optim.SGD",     "lr={lr}, momentum={momentum}"),
    "RMSprop": ("optim.RMSprop", "lr={lr}"),
    "Adagrad": ("optim.Adagrad", "lr={lr}"),
    "LBFGS":   ("optim.LBFGS",  "lr={lr}"),
}

_DATASET_MAP: dict[str, tuple[str, str]] = {
    "MNIST":        ("datasets.MNIST",        "root='{root}', train={train}, download={download}, transform=transform"),
    "CIFAR10":      ("datasets.CIFAR10",      "root='{root}', train={train}, download={download}, transform=transform"),
    "CIFAR100":     ("datasets.CIFAR100",     "root='{root}', train={train}, download={download}, transform=transform"),
    "FashionMNIST": ("datasets.FashionMNIST", "root='{root}', train={train}, download={download}, transform=transform"),
    "ImageFolder":  ("ImageFolder",           "root='{root}', transform=transform"),
}

_AUGMENTATION_MAP: dict[str, tuple[str, str]] = {
    "RandomCrop":     ("transforms.RandomCrop",              "{size}, padding={padding}"),
    "RandomHFlip":    ("transforms.RandomHorizontalFlip",    "p={p}"),
    "RandomVFlip":    ("transforms.RandomVerticalFlip",      "p={p}"),
    "ColorJitter":    ("transforms.ColorJitter",             "brightness={brightness}, contrast={contrast}, saturation={saturation}, hue={hue}"),
    "RandomRotation": ("transforms.RandomRotation",          "{degrees}"),
    "Normalize":      ("transforms.Normalize",               "mean={mean}, std={std}"),
    "Resize":         ("transforms.Resize",                  "{size}"),
    "CenterCrop":     ("transforms.CenterCrop",              "{size}"),
    "ToTensor":       ("transforms.ToTensor",                ""),
    "GaussianBlur":   ("transforms.GaussianBlur",            "{kernel_size}, sigma={sigma}"),
    "RandomErasing":  ("transforms.RandomErasing",           "p={p}"),
}


# ---------------------------------------------------------------------------
# Param defaults
# ---------------------------------------------------------------------------
# Values used when a node's param field is left empty.  Mirrors the sensible
# PyTorch constructor defaults so the generated script runs out of the box.

_PARAM_DEFAULTS: dict[str, str] = {
    # Layers
    "kernel_size":      "3",
    "stride":           "1",
    "padding":          "0",
    "num_layers":       "1",
    # Flatten
    "start_dim":        "1",
    "end_dim":          "-1",
    # Activations
    "dim":              "1",
    "negative_slope":   "0.01",
    # Normalisation
    "eps":              "1e-5",
    "momentum":         "0.1",
    "p":                "0.5",
    # Pooling
    "output_size":      "(1, 1)",
    # Loss
    "reduction":        "'mean'",
    "delta":            "1.0",
    # Optimizers
    "lr":               "1e-3",
    "weight_decay":     "0",
    "alpha":            "0.99",
    "lr_decay":         "0",
    "max_iter":         "20",
    "history_size":     "100",
    # Schedulers
    "step_size":        "10",
    "gamma":            "0.1",
    "T_max":            "50",
    "eta_min":          "0",
    "mode":             "'min'",
    "factor":           "0.1",
    "patience":         "10",
    # Augmentation
    "size":             "224",
    "degrees":          "15",
    "brightness":       "0",
    "contrast":         "0",
    "saturation":       "0",
    "hue":              "0",
    "mean":             "[0.485, 0.456, 0.406]",
    "std":              "[0.229, 0.224, 0.225]",
    "sigma":            "1.0",
    "scale":            "(0.02, 0.33)",
    "ratio":            "(0.3, 3.3)",
}


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _p(node: GraphNode, key: str, fallback: str = "...") -> str:
    """
    Resolve a single parameter value for a node, with layered fallback.

    Resolution order:

    1. The node's own param field value (stripped of surrounding whitespace).
    2. The global default from :data:`_PARAM_DEFAULTS` for that key.
    3. The explicit *fallback* argument (default ``"..."``).

    Parameters
    ----------
    node : GraphNode
        The node whose params are inspected.
    key : str
        Parameter name to look up (e.g. ``"lr"``, ``"kernel_size"``).
    fallback : str, optional
        Last-resort value if neither the node param nor the global default
        is available.  Defaults to ``"..."`` so the generated code remains
        syntactically valid but visually obvious as a placeholder.

    Returns
    -------
    str
        Resolved parameter value as a string, ready for direct insertion
        into generated source code.
    """
    val = node.params.get(key, "").strip()
    return val or _PARAM_DEFAULTS.get(key, fallback)


def _fill(template: str, node: GraphNode) -> str:
    """
    Substitute all ``{param}`` tokens in a template with node param values.

    For each parameter defined in ``node.params``, replaces the corresponding
    ``{key}`` token in *template* with the param's current value.  If the
    value is empty, the global default from :data:`_PARAM_DEFAULTS` is used
    instead; if no default exists either, the token is replaced with
    ``"..."``.

    Parameters
    ----------
    template : str
        Argument template string containing ``{param_name}`` placeholders,
        as stored in :data:`_LAYER_MAP`, :data:`_LOSS_MAP`, etc.
        Example: ``"{in_channels}, {out_channels}, kernel_size={kernel_size}"``.
    node : GraphNode
        Node whose ``params`` dict supplies the replacement values.

    Returns
    -------
    str
        Template with all known tokens replaced.  Tokens for which the node
        has no parameter entry are left untouched.

    Examples
    --------
    >>> _fill("{in_features}, {out_features}", node)
    '128, 10'
    """
    result = template
    for k, v in node.params.items():
        filled = v.strip() if v else _PARAM_DEFAULTS.get(k, "...")
        result = result.replace("{" + k + "}", filled)
    return result


def _safe_name(block_label: str, idx: int) -> str:
    """
    Derive a safe Python attribute name from a block label and an index.

    Converts the block label to lowercase, replaces spaces with underscores,
    and appends the numeric index to guarantee uniqueness within a class body.

    Parameters
    ----------
    block_label : str
        Human-readable block name, e.g. ``"BatchNorm2D"`` or
        ``"Conv2D"``.
    idx : int
        1-based positional index of the layer in the forward pass.

    Returns
    -------
    str
        A valid Python identifier, e.g. ``"batchnorm2d_2"`` or
        ``"conv2d_1"``.

    Examples
    --------
    >>> _safe_name("BatchNorm2D", 2)
    'batchnorm2d_2'
    >>> _safe_name("Conv2D", 1)
    'conv2d_1'
    """
    return block_label.lower().replace(" ", "_") + f"_{idx}"


def _I(n: int = 1) -> str:
    """
    Return *n* levels of Python indentation (4 spaces per level).

    Parameters
    ----------
    n : int, optional
        Number of indentation levels.  Defaults to ``1`` (four spaces).

    Returns
    -------
    str
        A string of ``4 * n`` space characters.
    """
    return "    " * n


# ---------------------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------------------

def _gen_model(tab: dict) -> str:
    """
    Generate the ``Model(nn.Module)`` class definition from the Model tab.

    Iterates nodes in topological order (falling back to dict insertion order
    if a cycle prevents sorting), filters out ``Input`` and ``Output``
    sentinel nodes, and emits:

    * An ``__init__`` method that registers each layer as a ``self.<attr>``
      attribute using the mapped ``nn.*`` constructor.
    * A ``forward`` method that chains the layers sequentially as ``x =
      self.<attr>(x)``.  LSTM layers receive special handling to unpack the
      ``(output, hidden)`` tuple.

    Layers whose block label is absent from :data:`_LAYER_MAP` generate a
    ``# NOTE:`` comment in ``__init__`` and are silently skipped in
    ``forward``.

    Parameters
    ----------
    tab : dict
        Tab dict with ``role == "model"`` from ``state.tabs``.

    Returns
    -------
    str
        A multi-line string containing the complete ``class Model`` definition,
        ready for insertion into the generated script.
    """
    try:
        nodes = topological_sort(tab)
    except Exception:
        nodes = list(build_graph(tab).values())

    # Filter to layer-type nodes only (skip Input/Output sentinels)
    layer_nodes = [n for n in nodes if n.block_label not in ("Input", "Output")]

    init_lines:    list[str] = []
    forward_lines: list[str] = []

    for idx, node in enumerate(layer_nodes, start=1):
        label = node.block_label
        attr  = _safe_name(label, idx)

        if label not in _LAYER_MAP:
            init_lines.append(f"# NOTE: No codegen mapping for '{label}'")
            continue

        module, template = _LAYER_MAP[label]
        args = _fill(template, node)
        init_lines.append(f"self.{attr} = {module}({args})")

        # LSTM returns (output, (h_n, c_n)) — unpack the hidden-state tuple
        if label == "LSTM":
            forward_lines.append(f"x, _ = self.{attr}(x)")
        else:
            forward_lines.append(f"x = self.{attr}(x)")

    # Retrieve input shape hint from the Input node if one exists
    input_node   = next((n for n in nodes if n.block_label == "Input"), None)
    input_shape  = _p(input_node, "shape", "# define input shape") if input_node else "# define input shape"

    lines = [
        "class Model(nn.Module):",
        f"{_I()}def __init__(self):",
        f"{_I(2)}super().__init__()",
    ]
    for l in init_lines:
        lines.append(f"{_I(2)}{l}")
    lines += [
        "",
        f"{_I()}def forward(self, x):",
        f"{_I(2)}# Input shape: {input_shape}",
    ]
    for l in forward_lines:
        lines.append(f"{_I(2)}{l}")
    lines.append(f"{_I(2)}return x")

    return "\n".join(lines)


def _gen_data(tab: dict) -> tuple[str, list[str]]:
    """
    Generate the data pipeline section from the Data Prep tab.

    Supports two layout modes, detected automatically from the node graph:

    **Mode A — Dual chain**
        Both a ``DataLoader (train)`` and a ``DataLoader (val)`` node are
        present.  Each loader has its own ancestor chain, so independent
        ``train_transform`` and ``val_transform`` ``Compose`` objects are
        emitted together with separate ``train_dataset`` / ``val_dataset``
        instantiations.

    **Mode B — Single chain**
        Only a single ``DataLoader (train)`` (or generic ``DataLoader``) node
        is present.  A single ``transform`` is emitted.  If a ``RandomSplit``
        node is found in the graph, the dataset is split using
        ``random_split``; otherwise a ``val_loader = None`` placeholder is
        emitted.

    Parameters
    ----------
    tab : dict
        Tab dict with ``role == "data_prep"`` from ``state.tabs``.

    Returns
    -------
    tuple[str, list[str]]
        A two-element tuple:

        * **main_code** (``str``) — The dataset instantiation and
          ``DataLoader`` construction lines, intended for placement inside the
          ``if __name__ == "__main__":`` block (will be indented by the
          caller).
        * **preamble_lines** (``list[str]``) — The ``transforms.Compose``
          (or bare ``transforms.ToTensor()``) definition lines, placed at
          module level *before* the ``Model`` class.

    Notes
    -----
    The ancestor-chain walk in ``_chain_for_loader`` performs a backwards
    reachability search from the loader node through ``tab["links"]``, so
    only augmentation and dataset nodes that feed *this specific* loader are
    included in its transform pipeline.
    """
    from ml_D_D.engine.graph import build_graph, topological_sort, _DATASET_BLOCKS, _AUG_BLOCKS

    try:
        ordered = topological_sort(tab)
    except Exception:
        ordered = list(build_graph(tab).values())

    graph = build_graph(tab)
    nodes = list(graph.values())

    def _chain_for_loader(loader_node):
        """
        Return topologically-ordered ancestor nodes that feed *loader_node*.

        Performs a backwards BFS/DFS over ``tab["links"]`` starting from
        *loader_node*, collecting every upstream node into an ``ancestors``
        set.  The result is filtered back through the topological order so
        the returned list respects execution sequence.

        Parameters
        ----------
        loader_node : GraphNode
            The DataLoader node whose upstream chain is to be traced.

        Returns
        -------
        list[GraphNode]
            Ancestor nodes in topological order (excluding *loader_node*
            itself).
        """
        targets   = {loader_node.ntag}
        ancestors = set()
        changed   = True
        while changed:
            changed = False
            for _, (a1, a2) in tab["links"].items():
                sp = a1.split("_"); dp = a2.split("_")
                if len(sp) >= 3 and len(dp) >= 3:
                    src = f"node_{sp[1]}_{sp[2]}"
                    dst = f"node_{dp[1]}_{dp[2]}"
                    if dst in targets and src not in ancestors:
                        ancestors.add(src)
                        targets.add(src)
                        changed = True
        return [n for n in ordered if n.ntag in ancestors]

    def _transform_lines(chain_nodes, var_name):
        """
        Emit lines that define a ``transforms.Compose`` for *var_name*.

        If the chain contains no augmentation nodes, a bare
        ``transforms.ToTensor()`` assignment is emitted instead.

        Parameters
        ----------
        chain_nodes : list[GraphNode]
            Ancestor nodes in execution order for a particular DataLoader.
        var_name : str
            Python variable name to assign the transform to (e.g.
            ``"train_transform"`` or ``"transform"``).

        Returns
        -------
        list[str]
            Source code lines (without trailing newlines).
        """
        aug = [n for n in chain_nodes if n.block_label in _AUGMENTATION_MAP]
        if not aug:
            return [f"{var_name} = transforms.ToTensor()"]
        lines = [f"{var_name} = transforms.Compose(["]
        for n in aug:
            module, template = _AUGMENTATION_MAP[n.block_label]
            args = _fill(template, n)
            lines.append(f"    {module}({args}),")
        lines.append("])")
        return lines

    def _dataset_line(ds_node, transform_var, var_name):
        """
        Emit a single line that instantiates a ``torchvision`` dataset.

        Looks up the dataset's constructor template from :data:`_DATASET_MAP`,
        fills in param values, and replaces the generic ``transform=transform``
        placeholder with the correct *transform_var* name.

        Parameters
        ----------
        ds_node : GraphNode
            The Dataset node (MNIST, CIFAR10, ImageFolder, etc.).
        transform_var : str
            Python name of the transform variable to pass to the dataset
            (e.g. ``"train_transform"``).
        var_name : str
            Python variable name to assign the dataset instance to
            (e.g. ``"train_dataset"``).

        Returns
        -------
        str
            A single source code line (without trailing newline).
        """
        label = ds_node.block_label
        if label not in _DATASET_MAP:
            return f"{var_name} = ...  # unsupported dataset: {label}"
        module, template = _DATASET_MAP[label]
        args = _fill(template, ds_node)
        # Replace the generic placeholder with the real transform variable name
        args = args.replace("transform=transform", f"transform={transform_var}")
        return f"{var_name} = {module}({args})"

    def _loader_line(loader_node, dataset_var, loader_var, shuffle_default):
        """
        Emit a single ``DataLoader(...)`` construction line.

        Reads ``batch_size``, ``num_workers``, ``pin_memory``, and ``shuffle``
        from the loader node's params, falling back to sensible defaults if
        any are empty.

        Parameters
        ----------
        loader_node : GraphNode
            The DataLoader node supplying constructor kwargs.
        dataset_var : str
            Python name of the dataset variable to wrap
            (e.g. ``"train_dataset"``).
        loader_var : str
            Python variable name for the resulting DataLoader instance
            (e.g. ``"train_loader"``).
        shuffle_default : str
            Stringified boolean default for the ``shuffle`` argument
            (``"True"`` for train loaders, ``"False"`` for val loaders).

        Returns
        -------
        str
            A single source code line (without trailing newline).
        """
        bs = _p(loader_node, "batch_size", "32")
        nw = _p(loader_node, "num_workers", "2")
        pm = _p(loader_node, "pin_memory", "True")
        sh = _p(loader_node, "shuffle", shuffle_default)
        return (f"{loader_var} = DataLoader({dataset_var}, batch_size={bs}, "
                f"shuffle={sh}, num_workers={nw}, pin_memory={pm})")

    # Detect layout mode
    train_loader_node = next(
        (n for n in nodes if n.block_label in ("DataLoader (train)", "DataLoader")), None)
    val_loader_node = next(
        (n for n in nodes if n.block_label == "DataLoader (val)"), None)

    preamble: list[str] = []
    main:     list[str] = []

    # ── Mode A: dual chain (separate train & val transforms) ──────────────
    if train_loader_node and val_loader_node:
        train_chain = _chain_for_loader(train_loader_node)
        val_chain   = _chain_for_loader(val_loader_node)

        train_ds_node = next((n for n in train_chain if n.block_label in _DATASET_BLOCKS), None)
        val_ds_node   = next((n for n in val_chain   if n.block_label in _DATASET_BLOCKS), None)

        preamble += _transform_lines(train_chain, "train_transform")
        preamble += [""]
        preamble += _transform_lines(val_chain, "val_transform")

        if train_ds_node:
            main.append(_dataset_line(train_ds_node, "train_transform", "train_dataset"))
        else:
            main.append("train_dataset = ...  # no dataset found in train chain")

        if val_ds_node:
            main.append(_dataset_line(val_ds_node, "val_transform", "val_dataset"))
        else:
            main.append("val_dataset = ...  # no dataset found in val chain")

        main.append("")
        main.append(_loader_line(train_loader_node, "train_dataset", "train_loader", "True"))
        main.append(_loader_line(val_loader_node,   "val_dataset",   "val_loader",   "False"))

    # ── Mode B: single chain (one DataLoader, optional split) ─────────────
    else:
        chain = (_chain_for_loader(train_loader_node)
                 if train_loader_node else ordered)
        ds_node = next((n for n in chain if n.block_label in _DATASET_BLOCKS), None)

        preamble += _transform_lines(chain, "transform")

        if ds_node:
            main.append(_dataset_line(ds_node, "transform", "dataset"))
        else:
            main.append("dataset = ...  # configure your dataset")

        split_nodes = [n for n in nodes if n.block_label == "RandomSplit"]
        if split_nodes:
            lengths = _p(split_nodes[0], "lengths", "0.8, 0.2")
            main.append(f"train_dataset, val_dataset = random_split(dataset, [{lengths}])")
        else:
            main.append("train_dataset = dataset")
            main.append("val_dataset   = None  # add a val DataLoader node for a proper split")

        main.append("")
        if train_loader_node:
            main.append(_loader_line(train_loader_node, "train_dataset", "train_loader", "True"))
        else:
            main.append("train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)")

        main.append(
            "val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) "
            "if val_dataset else None"
        )

    return "\n".join(main), preamble


def _gen_training(tab: dict) -> tuple[str, str]:
    """
    Generate the loss function and optimizer instantiation lines.

    Reads the first Loss node and the first Optimizer node found in the
    Training tab's graph, fills their constructor arguments from node param
    values, and returns ready-to-use source code lines.

    If no Loss node is found, a ``nn.CrossEntropyLoss`` default is emitted.
    If no Optimizer node is found, ``optim.Adam`` with ``lr=1e-3`` is emitted.
    The loss criterion is moved to ``device`` inline.

    Parameters
    ----------
    tab : dict
        Tab dict with ``role == "training"`` from ``state.tabs``.

    Returns
    -------
    tuple[str, str]
        A two-element tuple:

        * **loss_line** — Single source code line assigning the loss criterion
          to ``criterion``, e.g.
          ``"criterion = nn.CrossEntropyLoss().to(device)"``.
        * **optim_line** — Single source code line assigning the optimizer to
          ``optimizer``, e.g.
          ``"optimizer = optim.Adam(model.parameters(), lr=1e-3)"``.
    """
    graph = build_graph(tab)
    nodes = list(graph.values())

    loss_node = next((n for n in nodes if n.block_label in _LOSS_MAP), None)
    if loss_node:
        module, template = _LOSS_MAP[loss_node.block_label]
        args = _fill(template, loss_node)
        loss_line = f"criterion = {module}({args}).to(device)"
    else:
        loss_line = "criterion = nn.CrossEntropyLoss().to(device)"

    optim_node = next((n for n in nodes if n.block_label in _OPTIM_MAP), None)
    if optim_node:
        module, template = _OPTIM_MAP[optim_node.block_label]
        args = _fill(template, optim_node)
        optim_line = f"optimizer = {module}(model.parameters(), {args})"
    else:
        optim_line = "optimizer = optim.Adam(model.parameters(), lr=1e-3)"

    return loss_line, optim_line


# ---------------------------------------------------------------------------
# Static script template fragments
# ---------------------------------------------------------------------------
# These string constants are assembled verbatim into the generated script.
# They are intentionally kept as plain strings (not f-strings) so that curly
# braces in f-string expressions inside _MAIN_LOOP are escaped as ``{{}}``
# and are not evaluated at definition time.

_HEADER = '''\
"""
train.py - MLForge auto-generated training script.
Edit freely. All hyperparameters are in this file.
"""
'''

_IMPORTS = """\
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
"""

_DEVICE = """\
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
"""

_TRAIN_LOOP = """\
def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.inference_mode()
def evaluate(model, loader, criterion):
    if loader is None:
        return None
    model.eval()
    total_loss = 0.0
    correct = 0
    total   = 0
    for X, y in loader:
        X, y   = X.to(device), y.to(device)
        logits = model(X)
        loss   = criterion(logits, y)
        total_loss += loss.item()
        preds   = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    acc = correct / total if total > 0 else 0.0
    return total_loss / len(loader), acc
"""

_MAIN_LOOP = """\
if __name__ == "__main__":
    EPOCHS = 20

    model     = Model().to(device)
    print(model)

    # Data
{data_block}

    # Training setup
{loss_line}
{optim_line}

    # Training loop
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, criterion, optimizer)
        val_result = evaluate(model, val_loader, criterion)

        if val_result:
            val_loss, val_acc = val_result
            print(f"Epoch {{epoch+1:>3}}/{{EPOCHS}}  "
                  f"train_loss={{train_loss:.4f}}  "
                  f"val_loss={{val_loss:.4f}}  "
                  f"val_acc={{val_acc:.4f}}")
        else:
            print(f"Epoch {{epoch+1:>3}}/{{EPOCHS}}  train_loss={{train_loss:.4f}}")

    torch.save(model.state_dict(), "model.pth")
    print("Saved model.pth")
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_pytorch() -> str:
    """
    Assemble and return the complete ``train.py`` script as a string.

    Orchestrates the three section generators and stitches their output
    together with the static template fragments into a single coherent script.
    The assembly order is:

    1. :data:`_HEADER` — module docstring.
    2. :data:`_IMPORTS` — standard import block.
    3. :data:`_DEVICE` — CUDA / CPU device detection.
    4. Transform definitions (preamble returned by :func:`_gen_data`).
    5. ``Model`` class (from :func:`_gen_model`).
    6. ``train`` / ``evaluate`` helper functions (:data:`_TRAIN_LOOP`).
    7. ``if __name__ == "__main__":`` block (:data:`_MAIN_LOOP`) embedding
       the data pipeline, loss, and optimizer lines.

    Gracefully degrades when a tab is missing: each section falls back to a
    minimal placeholder that keeps the script syntactically valid.

    Returns
    -------
    str
        The complete, formatted Python source code for ``train.py``.

    Examples
    --------
    >>> code = generate_pytorch()
    >>> print(code[:200])
    >>> with open("train.py", "w") as f:
    ...     f.write(code)

    See Also
    --------
    export_pytorch : Save the generated script to a user-chosen file via a
                     DearPyGui file dialog.
    """
    model_tab    = get_tab_by_role("model")
    data_tab     = get_tab_by_role("data_prep")
    training_tab = get_tab_by_role("training")

    # ── Model class ───────────────────────────────────────────────────────
    model_code = _gen_model(model_tab) if model_tab else (
        "class Model(nn.Module):\n"
        "    def __init__(self): super().__init__()\n"
        "    def forward(self, x): return x"
    )

    # ── Data pipeline ─────────────────────────────────────────────────────
    if data_tab:
        data_code, aug_lines = _gen_data(data_tab)
    else:
        data_code = "train_loader = ...  # configure your data pipeline"
        aug_lines = ["transform = transforms.ToTensor()"]

    # Indent data_code for placement inside the if __name__ == "__main__" block
    data_indented = textwrap.indent(data_code, "    ")

    # aug_lines is placed at module level, before the Model class
    aug_block = "\n".join(aug_lines)

    # Section header adapts to whether dual (train + val) transforms were emitted
    has_dual = any("train_transform" in l for l in aug_lines)
    transforms_header = ("# Transforms (train + val)"
                         if has_dual else
                         "# Transforms")

    # ── Training setup ────────────────────────────────────────────────────
    if training_tab:
        loss_line, optim_line = _gen_training(training_tab)
    else:
        loss_line  = "criterion = nn.CrossEntropyLoss().to(device)"
        optim_line = "optimizer = optim.Adam(model.parameters(), lr=1e-3)"

    parts = [
        _HEADER,
        _IMPORTS,
        "\n",
        _DEVICE,
        "\n",
        transforms_header,
        aug_block,
        "\n",
        "# Model",
        model_code,
        "\n",
        "# Train / eval functions ",
        _TRAIN_LOOP,
        _MAIN_LOOP.format(
            data_block=data_indented,
            loss_line=f"    {loss_line}",
            optim_line=f"    {optim_line}",
        ),
    ]

    return "\n".join(p for p in parts if p is not None)


# ---------------------------------------------------------------------------
# Export entry point
# ---------------------------------------------------------------------------

def export_pytorch() -> None:
    """
    Open a DearPyGui save-file dialog and write the generated script to disk.

    Presents a modal file dialog filtered to ``*.py`` files.  On confirmation
    the script produced by :func:`generate_pytorch` is written to the chosen
    path (a ``.py`` extension is appended automatically if omitted).  Success
    and failure are both reported to the in-app console via
    ``ml_D_D.ui.console.log``.

    The dialog is registered under the DearPyGui tag
    ``"export_pytorch_dialog"``; any pre-existing dialog with that tag is
    deleted before a new one is created to prevent duplicate windows.

    This function is wired to *File > Export > Python > PyTorch* in the
    application menu.

    Raises
    ------
    This function does not raise directly; all exceptions during file I/O are
    caught internally and forwarded to the console log as error messages.

    Notes
    -----
    The DearPyGui ``file_dialog`` is non-blocking; the actual write happens
    in the ``_on_save`` callback once the user confirms the dialog.
    """
    import dearpygui.dearpygui as dpg
    from ml_D_D.ui.console import log

    def _on_save(sender, app_data):
        """
        Callback invoked when the user confirms the save-file dialog.

        Retrieves the chosen file path, ensures it ends with ``.py``,
        generates the script, and writes it.  Cleans up the dialog widget
        whether the write succeeds or fails.

        Parameters
        ----------
        sender : int | str
            DearPyGui item tag of the widget that triggered the callback
            (the file dialog).  Not used directly.
        app_data : dict
            DearPyGui callback data dict.  The key ``"file_path_name"``
            contains the absolute path chosen by the user.
        """
        path = app_data.get("file_path_name", "")
        if not path:
            return
        if not path.endswith(".py"):
            path += ".py"
        try:
            code = generate_pytorch()
            with open(path, "w", encoding="utf-8") as f:
                f.write(code)
            log(f"Exported PyTorch script → {path}", "success")
        except Exception as e:
            log(f"Export failed: {e}", "error")
        if dpg.does_item_exist("export_pytorch_dialog"):
            dpg.delete_item("export_pytorch_dialog")

    def _on_cancel(sender, app_data):
        """
        Callback invoked when the user dismisses the save-file dialog.

        Deletes the dialog widget to free DearPyGui resources.

        Parameters
        ----------
        sender : int | str
            DearPyGui item tag of the widget that triggered the callback.
        app_data : dict
            DearPyGui callback data dict (unused for cancel).
        """
        if dpg.does_item_exist("export_pytorch_dialog"):
            dpg.delete_item("export_pytorch_dialog")

    # Guard against duplicate dialogs
    if dpg.does_item_exist("export_pytorch_dialog"):
        dpg.delete_item("export_pytorch_dialog")

    with dpg.file_dialog(
        label="Export PyTorch Script",
        tag="export_pytorch_dialog",
        callback=_on_save,
        cancel_callback=_on_cancel,
        width=700,
        height=450,
        default_filename="train",
        modal=True,
    ):
        dpg.add_file_extension(".py", color=(100, 220, 100))
        dpg.add_file_extension(".*")