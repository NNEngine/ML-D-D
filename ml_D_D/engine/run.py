"""
run.py
Real PyTorch training execution in a background thread.

The thread communicates with the UI via a queue:
  _result_queue  — epoch results posted by the thread, drained each frame

PyTorch is required. The caller (training.py) is responsible for checking
that torch is installed before calling start_real_training().

Public API:
    start_training(cfg)   start the background thread
    stop_training()       signal the thread to stop
    pause_training()      toggle pause
    drain_result_queue()       called each frame; updates UI with results
"""

from __future__ import annotations

import pathlib
import queue
import threading
import time
import torch
from torch.utils.data import DataLoader

import dearpygui.dearpygui as dpg

import ml_D_D.state as state
from ml_D_D.ui.console import log
from ml_D_D.engine.graph import build_graph, topological_sort, get_tab_by_role


# ---------------------------------------------------------------------------
# Resolving Device
# ---------------------------------------------------------------------------

def _resolve_device(setting: str) -> "torch.device":
    """
    Resolve a device setting string to a concrete ``torch.device``.

    Supports automatic device detection (``"auto"``) as well as any explicit
    device string accepted by PyTorch (e.g. ``"cpu"``, ``"cuda"``,
    ``"cuda:1"``, ``"mps"``).

    Resolution order for ``"auto"``:

    1. CUDA — if ``torch.cuda.is_available()`` returns ``True``.
    2. Apple MPS — if ``torch.backends.mps.is_available()`` returns ``True``.
    3. CPU — unconditional fallback.

    Args:
        setting (str): Device preference.  Either ``"auto"`` for automatic
            selection or an explicit PyTorch device string.

    Returns:
        torch.device: The resolved device object ready to be passed to
            ``.to(device)`` calls.

    Example:
        >>> device = _resolve_device("auto")   # picks best available
        >>> device = _resolve_device("cuda:0") # explicit GPU
        >>> device = _resolve_device("cpu")    # force CPU
    """
    import torch
    if setting == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(setting)


# ---------------------------------------------------------------------------
# Model builder, instantiates from graph
# ---------------------------------------------------------------------------

def _build_torch_model(device: "torch.device") -> "torch.nn.Module":
    """
    Construct a live ``nn.Sequential`` model by walking the Model-tab node graph.

    The function performs a topological sort over the nodes in the tab tagged
    with the ``"model"`` role, filters out terminal ``Input`` / ``Output``
    nodes, then dynamically instantiates each remaining layer node via
    ``eval()`` using the parameter templates stored in ``_LAYER_MAP``.

    Node parameter validation:

    * Any layer whose template still contains ``"..."`` (i.e. a required field
      the user has not filled in) raises a ``ValueError`` with a descriptive
      message rather than silently producing a broken model.
    * Unsupported layer labels are skipped with a ``"warning"`` log message so
      partially-built graphs still train.

    Args:
        device (torch.device): The target device; the fully constructed model
            is moved to this device before being returned.

    Returns:
        torch.nn.Module: A ``nn.Sequential`` model containing all valid layer
            nodes in topological order, already residing on ``device``.

    Raises:
        ValueError: If no ``"model"`` tab exists, if the graph contains a
            cycle, if the graph has no layer nodes after filtering Input/Output,
            or if a required parameter field has not been filled in.

    Example:
        >>> model = _build_torch_model(torch.device("cuda"))
        >>> print(sum(p.numel() for p in model.parameters()), "params")
    """
    import torch.nn as nn
    from ml_D_D.engine.generator import _LAYER_MAP, _p, _fill

    tab = get_tab_by_role("model")
    if tab is None:
        raise ValueError("No Model tab found.")

    try:
        nodes = topological_sort(tab)
    except Exception as e:
        raise ValueError(f"Model graph error: {e}")

    layer_nodes = [n for n in nodes if n.block_label not in ("Input", "Output")]
    if not layer_nodes:
        raise ValueError("Model tab has no layer nodes.")

    layers = []
    for node in layer_nodes:
        label = node.block_label
        if label not in _LAYER_MAP:
            log(f"Skipping unsupported layer: {label}", "warning")
            continue
        module_name, template = _LAYER_MAP[label]
        args_str = _fill(template, node)

        # Check for unfilled required params
        if "..." in args_str:
            raise ValueError(
                f"Node '{label}' has unfilled parameters. "
                f"Fill all required fields before training."
            )

        # Dynamically evaluate the module expression
        import torch.nn as nn  # noqa: F811
        try:
            module = eval(f"nn.{module_name.replace('nn.', '')}({args_str})")
            layers.append(module)
        except Exception as e:
            raise ValueError(f"Could not instantiate {label}({args_str}): {e}")

    model = nn.Sequential(*layers).to(device)
    return model


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def _build_dataloaders(
    device: "torch.device",
    val_split: float,
    seed: int,
    shuffle: bool,
) -> "tuple[DataLoader, DataLoader | None]":
    """
    Build and return ``(train_loader, val_loader)`` from the Data Prep graph.

    The function inspects the node graph in the tab tagged ``"data_prep"``
    and supports two construction modes:

    **Mode A — Dual chain:**
        The user has placed two ``Dataset`` nodes (one with ``train=True``,
        one with ``train=False``) and wired each through its own transform
        chain into separate ``DataLoader (train)`` and ``DataLoader (val)``
        nodes.  Both chains are resolved independently and each loader is
        constructed with its own batch size, num_workers, and pin_memory
        settings.

    **Mode B — Single chain with val_split fallback:**
        Only a single ``Dataset`` node exists, wired into a
        ``DataLoader (train)`` node.  The dataset is split into train and
        validation subsets using ``torch.utils.data.random_split`` according
        to ``val_split``.  A ``val_loader`` is only created when
        ``val_split > 0``.

    Transform construction
        Augmentation nodes in the chain (e.g. ``ToTensor``, ``Resize``,
        ``Normalize``, ``ColorJitter``) are converted to
        ``torchvision.transforms`` objects.  If no augmentation nodes are
        present, ``transforms.ToTensor()`` is used as a minimal default.

    Supported dataset node labels
        ``MNIST``, ``CIFAR10``, ``CIFAR100``, ``FashionMNIST``,
        ``ImageFolder``.

    Args:
        device (torch.device): Used to enable ``pin_memory`` only when the
            device type is ``"cuda"``.
        val_split (float): Fraction of the dataset to reserve for validation
            in Mode B.  Ignored in Mode A (dual-chain).  ``0`` disables the
            validation loader.
        seed (int): Random seed used by ``random_split`` to ensure
            reproducible train/val splits in Mode B.
        shuffle (bool): Whether to shuffle the training loader.  The
            validation loader is never shuffled.

    Returns:
        tuple[DataLoader, DataLoader | None]:
            * ``train_loader`` — always present.
            * ``val_loader`` — present when a ``DataLoader (val)`` node
              exists (Mode A) or ``val_split > 0`` (Mode B); ``None``
              otherwise.

    Raises:
        ValueError: If no ``"data_prep"`` tab is found, if the graph
            contains a cycle, if a required ``DataLoader (train)`` chain
            has no ``Dataset`` node, or if an unsupported dataset label is
            encountered.

    Note:
        Progress and transform info are posted to ``_result_queue`` as
        ``"log"`` messages so the UI console reflects what is happening
        during data preparation.

    Example:
        >>> train_loader, val_loader = _build_dataloaders(
        ...     device=torch.device("cpu"),
        ...     val_split=0.1,
        ...     seed=42,
        ...     shuffle=True,
        ... )
        >>> print(len(train_loader.dataset), "training samples")
    """
    from torch.utils.data import random_split
    from torchvision import datasets, transforms
    from torchvision.datasets import ImageFolder
    from ml_D_D.engine.graph import (topological_sort, get_tab_by_role,
                               _DATASET_BLOCKS, _AUG_BLOCKS, build_graph)

    tab = get_tab_by_role("data_prep")
    if tab is None:
        raise ValueError("No Data Prep tab found.")

    try:
        ordered = topological_sort(tab)
    except Exception as e:
        raise ValueError(f"Data Prep graph error: {e}")

    graph  = build_graph(tab)
    nodes  = list(graph.values())

    TORCHVISION_DATASETS = {
        "MNIST":        datasets.MNIST,
        "CIFAR10":      datasets.CIFAR10,
        "CIFAR100":     datasets.CIFAR100,
        "FashionMNIST": datasets.FashionMNIST,
    }

    # Detect mode
    train_loader_node = next(
        (n for n in nodes if n.block_label in ("DataLoader (train)", "DataLoader")), None)
    val_loader_node   = next(
        (n for n in nodes if n.block_label == "DataLoader (val)"), None)

    dual_chain = val_loader_node is not None

    def _build_transform(chain_nodes: list) -> "transforms.Compose":
        """
        Convert a list of augmentation nodes into a ``transforms.Compose`` pipeline.

        Iterates over ``chain_nodes`` in order, recognising known augmentation
        node labels (``ToTensor``, ``Resize``, ``Normalize``, etc.) and
        building the corresponding ``torchvision.transforms`` objects with
        parameter values read directly from each node's ``params`` dict.

        Parameter parsing is lenient: missing or empty string values fall
        back to sensible defaults so that partially-configured nodes do not
        crash the build.  ``Normalize`` additionally supports list/tuple
        syntax for per-channel mean and std (e.g. ``"[0.485, 0.456, 0.406]"``).

        Args:
            chain_nodes (list): Ordered sequence of graph nodes that are
                ancestors of the target loader node.  Non-augmentation nodes
                are silently ignored.

        Returns:
            transforms.Compose: A composed transform pipeline.  Falls back to
                ``[transforms.ToTensor()]`` if no recognised augmentation
                nodes are found.

        Note:
            ``GaussianBlur`` kernel sizes are forced to odd values (incremented
            by 1 if even) to satisfy PyTorch's requirement.
        """
        tlist = []
        for n in chain_nodes:
            label = n.block_label
            if label not in _AUG_BLOCKS:
                continue
            if label == "ToTensor":
                tlist.append(transforms.ToTensor())
            elif label == "Resize":
                tlist.append(transforms.Resize(int(n.params.get("size","224") or "224")))
            elif label == "CenterCrop":
                tlist.append(transforms.CenterCrop(int(n.params.get("size","224") or "224")))
            elif label == "RandomCrop":
                sz  = int(n.params.get("size","32") or "32")
                pad = int(n.params.get("padding","0") or "0")
                tlist.append(transforms.RandomCrop(sz, padding=pad))
            elif label == "RandomHFlip":
                tlist.append(transforms.RandomHorizontalFlip(float(n.params.get("p","0.5") or "0.5")))
            elif label == "RandomVFlip":
                tlist.append(transforms.RandomVerticalFlip(float(n.params.get("p","0.5") or "0.5")))
            elif label == "RandomRotation":
                tlist.append(transforms.RandomRotation(float(n.params.get("degrees","15") or "15")))
            elif label == "Normalize":
                mean = n.params.get("mean","0.5").strip() or "0.5"
                std  = n.params.get("std", "0.5").strip() or "0.5"
                try:
                    mean = eval(mean) if "[" in mean or "(" in mean else [float(mean)]*3
                    std  = eval(std)  if "[" in std  or "(" in std  else [float(std)] *3
                except Exception:
                    mean, std = [0.5,0.5,0.5], [0.5,0.5,0.5]
                tlist.append(transforms.Normalize(mean=mean, std=std))
            elif label == "ColorJitter":
                tlist.append(transforms.ColorJitter(
                    float(n.params.get("brightness","0") or "0"),
                    float(n.params.get("contrast",  "0") or "0"),
                    float(n.params.get("saturation","0") or "0"),
                    float(n.params.get("hue",       "0") or "0"),
                ))
            elif label == "GaussianBlur":
                ks = int(n.params.get("kernel_size","3") or "3")
                if ks % 2 == 0: ks += 1
                tlist.append(transforms.GaussianBlur(ks, sigma=float(n.params.get("sigma","1.0") or "1.0")))
            elif label == "RandomErasing":
                tlist.append(transforms.RandomErasing(p=float(n.params.get("p","0.5") or "0.5")))
            elif label == "Grayscale":
                tlist.append(transforms.Grayscale(int(n.params.get("num_output_channels","1") or "1")))
        if not tlist:
            tlist = [transforms.ToTensor()]
        return transforms.Compose(tlist)

    def _make_dataset(ds_node, transform: "transforms.Compose"):
        """
        Instantiate a torchvision dataset from a graph node and a transform pipeline.

        Reads the ``root``, ``download``, and ``train`` parameters from the
        node's ``params`` dict, then delegates to the appropriate
        ``torchvision.datasets`` class or ``ImageFolder``.

        Supported dataset node labels:
            ``MNIST``, ``CIFAR10``, ``CIFAR100``, ``FashionMNIST``,
            ``ImageFolder``.

        Args:
            ds_node: A graph node whose ``block_label`` identifies the
                dataset type.  The node's ``params`` dict is expected to
                contain (at minimum) a ``"root"`` key.
            transform (transforms.Compose): The transform pipeline to attach
                to the constructed dataset.

        Returns:
            torch.utils.data.Dataset: The instantiated dataset object.

        Raises:
            ValueError: If the node label is not in the supported set, or
                if the ``root`` path for an ``ImageFolder`` dataset does not
                exist on disk.

        Note:
            For torchvision built-in datasets, ``download=True`` is the
            default so that first-run setups work without manual intervention.
            Set the ``download`` param to ``"False"`` on the node to disable
            this behaviour.
        """
        label    = ds_node.block_label
        root     = ds_node.params.get("root","./data").strip() or "./data"
        download = ds_node.params.get("download","True").strip()
        train_f  = ds_node.params.get("train","True").strip()
        is_train = train_f.lower() != "false"
        if label in TORCHVISION_DATASETS:
            return TORCHVISION_DATASETS[label](
                root=root, train=is_train,
                download=(download.lower() != "false"),
                transform=transform,
            )
        elif label == "ImageFolder":
            if not pathlib.Path(root).exists():
                raise ValueError(f"ImageFolder root '{root}' does not exist.")
            return ImageFolder(root=root, transform=transform)
        else:
            raise ValueError(
                f"'{label}' is not yet supported for training in this version.\n"
                f"Supported datasets: MNIST, CIFAR10, CIFAR100, FashionMNIST, ImageFolder."
            )

    def _chain_for_loader(loader_node) -> list:
        """
        Return the ordered list of ancestor nodes that feed into ``loader_node``.

        Performs a backward BFS over the tab's link registry, starting from
        ``loader_node`` and expanding to any node whose output attribute is
        connected (directly or transitively) to the loader.  The returned
        list preserves the global topological order computed earlier so that
        downstream consumers (transform builders, dataset finders) can
        iterate in dependency order.

        Link endpoints are stored as either integer DearPyGui tags or string
        aliases; both forms are normalised to the ``"node_<type>_<id>"``
        alias format before comparison.

        Args:
            loader_node: The graph node representing either
                ``DataLoader (train)`` or ``DataLoader (val)``.

        Returns:
            list: A subset of ``ordered`` containing only the nodes that are
                ancestors of ``loader_node``, in topological order.
        """
        loader_ntag = loader_node.ntag
        ancestors   = set()
        # Walk backwards via links — any node whose output connects to loader
        # or to another ancestor is an ancestor
        changed = True
        targets = {loader_ntag}
        while changed:
            changed = False
            for link_tag, (a1, a2) in tab["links"].items():
                import dearpygui.dearpygui as dpg
                if isinstance(a1, int): a1 = dpg.get_item_alias(a1) or str(a1)
                if isinstance(a2, int): a2 = dpg.get_item_alias(a2) or str(a2)
                dst_parts = a2.split("_")
                src_parts = a1.split("_")
                if len(dst_parts) >= 3 and len(src_parts) >= 3:
                    dst_ntag = f"node_{dst_parts[1]}_{dst_parts[2]}"
                    src_ntag = f"node_{src_parts[1]}_{src_parts[2]}"
                    if dst_ntag in targets and src_ntag not in ancestors:
                        ancestors.add(src_ntag)
                        targets.add(src_ntag)
                        changed = True
        return [n for n in ordered if n.ntag in ancestors]

    # Mode A: dual chain
    if dual_chain:
        train_chain = _chain_for_loader(train_loader_node)
        val_chain   = _chain_for_loader(val_loader_node)

        train_ds_node = next((n for n in train_chain if n.block_label in _DATASET_BLOCKS), None)
        val_ds_node   = next((n for n in val_chain   if n.block_label in _DATASET_BLOCKS), None)

        if train_ds_node is None:
            raise ValueError("DataLoader (train) chain has no Dataset node.")
        if val_ds_node is None:
            raise ValueError("DataLoader (val) chain has no Dataset node.")

        train_transform = _build_transform(train_chain)
        val_transform   = _build_transform(val_chain)

        _result_queue.put({"type":"log","level":"info",
            "msg": f"Train transforms: {[type(t).__name__ for t in train_transform.transforms]}"})
        _result_queue.put({"type":"log","level":"info",
            "msg": f"Val transforms:   {[type(t).__name__ for t in val_transform.transforms]}"})

        train_ds = _make_dataset(train_ds_node, train_transform)
        val_ds   = _make_dataset(val_ds_node,   val_transform)

        def _loader_params(ln) -> "tuple[int, int, bool]":
            """
            Extract ``DataLoader`` construction parameters from a loader node.

            Reads ``batch_size``, ``num_workers``, and ``pin_memory`` from
            the node's ``params`` dict, applying safe defaults when values
            are absent or empty.

            Args:
                ln: A graph node representing a ``DataLoader`` block.

            Returns:
                tuple[int, int, bool]:
                    * ``batch_size`` (int) — number of samples per batch
                      (default: 32).
                    * ``num_workers`` (int) — number of subprocesses for data
                      loading (default: 2).
                    * ``pin_memory`` (bool) — whether to use pinned (page-locked)
                      memory for faster host-to-GPU transfers (default: ``True``).
            """
            bs = int(ln.params.get("batch_size", "32") or "32")
            nw = int(ln.params.get("num_workers","2")  or "2")
            pm = ln.params.get("pin_memory","True").lower() != "false"
            return bs, nw, pm

        tbs, tnw, tpm = _loader_params(train_loader_node)
        vbs, vnw, vpm = _loader_params(val_loader_node)

        train_loader = DataLoader(train_ds, batch_size=tbs, shuffle=shuffle,
                                  num_workers=tnw, pin_memory=tpm and device.type=="cuda")
        val_loader   = DataLoader(val_ds,   batch_size=vbs, shuffle=False,
                                  num_workers=vnw, pin_memory=vpm and device.type=="cuda")
        return train_loader, val_loader

    # Mode B: single chain with val_split fallback
    train_chain = _chain_for_loader(train_loader_node) if train_loader_node else ordered
    ds_node     = next((n for n in train_chain if n.block_label in _DATASET_BLOCKS), None)
    if ds_node is None:
        raise ValueError("No Dataset node found in Data Prep tab.")

    transform = _build_transform(train_chain)
    _result_queue.put({"type":"log","level":"info",
        "msg": f"Transforms: {[type(t).__name__ for t in transform.transforms]}"})

    dataset = _make_dataset(ds_node, transform)

    generator = torch.Generator().manual_seed(seed)
    if val_split > 0:
        n_val   = max(1, int(len(dataset) * val_split))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    else:
        train_ds, val_ds = dataset, None

    loader_node = train_loader_node
    bs = int(loader_node.params.get("batch_size", "32") or "32") if loader_node else 32
    nw = int(loader_node.params.get("num_workers","2")  or "2")  if loader_node else 2
    pm = (loader_node.params.get("pin_memory","True").lower() != "false") if loader_node else True

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=shuffle,
                              num_workers=nw, pin_memory=pm and device.type=="cuda")
    val_loader   = (DataLoader(val_ds, batch_size=bs, shuffle=False,
                               num_workers=nw, pin_memory=pm and device.type=="cuda")
                    if val_ds else None)

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training setup builder
# ---------------------------------------------------------------------------

def _build_criterion_and_optimizer(
    model: "torch.nn.Module",
    device: "torch.device",
) -> "tuple[torch.nn.Module, torch.optim.Optimizer]":
    """
    Construct the loss criterion and parameter optimizer from the Training-tab graph.

    The function inspects all nodes in the tab tagged with the ``"training"``
    role and matches them against the ``_LOSS_MAP`` and ``_OPTIM_MAP``
    registries.  Parameter templates are filled via ``_fill()`` and
    instantiated dynamically using ``eval()``.

    Fallback behaviour
        If no loss or optimizer node is found in the graph (or if the
        ``eval()`` instantiation fails), the function silently falls back to:

        * Loss: ``nn.CrossEntropyLoss``
        * Optimizer: ``optim.Adam`` with ``lr=1e-3``

    This ensures the training thread can always proceed, at the cost of
    possibly ignoring user configuration on malformed graphs.

    Args:
        model (torch.nn.Module): The model whose parameters will be passed
            to the optimizer's ``model.parameters()`` call.
        device (torch.device): Target device; the criterion is moved to this
            device via ``.to(device)``.

    Returns:
        tuple[torch.nn.Module, torch.optim.Optimizer]:
            * ``criterion`` — the instantiated loss function, already on
              ``device``.
            * ``optimizer`` — the instantiated optimizer, bound to
              ``model.parameters()``.

    Raises:
        ValueError: If no ``"training"`` tab exists.

    Example:
        >>> criterion, optimizer = _build_criterion_and_optimizer(model, device)
        >>> loss = criterion(logits, targets)
        >>> optimizer.zero_grad(); loss.backward(); optimizer.step()
    """
    import torch.nn as nn
    import torch.optim as optim
    from ml_D_D.engine.generator import _LOSS_MAP, _OPTIM_MAP, _fill

    tab = get_tab_by_role("training")
    if tab is None:
        raise ValueError("No Training tab found.")

    graph = build_graph(tab)
    nodes = list(graph.values())

    loss_node = next((n for n in nodes if n.block_label in _LOSS_MAP), None)
    if loss_node:
        module_name, template = _LOSS_MAP[loss_node.block_label]
        args_str = _fill(template, loss_node).replace("...", "")
        try:
            criterion = eval(f"nn.{module_name.replace('nn.', '')}({args_str})").to(device)
        except Exception:
            criterion = nn.CrossEntropyLoss().to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    optim_node = next((n for n in nodes if n.block_label in _OPTIM_MAP), None)
    if optim_node:
        module_name, template = _OPTIM_MAP[optim_node.block_label]
        args_str = _fill(template, optim_node).replace("...", "")
        try:
            optimizer = eval(
                f"optim.{module_name.replace('optim.', '')}(model.parameters(), {args_str})"
            )
        except Exception:
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

    return criterion, optimizer


# ---------------------------------------------------------------------------
# Thread queues and control events
# ---------------------------------------------------------------------------

_result_queue: queue.Queue = queue.Queue()
_stop_event:  threading.Event = threading.Event()
_pause_event: threading.Event = threading.Event()
_train_thread: threading.Thread | None = None


# ---------------------------------------------------------------------------
# Training thread
# ---------------------------------------------------------------------------

def _training_thread(cfg: dict) -> None:
    """
    Background thread entry point: build, train, and report results to the UI.

    This function is intended to run in a ``daemon`` thread created by
    :func:`start_training`.  It owns the entire training lifecycle:

    1. **Device resolution** — calls :func:`_resolve_device`.
    2. **Checkpoint directory validation** — verifies write access early so
       the user sees a clear error before any time is spent building the model.
    3. **Model construction** — calls :func:`_build_torch_model`.
    4. **DataLoader construction** — calls :func:`_build_dataloaders`.
    5. **Criterion + optimizer construction** — calls
       :func:`_build_criterion_and_optimizer`.
    6. **Epoch loop** — iterates for ``cfg["epochs"]`` epochs, respecting
       pause/stop signals after each batch and at the start of each epoch.
    7. **Batch reporting** — every 10 batches a ``"batch"`` result dict is
       posted to ``_result_queue`` with a 10-step exponential moving average
       loss for smooth UI updates.
    8. **Validation** — if a ``val_loader`` is available, runs inference-mode
       evaluation and computes mean loss and top-1 accuracy.
    9. **Checkpointing** — saves ``best.pth`` when the monitored metric
       improves (if ``ckpt_best=True``) or periodic ``epoch_XXXX.pth``
       files (if ``ckpt_best=False`` and the epoch interval matches).
    10. **Early stopping** — if ``es_enable=True`` and ``val_loss`` is
        available, increments a patience counter when the metric does not
        improve by more than ``es_min_delta``; stops training when the
        counter reaches ``es_patience``.
    11. **Completion** — saves ``final.pth`` and posts a ``"done"`` result.

    Thread communication
        All communication back to the main thread goes through
        ``_result_queue`` as plain dicts with a ``"type"`` key:

        * ``"log"``     — a console message with an optional ``"level"``.
        * ``"batch"``   — intra-epoch progress (epoch, batch, smoothed loss).
        * ``"epoch"``   — end-of-epoch metrics (train_loss, val_loss, val_acc).
        * ``"done"``    — training completed successfully.
        * ``"stopped"`` — training was aborted by :func:`stop_training`.
        * ``"error"``   — an exception occurred; ``"msg"`` contains the
          string representation.

    Pause/stop handling
        * **Pause**: the thread spins in a 100 ms sleep loop while
          ``_pause_event`` is set; it breaks immediately if ``_stop_event``
          is also set.
        * **Stop**: checked at the start of every epoch and after every
          batch; posts ``"stopped"`` and returns cleanly.

    Args:
        cfg (dict): Training configuration dictionary produced by the UI.
            Expected keys:

            * ``device`` (str) — device string or ``"auto"``.
            * ``epochs`` (int) — total number of training epochs.
            * ``val_split`` (float) — validation fraction for Mode B splits.
            * ``seed`` (int) — RNG seed for reproducible splits.
            * ``shuffle`` (bool) — whether to shuffle the training loader.
            * ``amp`` (bool) — enable automatic mixed precision (CUDA only).
            * ``ckpt_dir`` (str) — path to the checkpoint output directory.
            * ``ckpt_every`` (int) — epoch interval for periodic checkpoints.
            * ``ckpt_best`` (bool) — save ``best.pth`` on metric improvement.
            * ``ckpt_monitor`` (str) — metric to monitor: ``"val_loss"``,
              ``"val_acc"``, or ``"train_loss"``.
            * ``es_enable`` (bool) — enable early stopping.
            * ``es_patience`` (int) — epochs without improvement before stop.
            * ``es_min_delta`` (float) — minimum improvement threshold.

    Returns:
        None

    Note:
        AMP (``torch.amp.autocast`` + ``GradScaler``) is only activated when
        both ``cfg["amp"]`` is ``True`` **and** the resolved device type is
        ``"cuda"``; it is silently disabled on CPU and MPS devices.
    """
    import torch

    try:
        device       = _resolve_device(cfg["device"])
        epochs       = cfg["epochs"]
        val_split    = cfg["val_split"]
        seed         = cfg["seed"]
        shuffle      = cfg["shuffle"]
        use_amp      = cfg["amp"] and device.type == "cuda"
        ckpt_dir     = pathlib.Path(cfg["ckpt_dir"])
        ckpt_every   = cfg["ckpt_every"]
        ckpt_best    = cfg["ckpt_best"]
        ckpt_monitor = cfg["ckpt_monitor"]
        es_enable    = cfg["es_enable"]
        es_patience  = cfg["es_patience"]
        es_min_delta = cfg["es_min_delta"]

        _result_queue.put({"type": "log", "msg": f"Device: {device}", "level": "info"})

        # Validate checkpoint dir early
        if ckpt_dir:
            try:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                # Verify write permission by touching a temp file
                test_file = ckpt_dir / ".write_test"
                test_file.touch()
                test_file.unlink()
            except PermissionError:
                raise ValueError(
                    f"Cannot write to checkpoint directory: {ckpt_dir}\n"
                    f"Check folder permissions or choose a different Save Dir."
                )
            except Exception as e:
                raise ValueError(f"Checkpoint directory error: {e}")

        # Build
        model = _build_torch_model(device)
        _result_queue.put({"type": "log",
                           "msg": f"Model: {sum(p.numel() for p in model.parameters()):,} params",
                           "level": "info"})

        train_loader, val_loader = _build_dataloaders(device, val_split, seed, shuffle)
        _result_queue.put({"type": "log",
                           "msg": f"Dataset: {len(train_loader.dataset):,} train samples",
                           "level": "info"})

        criterion, optimizer = _build_criterion_and_optimizer(model, device)
        scaler = torch.amp.GradScaler(device.type) if use_amp else None

        best_metric   = float("inf") if "loss" in ckpt_monitor else 0.0
        es_counter    = 0
        start_time    = time.time()

        # Epoch loop
        for epoch in range(1, epochs + 1):

            # Pause check
            while _pause_event.is_set():
                if _stop_event.is_set():
                    break
                time.sleep(0.1)

            if _stop_event.is_set():
                _result_queue.put({"type": "stopped"})
                return

            # Train
            model.train()
            train_loss   = 0.0
            batch_losses = []
            for batch_idx, (X, y) in enumerate(train_loader):
                if _stop_event.is_set():
                    break
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast(device.type, enabled=use_amp):
                    logits = model(X)
                    loss   = criterion(logits, y)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                batch_loss  = loss.item()
                train_loss += batch_loss
                batch_losses.append(batch_loss)

                # Post smoothed batch loss every 10 steps
                if (batch_idx + 1) % 10 == 0:
                    smoothed = sum(batch_losses[-10:]) / len(batch_losses[-10:])
                    _result_queue.put({
                        "type":       "batch",
                        "epoch":      epoch,
                        "batch":      batch_idx + 1,
                        "batch_loss": smoothed,
                        "total_batches": len(train_loader),
                    })

            train_loss /= max(len(train_loader), 1)

            # Validate
            val_loss = val_acc = None
            if val_loader:
                model.eval()
                v_loss = 0.0
                correct = total = 0
                with torch.inference_mode():
                    for X, y in val_loader:
                        X, y   = X.to(device), y.to(device)
                        logits = model(X)
                        v_loss += criterion(logits, y).item()
                        preds   = logits.argmax(dim=1)
                        correct += (preds == y).sum().item()
                        total   += y.size(0)
                val_loss = v_loss / max(len(val_loader), 1)
                val_acc  = correct / total if total > 0 else 0.0

            # Checkpoint
            current_metric = (
                val_loss if ckpt_monitor == "val_loss" and val_loss is not None
                else val_acc if ckpt_monitor == "val_acc" and val_acc is not None
                else train_loss
            )
            is_best = (
                current_metric < best_metric if "loss" in ckpt_monitor
                else current_metric > best_metric
            )
            if is_best:
                best_metric = current_metric
                if ckpt_best:
                    torch.save(model.state_dict(), ckpt_dir / "best.pth")
                    _result_queue.put({"type": "log",
                                      "msg": f"Best checkpoint saved ({ckpt_monitor}={best_metric:.4f})",
                                      "level": "success"})
            if not ckpt_best and epoch % ckpt_every == 0:
                torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:04d}.pth")

            # Post epoch result first, before any early-stop break
            _result_queue.put({
                "type":       "epoch",
                "epoch":      epoch,
                "total":      epochs,
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "val_acc":    val_acc,
            })

            # Early stopping
            if es_enable and val_loss is not None:
                improved = (
                    (best_metric - current_metric) > es_min_delta
                    if "loss" in ckpt_monitor
                    else (current_metric - best_metric) > es_min_delta
                )
                if not improved:
                    es_counter += 1
                    if es_counter >= es_patience:
                        _result_queue.put({"type": "log",
                                          "msg": f"Early stopping at epoch {epoch}.",
                                          "level": "warning"})
                        break
                else:
                    es_counter = 0

        elapsed = time.time() - start_time
        torch.save(model.state_dict(), ckpt_dir / "final.pth")
        _result_queue.put({
            "type":    "done",
            "elapsed": elapsed,
            "msg":     f"Training complete in {elapsed:.1f}s. Saved final.pth",
        })

    except Exception as e:
        _result_queue.put({"type": "error", "msg": str(e)})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_training(cfg: dict) -> None:
    """
    Start the background training thread with the given configuration.

    Clears any lingering stop/pause signals from a previous run, then
    spawns a new daemon thread executing :func:`_training_thread`.  The
    thread is stored in the module-level ``_train_thread`` variable so that
    :func:`stop_training` and :func:`pause_training` can control it.

    The thread is a daemon so it does not prevent the Python process from
    exiting if the main window is closed.

    Args:
        cfg (dict): Training configuration dictionary.  See
            :func:`_training_thread` for a full description of expected keys.

    Returns:
        None

    Note:
        Calling ``start_training`` while a previous thread is still running
        is not prevented here; the caller is responsible for ensuring
        :func:`stop_training` has been called and the previous thread has
        finished before starting a new run.

    Example:
        >>> start_training({
        ...     "device": "auto", "epochs": 20, "val_split": 0.1,
        ...     "seed": 0, "shuffle": True, "amp": False,
        ...     "ckpt_dir": "./checkpoints", "ckpt_every": 5,
        ...     "ckpt_best": True, "ckpt_monitor": "val_loss",
        ...     "es_enable": False, "es_patience": 5, "es_min_delta": 1e-4,
        ... })
    """
    global _train_thread
    _stop_event.clear()
    _pause_event.clear()
    _train_thread = threading.Thread(
        target=_training_thread, args=(cfg,), daemon=True
    )
    _train_thread.start()


def stop_training() -> None:
    """
    Signal the background training thread to stop as soon as possible.

    Sets the module-level ``_stop_event`` so that the training loop will
    exit at the next stop-check point (beginning of each epoch or after each
    batch).  Also clears ``_pause_event`` so a paused thread is no longer
    blocked and can observe the stop signal.

    This function returns immediately; it does **not** block until the thread
    finishes.  If the caller needs to wait for the thread to terminate it
    should call ``_train_thread.join()`` separately.

    Returns:
        None

    Example:
        >>> stop_training()          # send signal
        >>> _train_thread.join()     # optional: block until thread exits
    """
    _stop_event.set()
    _pause_event.clear()


def pause_training() -> None:
    """
    Toggle the pause state of the background training thread.

    If the thread is currently running (pause event is clear), this sets
    ``_pause_event`` which causes the thread to enter a 100 ms sleep loop
    at the next epoch boundary.

    If the thread is already paused (pause event is set), this clears
    ``_pause_event`` which allows the thread to resume from where it left
    off.

    Returns:
        None

    Note:
        Pause is checked only between epochs and between batches (when
        observing the stop event), not mid-batch.  The current batch will
        always complete before the thread pauses.

    Example:
        >>> pause_training()   # pause
        >>> pause_training()   # resume
    """
    if _pause_event.is_set():
        _pause_event.clear()
    else:
        _pause_event.set()


def is_paused() -> bool:
    """
    Return whether the training thread is currently in the paused state.

    Reflects the internal ``_pause_event`` flag; does **not** guarantee that
    the thread has actually reached its pause-check loop yet (it may still
    be finishing the current batch).

    Returns:
        bool: ``True`` if ``_pause_event`` is set (pause requested or active),
            ``False`` otherwise.

    Example:
        >>> if is_paused():
        ...     print("Training is paused")
    """
    return _pause_event.is_set()


# ---------------------------------------------------------------------------
# Result queue drain  (called every frame from render loop)
# ---------------------------------------------------------------------------

def drain_result_queue() -> None:
    """
    Drain all pending messages from the training thread and dispatch them to the UI.

    This function is designed to be called once per render frame from the
    main DearPyGUI event loop.  It empties ``_result_queue`` in a tight
    non-blocking loop (``get_nowait``), passing each message dict to
    :func:`_handle_result` for UI updates.  The loop exits silently when
    the queue is empty (``queue.Empty``).

    Because all DearPyGUI calls must occur on the main thread, the training
    thread never touches the UI directly; instead, it posts structured dicts
    to ``_result_queue`` which this function processes on the main thread.

    Returns:
        None

    Note:
        Must be called from the **main thread** only.  Calling it from the
        training thread or any other background thread is unsafe.

    Example:
        >>> # Inside the DearPyGUI render loop:
        >>> while dpg.is_dearpygui_running():
        ...     drain_result_queue()
        ...     dpg.render_dearpygui_frame()
    """
    from ml_D_D.ui.training import apply_train_btn_style, update_status_indicator

    try:
        while True:
            item = _result_queue.get_nowait()
            _handle_result(item)
    except queue.Empty:
        pass


def _handle_result(item: dict) -> None:
    """
    Dispatch a single result dict from the training thread to the appropriate UI handler.

    This is the central routing function for all inter-thread communication.
    It is called exclusively by :func:`drain_result_queue` on the main thread
    and updates DearPyGUI widgets, state dictionaries, and the console log
    based on the ``"type"`` field of ``item``.

    Handled message types
    ---------------------
    ``"log"``
        Forwards the message to the console via :func:`log`.  The optional
        ``"level"`` key controls the log level (default: ``"info"``).

    ``"batch"``
        Updates the progress bar with fine-grained intra-epoch progress
        (computed from batch index and total batches), overlays a status
        string, appends to the batch-loss plot series, and applies an
        exponential moving average smoothing whose window size is read from
        the ``cfg_batch_smooth`` DearPyGUI widget.

    ``"epoch"``
        Updates the progress bar to the coarse epoch fraction, logs the
        epoch summary line, and appends to the epoch-level train-loss,
        val-loss, and accuracy plot series.  Duplicate epoch messages (same
        epoch number as the last logged) are silently ignored to prevent
        jitter when the queue is drained mid-epoch.

    ``"done"`` / ``"stopped"``
        Resets ``state.train_state`` to idle, clears all in-memory plot data,
        resets the batch-loss series to empty, sets the progress bar to 1.0
        (done) or 0.0 (stopped), logs a final message, resets block labels,
        and refreshes the train button style and status indicator.

    ``"error"``
        Sets state to idle, logs the error message at level ``"error"``, and
        refreshes the train button style and status indicator.

    Args:
        item (dict): A message dict produced by :func:`_training_thread`.
            Must contain a ``"type"`` key.  Additional keys vary by type —
            see the type descriptions above for details.

    Returns:
        None

    Note:
        Plot series updates use ``dpg.does_item_exist`` guards so that
        partial UI setups (e.g. during testing without a full window) do not
        raise errors.
    """
    from ml_D_D.ui.training import apply_train_btn_style, update_status_indicator

    t = item["type"]

    if t == "log":
        log(item["msg"], item.get("level", "info"))

    elif t == "batch":
        # Fine-grained progress within the current epoch
        epoch        = item["epoch"]
        batch        = item["batch"]
        total_b      = item["total_batches"]
        batch_loss   = item["batch_loss"]
        total_epochs = state.train_state.get("total_epochs", 1)

        # Only update progress if we have a valid total_epochs already set
        # (avoids jitter before the first epoch result arrives)
        if total_epochs > 1 or state.train_state.get("epoch", 0) > 0:
            coarse = (epoch - 1) / total_epochs
            fine   = (batch / total_b) / total_epochs
            prog   = min(coarse + fine, epoch / total_epochs)
            dpg.set_value("train_progress", prog)

        if dpg.does_item_exist("train_progress"):
            dpg.configure_item("train_progress",
                               overlay=f"Epoch {epoch}/{total_epochs}  "
                                       f"batch {batch}/{total_b}  "
                                       f"loss={batch_loss:.4f}")

        # Append to batch loss series
        ts = state.train_state
        if "plot_batch_x" not in ts:
            ts["plot_batch_x"] = []
            ts["plot_batch_y"] = []
        global_step = (epoch - 1) * total_b + batch
        ts["plot_batch_x"].append(global_step / total_b)
        ts["plot_batch_y"].append(batch_loss)

        if dpg.does_item_exist("series_batch_loss"):
            # Apply smoothing window from the slider
            window = 10
            if dpg.does_item_exist("cfg_batch_smooth"):
                window = max(1, int(dpg.get_value("cfg_batch_smooth")))
            raw_y = ts["plot_batch_y"]
            if window > 1 and len(raw_y) >= window:
                # Exponential moving average for a smooth line
                alpha = 2.0 / (window + 1)
                smoothed = [raw_y[0]]
                for v in raw_y[1:]:
                    smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
                display_y = smoothed
            else:
                display_y = raw_y
            dpg.set_value("series_batch_loss",
                          [ts["plot_batch_x"], display_y])
            dpg.fit_axis_data("loss_y")

    elif t == "epoch":
        e     = item["epoch"]
        total = item["total"]
        tl    = item["train_loss"]
        vl    = item.get("val_loss")
        va    = item.get("val_acc")

        # Guard against duplicate epoch results (can occur if queue drains mid-epoch)
        last_logged = state.train_state.get("_last_logged_epoch", 0)
        if e <= last_logged:
            return
        state.train_state["_last_logged_epoch"] = e

        # Set total_epochs first so batch handler has it before next epoch starts
        state.train_state["total_epochs"] = total
        state.train_state["epoch"]        = e

        # Clean epoch-level progress
        dpg.set_value("train_progress", e / total)
        if dpg.does_item_exist("train_progress"):
            dpg.configure_item("train_progress",
                               overlay=f"Epoch {e}/{total}")

        if vl is not None and va is not None:
            log(f"Epoch {e:>3}/{total}  loss={tl:.4f}  "
                f"val_loss={vl:.4f}  val_acc={va:.4f}", "info")
        else:
            log(f"Epoch {e:>3}/{total}  loss={tl:.4f}", "info")

        # Update plots
        ts = state.train_state
        if "plot_epochs" not in ts:
            ts["plot_epochs"] = []
            ts["plot_tl"]     = []
            ts["plot_vl"]     = []
            ts["plot_ta"]     = []
            ts["plot_va"]     = []

        ts["plot_epochs"].append(e)
        ts["plot_tl"].append(tl)
        ts["plot_vl"].append(vl if vl is not None else tl)
        ts["plot_ta"].append(va if va is not None else 0.0)
        ts["plot_va"].append(va if va is not None else 0.0)

        if dpg.does_item_exist("series_train_loss"):
            xs = ts["plot_epochs"]
            dpg.set_value("series_train_loss", [xs, ts["plot_tl"]])
            dpg.set_value("series_val_loss",   [xs, ts["plot_vl"]])
            dpg.set_value("series_train_acc",  [xs, ts["plot_ta"]])
            dpg.set_value("series_val_acc",    [xs, ts["plot_va"]])
            dpg.fit_axis_data("loss_y")
            dpg.fit_axis_data("acc_y")

    elif t in ("done", "stopped"):
        state.train_state["status"] = "idle"
        state.train_state["epoch"]  = 0
        if "plot_epochs" in state.train_state:
            for key in ("plot_epochs", "plot_tl", "plot_vl", "plot_ta", "plot_va",
                        "plot_batch_x", "plot_batch_y"):
                state.train_state.pop(key, None)
        if dpg.does_item_exist("series_batch_loss"):
            dpg.set_value("series_batch_loss", [[], []])
        dpg.set_value("train_progress", 1.0 if t == "done" else 0.0)
        if "msg" in item:
            log(item["msg"], "success" if t == "done" else "warning")
        from ml_D_D.engine.training_setup import reset_block_labels
        reset_block_labels()
        apply_train_btn_style()
        update_status_indicator()

    elif t == "error":
        state.train_state["status"] = "idle"
        log(f"Training error: {item['msg']}", "error")
        apply_train_btn_style()
        update_status_indicator()