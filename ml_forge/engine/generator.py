"""
generator.py
PyTorch training script generator.

Takes the live graph state from all three pipeline tabs and emits a
complete, runnable train.py file.

Public:
    generate_pytorch()  -> str   full script as a string
    export_pytorch()             file-dialog save, wired to menu
"""

from __future__ import annotations
import textwrap
from ml_forge.engine.graph import build_graph, topological_sort, get_tab_by_role, GraphNode


#  PyTorch module name mapping
#  block label  ->  (torch_module, arg_template)
#  {param} tokens in arg_template are filled from the node's param values.

_LAYER_MAP: dict[str, tuple[str, str]] = {
    # Layers
    "Linear":          ("nn.Linear",          "{in_features}, {out_features}"),
    "Conv2D":          ("nn.Conv2d",           "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}"),
    "ConvTranspose2D": ("nn.ConvTranspose2d",  "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}"),
    "Flatten":         ("nn.Flatten",          ""),
    # Activations
    "ReLU":            ("nn.ReLU",             ""),
    "Sigmoid":         ("nn.Sigmoid",          ""),
    "Tanh":            ("nn.Tanh",             ""),
    "Softmax":         ("nn.Softmax",          "dim={dim}"),
    "GELU":            ("nn.GELU",             ""),
    "LeakyReLU":       ("nn.LeakyReLU",        "negative_slope={negative_slope}"),
    # Normalisation
    "BatchNorm2D":     ("nn.BatchNorm2d",      "{num_features}"),
    "LayerNorm":       ("nn.LayerNorm",        "{normalized_shape}"),
    "GroupNorm":       ("nn.GroupNorm",        "{num_groups}, {num_channels}"),
    "Dropout":         ("nn.Dropout",          "p={p}"),
    # Pooling
    "MaxPool2D":       ("nn.MaxPool2d",        "kernel_size={kernel_size}, stride={stride}, padding={padding}"),
    "AvgPool2D":       ("nn.AvgPool2d",        "kernel_size={kernel_size}, stride={stride}, padding={padding}"),
    "AdaptiveAvgPool2D": ("nn.AdaptiveAvgPool2d", "{output_size}"),
}

_LOSS_MAP: dict[str, tuple[str, str]] = {
    "CrossEntropyLoss": ("nn.CrossEntropyLoss", ""),
    "MSELoss":          ("nn.MSELoss",          "reduction='{reduction}'"),
    "BCELoss":          ("nn.BCELoss",          "reduction='{reduction}'"),
    "BCEWithLogits":    ("nn.BCEWithLogitsLoss","reduction='{reduction}'"),
    "NLLLoss":          ("nn.NLLLoss",          "reduction='{reduction}'"),
    "HuberLoss":        ("nn.HuberLoss",        "delta={delta}"),
    "KLDivLoss":        ("nn.KLDivLoss",        "reduction='{reduction}'"),
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
    "RandomCrop":     ("transforms.RandomCrop",     "{size}, padding={padding}"),
    "RandomHFlip":    ("transforms.RandomHorizontalFlip", "p={p}"),
    "RandomVFlip":    ("transforms.RandomVerticalFlip",   "p={p}"),
    "ColorJitter":    ("transforms.ColorJitter",    "brightness={brightness}, contrast={contrast}, saturation={saturation}, hue={hue}"),
    "RandomRotation": ("transforms.RandomRotation", "{degrees}"),
    "Normalize":      ("transforms.Normalize",      "mean={mean}, std={std}"),
    "Resize":         ("transforms.Resize",         "{size}"),
    "CenterCrop":     ("transforms.CenterCrop",     "{size}"),
    "ToTensor":       ("transforms.ToTensor",       ""),
    "GaussianBlur":   ("transforms.GaussianBlur",   "{kernel_size}, sigma={sigma}"),
    "RandomErasing":  ("transforms.RandomErasing",  "p={p}"),
}


#  Param defaults  (used when a node field is left empty)

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


def _p(node: GraphNode, key: str, fallback: str = "...") -> str:
    """Return a node's param value, its default, or a fallback placeholder."""
    val = node.params.get(key, "").strip()
    return val or _PARAM_DEFAULTS.get(key, fallback)


def _fill(template: str, node: GraphNode) -> str:
    """
    Fill a template string with a node's param values.
    Empty fields fall back to _PARAM_DEFAULTS before using '...'.
    """
    result = template
    for k, v in node.params.items():
        filled = v.strip() if v else _PARAM_DEFAULTS.get(k, "...")
        result = result.replace("{" + k + "}", filled)
    return result


def _safe_name(block_label: str, idx: int) -> str:
    """Turn 'BatchNorm2D' + 2 into 'batch_norm2d_2'."""
    return block_label.lower().replace(" ", "_") + f"_{idx}"


def _I(n: int = 1) -> str:
    return "    " * n

#  Section generators

def _gen_model(tab: dict) -> str:
    """Generate the nn.Module class from the Model tab."""
    try:
        nodes = topological_sort(tab)
    except Exception:
        nodes = list(build_graph(tab).values())

    # Filter to layer-type nodes only (skip Input/Output)
    layer_nodes = [n for n in nodes if n.block_label not in ("Input", "Output")]

    init_lines: list[str] = []
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

        # LSTM returns tuple — unpack hidden states
        if label == "LSTM":
            forward_lines.append(f"x, _ = self.{attr}(x)")
        else:
            forward_lines.append(f"x = self.{attr}(x)")

    # Get input shape from Input node if present
    input_node = next((n for n in nodes if n.block_label == "Input"), None)
    input_shape = _p(input_node, "shape", "# define input shape") if input_node else "# define input shape"

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
    Generate the data pipeline section.

    Mirrors the dual-chain / single-chain logic in runner.py:
      Mode A (dual chain): separate DataLoader (train) and DataLoader (val),
               each with their own ancestor chain -> separate transforms,
               datasets and loaders.
      Mode B (single chain): one DataLoader (train), optional val_split.

    Returns (main_code_block, preamble_lines_before_class).
    The preamble contains the transform definitions.
    The main block contains the dataset + dataloader instantiation.
    """
    from ml_forge.engine.graph import build_graph, topological_sort, _DATASET_BLOCKS, _AUG_BLOCKS

    try:
        ordered = topological_sort(tab)
    except Exception:
        ordered = list(build_graph(tab).values())

    graph = build_graph(tab)
    nodes = list(graph.values())

    def _chain_for_loader(loader_node):
        """Return topologically-ordered ancestor nodes for a loader."""
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
        """Return lines that define a transforms.Compose(...) for var_name."""
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
        """Return a line instantiating a dataset."""
        label = ds_node.block_label
        if label not in _DATASET_MAP:
            return f"{var_name} = ...  # unsupported dataset: {label}"
        module, template = _DATASET_MAP[label]
        args = _fill(template, ds_node)
        # substitute the correct transform variable name
        args = args.replace("transform=transform", f"transform={transform_var}")
        return f"{var_name} = {module}({args})"

    def _loader_line(loader_node, dataset_var, loader_var, shuffle_default):
        bs = _p(loader_node, "batch_size", "32")
        nw = _p(loader_node, "num_workers", "2")
        pm = _p(loader_node, "pin_memory", "True")
        sh = _p(loader_node, "shuffle", shuffle_default)
        return (f"{loader_var} = DataLoader({dataset_var}, batch_size={bs}, "
                f"shuffle={sh}, num_workers={nw}, pin_memory={pm})")

    # Detect mode
    train_loader_node = next(
        (n for n in nodes if n.block_label in ("DataLoader (train)", "DataLoader")), None)
    val_loader_node = next(
        (n for n in nodes if n.block_label == "DataLoader (val)"), None)

    preamble: list[str] = []
    main:     list[str] = []

    # Mode A: dual chain
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

    # Mode B: single chain
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
    Generate loss and optimizer lines.
    Returns (loss_line, optim_line).
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


# script assembly!

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


def generate_pytorch() -> str:
    """
    Build and return the full training script as a string.
    Reads live graph state from all three pipeline tabs.
    """
    model_tab    = get_tab_by_role("model")
    data_tab     = get_tab_by_role("data_prep")
    training_tab = get_tab_by_role("training")

    # ── Model class ───────────────────────────────────────
    model_code = _gen_model(model_tab) if model_tab else (
        "class Model(nn.Module):\n"
        "    def __init__(self): super().__init__()\n"
        "    def forward(self, x): return x"
    )

    # ── Data pipeline ─────────────────────────────────────
    if data_tab:
        data_code, aug_lines = _gen_data(data_tab)
    else:
        data_code = "train_loader = ...  # configure your data pipeline"
        aug_lines = ["transform = transforms.ToTensor()"]

    # Indent data_code for use inside if __name__ block
    data_indented = textwrap.indent(data_code, "    ")
    # aug_lines is a preamble (transform definitions) placed before the model class
    aug_block = "\n".join(aug_lines)
    # Section header adapts to single vs dual chain
    has_dual = any("train_transform" in l for l in aug_lines)
    transforms_header = ("# Transforms (train + val)"
                         if has_dual else
                         "# Transforms")

    # ── Training setup ────────────────────────────────────
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


#  Export entry point (wired to menu)

def export_pytorch() -> None:
    """
    Open a save-file dialog and write the generated script.
    Wired to File > Export > Python > PyTorch.
    """
    import dearpygui.dearpygui as dpg
    from ml_forge.ui.console import log

    def _on_save(sender, app_data):
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
        if dpg.does_item_exist("export_pytorch_dialog"):
            dpg.delete_item("export_pytorch_dialog")

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