"""
engine/autofill.py

Automatic parameter filling and shape inference system for the node editor.

This module provides:
    - Dataset-driven input/output shape inference
    - Feature/channel propagation across layers
    - Shape simulation through the model graph
    - Automatic parameter filling (e.g., in_features, in_channels)
    - Detection and highlighting of dimension mismatches

It acts as a lightweight static analysis engine for the visual model graph.
"""

from __future__ import annotations
import dearpygui.dearpygui as dpg
import ml_D_D.state as state


# Known shapes per dataset

_DATASET_SHAPES: dict[str, tuple[str, int]] = {
    "MNIST":        ("1, 28, 28",  10),
    "FashionMNIST": ("1, 28, 28",  10),
    "CIFAR10":      ("3, 32, 32",  10),
    "CIFAR100":     ("3, 32, 32", 100),
    "ImageFolder":  ("3, 224, 224", 0),
}

_OUT_PARAM: dict[str, str] = {
    "Linear":          "out_features",
    "Conv2D":          "out_channels",
    "ConvTranspose2D": "out_channels",
}

_IN_PARAM: dict[str, str] = {
    "Linear":          "in_features",
    "Conv2D":          "in_channels",
    "ConvTranspose2D": "in_channels",
}


def _field(ntag: str, param: str) -> str:
    """
    Construct the DearPyGui tag for a node parameter input field.

    Args:
        ntag (str): Node tag identifier.
        param (str): Parameter name.

    Returns:
        str: Fully qualified input field tag.
    """
    parts = ntag.split("_")
    return f"node_{parts[1]}_{parts[2]}_input_{param}"


def _get_field(ntag: str, param: str) -> str:
    """
    Retrieve the current value of a node parameter field.

    Args:
        ntag (str): Node tag identifier.
        param (str): Parameter name.

    Returns:
        str: Parameter value (trimmed), or empty string if not found.
    """
    tag = _field(ntag, param)
    return dpg.get_value(tag).strip() if dpg.does_item_exist(tag) else ""


def _set_field(ntag: str, param: str, value: str) -> None:
    """
    Set a parameter field value only if it is currently empty.

    This preserves user input while allowing automatic inference.

    Args:
        ntag (str): Node tag identifier.
        param (str): Parameter name.
        value (str): Value to assign.
    """
    tag = _field(ntag, param)
    if dpg.does_item_exist(tag):
        current = dpg.get_value(tag).strip()
        if not current:
            dpg.set_value(tag, value)


def _set_field_force(ntag: str, param: str, value: str) -> None:
    """
    Forcefully overwrite a parameter field value.

    Args:
        ntag (str): Node tag identifier.
        param (str): Parameter name.
        value (str): Value to assign.
    """
    tag = _field(ntag, param)
    if dpg.does_item_exist(tag):
        dpg.set_value(tag, value)


def _block_label(tab: dict, ntag: str) -> str:
    """
    Retrieve the display label of a node.

    Args:
        tab (dict): Tab state containing nodes.
        ntag (str): Node tag identifier.

    Returns:
        str: Node label.
    """
    ni = tab["nodes"].get(ntag, {})
    return ni["label"] if isinstance(ni, dict) else ni


def _get_tab_by_role(role: str) -> dict | None:
    """
    Retrieve a tab configuration by its assigned role.

    Args:
        role (str): Role identifier (e.g., "model", "data_prep").

    Returns:
        dict | None: Matching tab dictionary if found, otherwise None.
    """
    for t in state.tabs.values():
        if t.get("role") == role:
            return t
    return None


def _safe_int(v, default=0) -> int:
    """
    Safely convert a value to an integer.

    Args:
        v (Any): Value to convert.
        default (int, optional): Fallback value if conversion fails.

    Returns:
        int: Parsed integer or default value.
    """
    try:
        return int(str(v).strip())
    except Exception:
        return default


def _simulate_shapes(tab: dict) -> dict[str, tuple | None]:
    """
    Simulate forward shape propagation through the model graph.

    Args:
        tab (dict): Model tab state.

    Returns:
        dict[str, tuple | None]: Mapping of node tag → inferred output shape.
    """
    try:
        from ml_D_D.engine.graph import topological_sort
        ordered = topological_sort(tab)
    except Exception:
        return {}

    shapes: dict[str, tuple | None] = {}

    def _input_shape() -> tuple | None:
        """Extract input shape from the Input node."""
        for node in ordered:
            if node.block_label == "Input":
                raw = _get_field(node.ntag, "shape").replace(" ", "")
                parts = [p for p in raw.split(",") if p]
                if len(parts) == 3:
                    return tuple(_safe_int(p) for p in parts)
                if len(parts) == 1 and _safe_int(parts[0]) > 0:
                    return (_safe_int(parts[0]),)
        return None

    def _upstream_shape(ntag: str) -> tuple | None:
        """Retrieve shape from upstream connected node."""
        for _, (a1, a2) in tab["links"].items():
            sp = a1.split("_"); dp = a2.split("_")
            if len(sp) >= 3 and len(dp) >= 3:
                src = f"node_{sp[1]}_{sp[2]}"
                dst = f"node_{dp[1]}_{dp[2]}"
                if dst == ntag:
                    return shapes.get(src)
        return None

    seed = _input_shape()

    for node in ordered:
        label = node.block_label
        up    = _upstream_shape(node.ntag)

        if label == "Input":
            shapes[node.ntag] = seed
            continue

        if label in ("ReLU", "Sigmoid", "Tanh", "GELU", "LeakyReLU",
                     "Softmax", "Dropout", "BatchNorm2D", "LayerNorm",
                     "GroupNorm", "Output"):
            shapes[node.ntag] = up
            continue

        if label == "Flatten":
            if up and len(up) == 3:
                c, h, w = up
                shapes[node.ntag] = (c * h * w,)
            elif up and len(up) == 1:
                shapes[node.ntag] = up
            else:
                shapes[node.ntag] = None
            continue

        if label == "Linear":
            out = _safe_int(_get_field(node.ntag, "out_features"))
            shapes[node.ntag] = (out,) if out > 0 else None
            continue

        if label in ("Conv2D", "ConvTranspose2D"):
            out_c = _safe_int(_get_field(node.ntag, "out_channels"))
            k = _safe_int(_get_field(node.ntag, "kernel_size"), 3)
            s = _safe_int(_get_field(node.ntag, "stride"), 1) or 1
            p = _safe_int(_get_field(node.ntag, "padding"), 0)
            if up and len(up) == 3 and out_c > 0:
                _, h, w = up
                if label == "Conv2D":
                    nh = (h + 2*p - k) // s + 1
                    nw = (w + 2*p - k) // s + 1
                else:
                    nh = (h - 1)*s - 2*p + k
                    nw = (w - 1)*s - 2*p + k
                shapes[node.ntag] = (out_c, max(nh, 1), max(nw, 1))
            else:
                shapes[node.ntag] = (out_c,) if out_c > 0 else None
            continue

        if label in ("MaxPool2D", "AvgPool2D"):
            k = _safe_int(_get_field(node.ntag, "kernel_size"), 2)
            s = _safe_int(_get_field(node.ntag, "stride"), k) or k
            p = _safe_int(_get_field(node.ntag, "padding"), 0)
            if up and len(up) == 3:
                c, h, w = up
                nh = (h + 2*p - k) // s + 1
                nw = (w + 2*p - k) // s + 1
                shapes[node.ntag] = (c, max(nh, 1), max(nw, 1))
            else:
                shapes[node.ntag] = up
            continue

        if label == "AdaptiveAvgPool2D":
            raw = _get_field(node.ntag, "output_size").replace(" ", "")
            parts = [p for p in raw.split(",") if p]
            if up and len(up) == 3 and len(parts) >= 1:
                c = up[0]
                oh = _safe_int(parts[0])
                ow = _safe_int(parts[1]) if len(parts) > 1 else oh
                shapes[node.ntag] = (c, oh, ow) if oh > 0 else None
            else:
                shapes[node.ntag] = up
            continue

        shapes[node.ntag] = up

    return shapes


def infer_from_dataset() -> None:
    """
    Infer and populate model input/output shapes from selected dataset.
    """
    data_tab  = _get_tab_by_role("data_prep")
    model_tab = _get_tab_by_role("model")
    if not data_tab or not model_tab:
        return

    ds_label = None
    for ntag, ni in data_tab["nodes"].items():
        label = ni["label"] if isinstance(ni, dict) else ni
        if label in _DATASET_SHAPES:
            ds_label = label
            break

    if not ds_label:
        return

    input_shape, num_classes = _DATASET_SHAPES[ds_label]

    for ntag, ni in model_tab["nodes"].items():
        label = ni["label"] if isinstance(ni, dict) else ni
        if label == "Input":
            _set_field(ntag, "shape", input_shape)

    if num_classes > 0:
        for ntag, ni in model_tab["nodes"].items():
            label = ni["label"] if isinstance(ni, dict) else ni
            if label == "Output":
                _set_field(ntag, "shape", str(num_classes))


def propagate_all(tab: dict) -> None:
    """
    Perform full parameter propagation across the model graph.
    """
    if not tab:
        return

    shapes = _simulate_shapes(tab)

    try:
        from ml_D_D.engine.graph import topological_sort
        ordered = topological_sort(tab)
    except Exception:
        return

    for node in ordered:
        label = node.block_label

        out_param = _OUT_PARAM.get(label)
        if out_param:
            out_val = _get_field(node.ntag, out_param)
            if out_val:
                for _, (a1, a2) in tab["links"].items():
                    sp = a1.split("_"); dp = a2.split("_")
                    if len(sp) >= 3 and len(dp) >= 3:
                        src = f"node_{sp[1]}_{sp[2]}"
                        dst = f"node_{dp[1]}_{dp[2]}"
                        if src == node.ntag:
                            dst_label = _block_label(tab, dst)
                            in_param  = _IN_PARAM.get(dst_label)
                            if in_param:
                                _set_field(dst, in_param, out_val)

        if label == "Flatten":
            flat_shape = shapes.get(node.ntag)
            if flat_shape and len(flat_shape) == 1 and flat_shape[0] > 0:
                flat_size = str(flat_shape[0])
                for _, (a1, a2) in tab["links"].items():
                    sp = a1.split("_"); dp = a2.split("_")
                    if len(sp) >= 3 and len(dp) >= 3:
                        src = f"node_{sp[1]}_{sp[2]}"
                        dst = f"node_{dp[1]}_{dp[2]}"
                        if src == node.ntag:
                            dst_label = _block_label(tab, dst)
                            if dst_label == "Linear":
                                _set_field(dst, "in_features", flat_size)


def propagate_from_link(tab: dict, src_ntag: str, dst_ntag: str) -> None:
    """
    Trigger parameter propagation after a new link is created.
    """
    propagate_all(tab)


_mismatch_themes: dict[str, int] = {}


def check_dimension_mismatches(tab: dict) -> None:
    """
    Detect and visually highlight dimension mismatches between connected nodes.
    """
    global _mismatch_themes
    if not tab:
        return

    for ntag, th in _mismatch_themes.items():
        if dpg.does_item_exist(ntag):
            ni = tab["nodes"].get(ntag)
            if ni:
                original = ni.get("theme", 0) if isinstance(ni, dict) else 0
                dpg.bind_item_theme(ntag, original if original and dpg.does_item_exist(original) else 0)
    _mismatch_themes = {}

    shapes = _simulate_shapes(tab)
    mismatched: set[str] = set()

    for _, (a1, a2) in tab["links"].items():
        sp = a1.split("_"); dp = a2.split("_")
        if len(sp) < 3 or len(dp) < 3:
            continue
        src_n = f"node_{sp[1]}_{sp[2]}"
        dst_n = f"node_{dp[1]}_{dp[2]}"

        src_label = _block_label(tab, src_n)
        dst_label = _block_label(tab, dst_n)

        out_param = _OUT_PARAM.get(src_label)
        in_param  = _IN_PARAM.get(dst_label)
        if out_param and in_param:
            out_val = _get_field(src_n, out_param).strip()
            in_val  = _get_field(dst_n, in_param).strip()
            if out_val and in_val and out_val != in_val:
                mismatched.add(dst_n)

        if src_label == "Flatten" and dst_label == "Linear":
            flat_shape = shapes.get(src_n)
            if flat_shape and len(flat_shape) == 1 and flat_shape[0] > 0:
                expected = str(flat_shape[0])
                actual   = _get_field(dst_n, "in_features").strip()
                if actual and actual != expected:
                    mismatched.add(dst_n)

    for ntag in mismatched:
        if not dpg.does_item_exist(ntag):
            continue
        with dpg.theme() as th:
            with dpg.theme_component(dpg.mvNode):
                dpg.add_theme_color(dpg.mvNodeCol_NodeOutline, (220, 120, 40), category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBar, (120, 60, 20), category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered, (150, 80, 30), category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected, (180, 100, 40), category=dpg.mvThemeCat_Nodes)
        dpg.bind_item_theme(ntag, th)
        _mismatch_themes[ntag] = th


def on_link_made(tab: dict, src_ntag: str, dst_ntag: str) -> None:
    """
    Handle logic triggered when a new link is created in the graph.
    """
    if tab.get("role") != "model":
        return
    propagate_from_link(tab, src_ntag, dst_ntag)
    check_dimension_mismatches(tab)


def on_param_changed(tab: dict) -> None:
    """
    Handle updates when node parameters are modified.
    """
    if tab.get("role") != "model":
        return
    propagate_all(tab)
    check_dimension_mismatches(tab)


def on_node_spawned(tab: dict) -> None:
    """
    Handle initialization logic when a new node is added.
    """
    infer_from_dataset()
    if tab.get("role") == "model":
        propagate_all(tab)
        check_dimension_mismatches(tab)


def on_dataset_changed() -> None:
    """
    Handle updates when the dataset selection changes.
    """
    infer_from_dataset()