"""
ui/summary.py
Model summary panel.
"""

import dearpygui.dearpygui as dpg

from ml_forge.engine.blocks import get_block_def
from ml_forge.graph.nodes   import input_field_tag


def _safe_int(v) -> int:
    try:    return int(v)
    except: return 0


PARAM_ESTIMATES: dict = {
    "Linear":         lambda p: _safe_int(p.get("in_features",  0)) * _safe_int(p.get("out_features",  0)),
    "Conv2D":         lambda p: _safe_int(p.get("in_channels",  0)) * _safe_int(p.get("out_channels",  0)) * _safe_int(p.get("kernel_size", 3)) ** 2,
    "ConvTranspose2D":lambda p: _safe_int(p.get("in_channels",  0)) * _safe_int(p.get("out_channels",  0)) * _safe_int(p.get("kernel_size", 3)) ** 2,
    "BatchNorm2D":    lambda p: 2 * _safe_int(p.get("num_features",     0)),
    "LayerNorm":      lambda p: 2 * _safe_int(p.get("normalized_shape", 0)),
    "GroupNorm":      lambda p: 2 * _safe_int(p.get("num_channels",     0)),
}


def refresh_model_summary() -> None:
    import ml_forge.state as state
    from ml_forge.graph.tabs import current_tab

    if not dpg.does_item_exist("summary_content"):
        return
    dpg.delete_item("summary_content", children_only=True)

    t = current_tab()
    if not t or not t["nodes"]:
        dpg.add_text("No nodes on canvas.", color=(120, 120, 120), parent="summary_content")
        return

    total_params = 0
    rows: list[tuple[str, int]] = []

    for ntag, node_info in t["nodes"].items():
        block_label = node_info["label"] if isinstance(node_info, dict) else node_info
        estimator = PARAM_ESTIMATES.get(block_label)
        if estimator:
            parts = ntag.split("_")
            tid_s, nid_s = parts[1], parts[2]
            block = get_block_def(block_label)
            vals  = {}
            if block:
                for param in block["params"]:
                    ftag = f"node_{tid_s}_{nid_s}_input_{param}"
                    if dpg.does_item_exist(ftag):
                        vals[param] = dpg.get_value(ftag)
            count = estimator(vals)
            total_params += count
            rows.append((block_label, count))
        else:
            rows.append((block_label, 0))

    dpg.add_text(f"{'Layer':<18} {'Params':>12}", color=(160, 160, 160), parent="summary_content")
    dpg.add_separator(parent="summary_content")

    for label, count in rows:
        col = (200, 200, 200) if count > 0 else (120, 120, 120)
        dpg.add_text(f"{label:<18} {count:>12,}", color=col, parent="summary_content")

    dpg.add_separator(parent="summary_content")
    dpg.add_spacer(height=2, parent="summary_content")
    dpg.add_text(f"{'Total params':<18} {total_params:>12,}", color=(100, 200, 255), parent="summary_content")

    mem_mb = (total_params * 4) / (1024 ** 2)
    mem_str = f"{mem_mb:.1f} MB" if mem_mb < 1024 else f"{mem_mb/1024:.2f} GB"
    dpg.add_text(f"{'Est. mem (fp32)':<18} {mem_str:>12}", color=(180, 180, 120), parent="summary_content")
    dpg.add_spacer(height=4, parent="summary_content")