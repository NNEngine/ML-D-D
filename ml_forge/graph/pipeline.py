"""
graph/pipeline.py
Pipeline stage definitions and completion detection.
"""

import dearpygui.dearpygui as dpg
import ml_forge.state as state
from ml_forge.engine.blocks import get_block_def

ROLES = {
    "data_prep": {"label": "Data Prep", "color": (160, 100, 255), "dim": (80, 50, 130),  "description": "Datasets, loaders, augmentation"},
    "model":     {"label": "Model",     "color": (100, 180, 255), "dim": (50, 90, 130),  "description": "Layers, activations, normalization"},
    "training":  {"label": "Training",  "color": (100, 220, 180), "dim": (50, 110, 90),  "description": "Loss, optimizer, scheduler, metrics"},
}

ROLE_ORDER = ["data_prep", "model", "training"]


def _tab_complete(t: dict) -> bool:
    if not t["nodes"]:
        return False
    for ntag, node_info in t["nodes"].items():
        block_label = node_info["label"] if isinstance(node_info, dict) else node_info
        block = get_block_def(block_label)
        if not block or not block["params"]:
            continue
        parts = ntag.split("_")
        tid_s, nid_s = parts[1], parts[2]
        any_filled = False
        for param in block["params"]:
            ftag = f"node_{tid_s}_{nid_s}_input_{param}"
            if dpg.does_item_exist(ftag) and dpg.get_value(ftag).strip():
                any_filled = True
                break
        if not any_filled:
            return False
    return True


def get_stage_statuses() -> list[dict]:
    role_tab: dict[str, dict | None] = {r: None for r in ROLE_ORDER}
    for t in state.tabs.values():
        role = t.get("role")
        if role in role_tab:
            role_tab[role] = t

    results = []
    for role in ROLE_ORDER:
        info = ROLES[role]
        t    = role_tab[role]
        if t is None:
            status = "unassigned"
        elif not t["nodes"]:
            status = "empty"
        elif _tab_complete(t):
            status = "complete"
        else:
            status = "partial"
        results.append({"role": role, "label": info["label"], "color": info["color"], "status": status})
    return results


def pipeline_ready() -> bool:
    return all(s["status"] == "complete" for s in get_stage_statuses())


_last_pipeline_state = None


def refresh_pipeline_bar() -> None:
    global _last_pipeline_state

    if not dpg.does_item_exist("pipeline_bar_content"):
        return

    statuses  = get_stage_statuses()
    any_empty = any(s["status"] in ("empty", "unassigned") for s in statuses)

    errors = warnings = 0
    if not any_empty:
        try:
            from ml_forge.engine.graph import validate_pipeline
            result   = validate_pipeline()
            errors   = len(result.errors)
            warnings = len(result.warnings)
        except Exception:
            errors = 1

    summary_key = (errors, warnings, any_empty)
    if summary_key == _last_pipeline_state:
        return
    _last_pipeline_state = summary_key

    dpg.delete_item("pipeline_bar_content", children_only=True)

    if any_empty:
        col = (120, 120, 120)
        msg = "Pipeline incomplete - fill all three tabs to train"
    elif errors > 0:
        col = (220, 80, 80)
        msg = f"Pipeline error ({errors} issue(s)) - check nodes before training"
    elif warnings > 0:
        col = (220, 180, 60)
        msg = f"Pipeline ready - {warnings} warning(s)"
    else:
        col = (80, 220, 120)
        msg = "Pipeline ready"

    dpg.add_text(msg, color=col, parent="pipeline_bar_content")