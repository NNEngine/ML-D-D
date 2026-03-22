"""
undo.py
Per-tab undo / redo stack using snapshots of node and link state.

Snapshots capture: block label, canvas position, and all param values.
This means undo/redo fully restores a node including everything typed into it.
"""

import copy
import dearpygui.dearpygui as dpg

import ml_forge.state as state
from ml_forge.constants import MAX_UNDO
from ml_forge.ui.console import log


# Snapshot helpers

def _read_node_params(ntag: str, block_label: str) -> dict[str, str]:
    """Read current param field values for a node."""
    from ml_forge.engine.blocks import get_block_def
    block = get_block_def(block_label)
    if not block:
        return {}
    parts = ntag.split("_")   # node_{tid}_{nid}
    params = {}
    for param in block["params"]:
        ftag = f"node_{parts[1]}_{parts[2]}_input_{param}"
        params[param] = dpg.get_value(ftag) if dpg.does_item_exist(ftag) else ""
    return params


def _snapshot(tid: int) -> dict:
    """Return a snapshot of nodes (with params and positions) and links."""
    t = state.tabs[tid]
    nodes_snap = {}
    for ntag, info in t["nodes"].items():
        label = info["label"] if isinstance(info, dict) else info
        pos   = [0, 0]
        if dpg.does_item_exist(ntag):
            raw = dpg.get_item_pos(ntag)
            pos = [int(raw[0]), int(raw[1])]
        params = _read_node_params(ntag, label)
        nodes_snap[ntag] = {"label": label, "pos": pos, "params": params}
    return {
        "nodes": nodes_snap,
        "links": copy.deepcopy(t["links"]),
    }


def _apply_snapshot(tid: int, snap: dict) -> None:
    """Wipe the canvas for tab `tid` and rebuild from snap."""
    from ml_forge.graph.nodes import raw_delete_node, raw_spawn_node
    from ml_forge.ui.statusbar import refresh_status

    t      = state.tabs[tid]
    editor = t["editor_tag"]

    for ntag in list(t["nodes"].keys()):
        raw_delete_node(tid, ntag)
    t["nodes"] = {}
    t["links"] = {}

    for ntag, data in snap["nodes"].items():
        nid    = int(ntag.split("_")[2])
        label  = data["label"]
        pos    = tuple(data.get("pos", [0, 0]))
        params = data.get("params", {})
        raw_spawn_node(tid, label, nid=nid, pos=pos, params=params)

    for link_tag, (a1, a2) in snap["links"].items():
        if dpg.does_item_exist(a1) and dpg.does_item_exist(a2):
            dpg.add_node_link(a1, a2, parent=editor, tag=link_tag)
            t["links"][link_tag] = (a1, a2)

    refresh_status()


# Public

def push_undo(tid: int) -> None:
    """
    Save a snapshot of tab `tid` onto its undo stack and clear its redo stack.
    Call this BEFORE any destructive operation.
    """
    t = state.tabs.get(tid)
    if t is None:
        return
    t["undo_stack"].append(_snapshot(tid))
    if len(t["undo_stack"]) > MAX_UNDO:
        t["undo_stack"].pop(0)
    t["redo_stack"].clear()
    refresh_undo_menu()


def undo() -> None:
    """Undo the last operation on the active tab."""
    tid = state.active_tab_id
    t   = state.tabs.get(tid)
    if not t or not t["undo_stack"]:
        log("Nothing to undo.", "warning")
        return
    t["redo_stack"].append(_snapshot(tid))
    snap = t["undo_stack"].pop()
    _apply_snapshot(tid, snap)
    refresh_undo_menu()
    log("Undo.", "debug")


def redo() -> None:
    """Redo the last undone operation on the active tab."""
    tid = state.active_tab_id
    t   = state.tabs.get(tid)
    if not t or not t["redo_stack"]:
        log("Nothing to redo.", "warning")
        return
    t["undo_stack"].append(_snapshot(tid))
    snap = t["redo_stack"].pop()
    _apply_snapshot(tid, snap)
    refresh_undo_menu()
    log("Redo.", "debug")


def refresh_undo_menu() -> None:
    """Grey out / enable the Edit > Undo and Edit > Redo menu items."""
    t        = state.tabs.get(state.active_tab_id)
    can_undo = bool(t and t["undo_stack"])
    can_redo = bool(t and t["redo_stack"])
    if dpg.does_item_exist("menu_undo"):
        dpg.configure_item("menu_undo", enabled=can_undo)
    if dpg.does_item_exist("menu_redo"):
        dpg.configure_item("menu_redo", enabled=can_redo)