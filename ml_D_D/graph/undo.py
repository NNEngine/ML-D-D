"""
undo.py

Per-tab undo / redo stack implementation using full-state snapshots
of nodes and links.

Snapshots capture:
    - Block label
    - Canvas position
    - All parameter values

This ensures undo/redo operations fully restore node state,
including user-entered inputs.
"""

import copy
import dearpygui.dearpygui as dpg

import ml_D_D.state as state
from ml_D_D.constants import MAX_UNDO
from ml_D_D.ui.console import log


# Snapshot helpers

def _read_node_params(ntag: str, block_label: str) -> dict[str, str]:
    """
    Extract current parameter values from a node's input fields.

    Args:
        ntag (str): Node tag identifier.
        block_label (str): Block type associated with the node.

    Returns:
        dict[str, str]: Mapping of parameter names to their current values.
    """
    from ml_D_D.engine.blocks import get_block_def
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
    """
    Capture a complete snapshot of a tab's state.

    The snapshot includes:
        - Node definitions (label, position, parameters)
        - Link connections

    Args:
        tid (int): Tab ID.

    Returns:
        dict: Snapshot containing nodes and links.
    """
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
    """
    Restore a tab's state from a snapshot.

    This function clears the current canvas and reconstructs all nodes
    and links exactly as captured in the snapshot.

    Args:
        tid (int): Tab ID.
        snap (dict): Snapshot data.

    Returns:
        None
    """
    from ml_D_D.graph.nodes import raw_delete_node, raw_spawn_node
    from ml_D_D.ui.statusbar import refresh_status

    t      = state.tabs[tid]
    editor = t["editor_tag"]

    # Clear existing nodes and links
    for ntag in list(t["nodes"].keys()):
        raw_delete_node(tid, ntag)
    t["nodes"] = {}
    t["links"] = {}

    # Recreate nodes
    for ntag, data in snap["nodes"].items():
        nid    = int(ntag.split("_")[2])
        label  = data["label"]
        pos    = tuple(data.get("pos", [0, 0]))
        params = data.get("params", {})
        raw_spawn_node(tid, label, nid=nid, pos=pos, params=params)

    # Recreate links
    for link_tag, (a1, a2) in snap["links"].items():
        if dpg.does_item_exist(a1) and dpg.does_item_exist(a2):
            dpg.add_node_link(a1, a2, parent=editor, tag=link_tag)
            t["links"][link_tag] = (a1, a2)

    refresh_status()


# Public API

def push_undo(tid: int) -> None:
    """
    Record the current state of a tab for undo operations.

    This function:
        - Saves a snapshot to the undo stack
        - Enforces maximum stack size
        - Clears the redo stack

    Args:
        tid (int): Tab ID.

    Returns:
        None
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
    """
    Revert the most recent operation on the active tab.

    Moves the current state to the redo stack and restores
    the previous snapshot.

    Returns:
        None
    """
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
    """
    Reapply the most recently undone operation.

    Moves the current state to the undo stack and restores
    the next snapshot from the redo stack.

    Returns:
        None
    """
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
    """
    Update UI state of Undo and Redo menu items.

    Enables or disables menu entries based on whether undo
    or redo operations are currently available.

    Returns:
        None
    """
    t        = state.tabs.get(state.active_tab_id)
    can_undo = bool(t and t["undo_stack"])
    can_redo = bool(t and t["redo_stack"])

    if dpg.does_item_exist("menu_undo"):
        dpg.configure_item("menu_undo", enabled=can_undo)

    if dpg.does_item_exist("menu_redo"):
        dpg.configure_item("menu_redo", enabled=can_redo)
