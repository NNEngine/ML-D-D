"""
nodes.py

Node creation and deletion utilities for the graph editor.

This module provides both raw operations (no undo tracking) and public
operations (with undo integration). Raw functions are used internally by
state restoration and persistence layers, while public functions are
invoked through the UI and always record undo snapshots.
"""

import dearpygui.dearpygui as dpg

import ml_D_D.state as state
from ml_D_D.engine.blocks import get_block_def
from ml_D_D.constants import NODE_GRID_COLS, NODE_GRID_X_STEP, NODE_GRID_Y_STEP, NODE_GRID_ORIGIN


def node_tag(tid: int, nid: int) -> str:
    """
    Generate a unique node tag identifier.

    Args:
        tid (int): Tab ID.
        nid (int): Node ID.

    Returns:
        str: Formatted node tag string.
    """
    return f"node_{tid}_{nid}"


def attr_in_tag(tid: int, nid: int, pin: str) -> str:
    """
    Generate a tag for an input attribute of a node.

    Args:
        tid (int): Tab ID.
        nid (int): Node ID.
        pin (str): Input pin name.

    Returns:
        str: Input attribute tag.
    """
    return f"node_{tid}_{nid}_in_{pin}"


def attr_out_tag(tid: int, nid: int, pin: str) -> str:
    """
    Generate a tag for an output attribute of a node.

    Args:
        tid (int): Tab ID.
        nid (int): Node ID.
        pin (str): Output pin name.

    Returns:
        str: Output attribute tag.
    """
    return f"node_{tid}_{nid}_out_{pin}"


def attr_param_tag(tid: int, nid: int, param: str) -> str:
    """
    Generate a tag for a parameter attribute of a node.

    Args:
        tid (int): Tab ID.
        nid (int): Node ID.
        param (str): Parameter name.

    Returns:
        str: Parameter attribute tag.
    """
    return f"node_{tid}_{nid}_param_{param}"


def input_field_tag(tid: int, nid: int, param: str) -> str:
    """
    Generate a tag for an input field associated with a node parameter.

    Args:
        tid (int): Tab ID.
        nid (int): Node ID.
        param (str): Parameter name.

    Returns:
        str: Input field tag.
    """
    return f"node_{tid}_{nid}_input_{param}"


def raw_spawn_node(tid: int, block_label: str, nid: int | None = None,
                   pos: tuple | None = None,
                   params: dict | None = None) -> str | None:
    """
    Create a node in the specified tab without recording undo state.

    This function is used internally for operations such as undo/redo,
    project loading, and state restoration.

    Behavior:
        - Assigns a unique node ID if not provided
        - Applies grid-based positioning if no position is specified
        - Builds node UI elements (inputs, parameters, outputs)
        - Applies theme styling based on block definition

    Args:
        tid (int): Target tab ID.
        block_label (str): Block type identifier.
        nid (int, optional): Explicit node ID.
        pos (tuple, optional): (x, y) position in canvas.
        params (dict, optional): Parameter values for initialization.

    Returns:
        str | None: Node tag if created successfully, otherwise None.
    """
    t = state.tabs[tid]

    RESERVED_NID_THRESHOLD = 9000

    if nid is None:
        t["node_counter"] += 1
        nid = t["node_counter"]
    else:
        if nid < RESERVED_NID_THRESHOLD:
            t["node_counter"] = max(t["node_counter"], nid)

    block = get_block_def(block_label)
    if block is None:
        return None

    color = block["color"]
    ntag  = node_tag(tid, nid)

    if pos is not None:
        pos_x, pos_y = pos
    elif nid < RESERVED_NID_THRESHOLD:
        col   = nid - 1
        pos_x = NODE_GRID_ORIGIN[0] + (col % NODE_GRID_COLS) * NODE_GRID_X_STEP
        pos_y = NODE_GRID_ORIGIN[1] + (col // NODE_GRID_COLS) * NODE_GRID_Y_STEP
    else:
        pos_x, pos_y = NODE_GRID_ORIGIN

    with dpg.node(label=block_label, tag=ntag,
                  parent=t["editor_tag"], pos=(pos_x, pos_y)):

        with dpg.theme() as nth:
            with dpg.theme_component(dpg.mvNode):
                dpg.add_theme_color(dpg.mvNodeCol_TitleBar,
                                    color, category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered,
                                    tuple(min(c + 30, 255) for c in color),
                                    category=dpg.mvThemeCat_Nodes)
                dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected,
                                    tuple(min(c + 50, 255) for c in color),
                                    category=dpg.mvThemeCat_Nodes)
        dpg.bind_item_theme(ntag, nth)

        for pin in block["inputs"]:
            with dpg.node_attribute(label=pin,
                                    attribute_type=dpg.mvNode_Attr_Input,
                                    tag=attr_in_tag(tid, nid, pin)):
                dpg.add_text(pin, color=(180, 180, 180))

        for param in block["params"]:
            default = (params or {}).get(param, "")
            with dpg.node_attribute(label=param,
                                    attribute_type=dpg.mvNode_Attr_Static,
                                    tag=attr_param_tag(tid, nid, param)):
                dpg.add_input_text(label=param, default_value=default,
                                   width=110, hint=param,
                                   tag=input_field_tag(tid, nid, param))

        for pin in block["outputs"]:
            with dpg.node_attribute(label=pin,
                                    attribute_type=dpg.mvNode_Attr_Output,
                                    tag=attr_out_tag(tid, nid, pin)):
                dpg.add_text(pin, color=(180, 180, 180))

    t["nodes"][ntag] = {"label": block_label, "theme": nth}
    return ntag


def raw_delete_node(tid: int, ntag: str) -> None:
    """
    Remove a node and its associated links without recording undo state.

    Args:
        tid (int): Target tab ID.
        ntag (str): Node tag identifier.

    Returns:
        None
    """
    t = state.tabs[tid]
    if dpg.does_item_exist(ntag):
        raw_attrs  = dpg.get_item_children(ntag, slot=1) or []
        attr_aliases = set()
        for attr_id in raw_attrs:
            alias = dpg.get_item_alias(attr_id) if isinstance(attr_id, int) else attr_id
            if alias:
                attr_aliases.add(alias)
            attr_aliases.add(str(attr_id))

        for link_tag, (a1, a2) in list(t["links"].items()):
            s1 = str(a1) if isinstance(a1, int) else a1
            s2 = str(a2) if isinstance(a2, int) else a2
            if s1 in attr_aliases or s2 in attr_aliases:
                if dpg.does_item_exist(link_tag):
                    dpg.delete_item(link_tag)
                t["links"].pop(link_tag, None)

        dpg.delete_item(ntag)
    t["nodes"].pop(ntag, None)


# Public API

def spawn_node(block_label: str) -> None:
    """
    Create a node in the active tab with undo tracking.

    This is the primary UI-facing function for node creation.

    Args:
        block_label (str): Block type to instantiate.

    Returns:
        None
    """
    from ml_D_D.graph.undo   import push_undo
    from ml_D_D.graph.tabs   import current_tab, _remove_hint_node
    from ml_D_D.ui.statusbar import refresh_status

    t = current_tab()
    if t is None:
        return
    push_undo(state.active_tab_id)
    _remove_hint_node(state.active_tab_id)
    raw_spawn_node(state.active_tab_id, block_label)
    refresh_status()
    _maybe_refresh_summary()

    try:
        from ml_D_D.engine.autofill import on_node_spawned
        on_node_spawned(t)
    except Exception:
        pass


def delete_node(ntag: str) -> None:
    """
    Delete a specific node with undo support.

    Args:
        ntag (str): Node tag identifier.

    Returns:
        None
    """
    from ml_D_D.graph.undo import push_undo
    from ml_D_D.ui.statusbar import refresh_status

    push_undo(state.active_tab_id)
    raw_delete_node(state.active_tab_id, ntag)
    refresh_status()
    _maybe_refresh_summary()


def delete_selected_nodes() -> None:
    """
    Delete all currently selected nodes in the active tab.

    Returns:
        None
    """
    from ml_D_D.graph.undo import push_undo
    from ml_D_D.graph.tabs import current_tab
    from ml_D_D.ui.statusbar import refresh_status

    t = current_tab()
    if t is None:
        return
    selected = dpg.get_selected_nodes(t["editor_tag"])
    if not selected:
        return
    push_undo(state.active_tab_id)
    for nid in selected:
        alias = dpg.get_item_alias(nid)
        raw_delete_node(state.active_tab_id, alias if alias else str(nid))
    refresh_status()
    _maybe_refresh_summary()


def clear_canvas() -> None:
    """
    Remove all nodes from the active tab.

    This operation records an undo snapshot before clearing.

    Returns:
        None
    """
    from ml_D_D.graph.undo import push_undo
    from ml_D_D.graph.tabs import current_tab
    from ml_D_D.ui.statusbar import refresh_status

    t = current_tab()
    if t is None:
        return
    push_undo(state.active_tab_id)
    for ntag in list(t["nodes"].keys()):
        raw_delete_node(state.active_tab_id, ntag)
    refresh_status()
    _maybe_refresh_summary()


def _maybe_refresh_summary() -> None:
    """
    Refresh the model summary panel when applicable.

    Triggered after node mutations to ensure the summary reflects
    the current graph state.

    Returns:
        None
    """
    t = state.tabs.get(state.active_tab_id)
    if t and t.get("role") == "model":
        try:
            from ml_D_D.ui.summary import refresh_model_summary
            refresh_model_summary()
        except Exception:
            pass
