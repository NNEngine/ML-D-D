"""
engine/training_setup.py
Manages the auto-spawned ModelBlock and DataLoaderBlock nodes in the Training tab.
"""

from __future__ import annotations

import dearpygui.dearpygui as dpg

import ml_D_D.state as state
from ml_D_D.ui.console import log

_MODEL_POS      = (40,  60)
_DATALOADER_POS = (40, 260)
_MODEL_NID      = 9901
_DATALOADER_NID = 9902


def _get_training_tab() -> dict | None:
    """
    Retrieve the tab configuration corresponding to the Training workspace.

    Iterates through all registered tabs in the global state and returns the one
    whose role is set to "training".

    Returns:
        dict | None: The training tab dictionary if found, otherwise None.
    """

    for t in state.tabs.values():
        if t.get("role") == "training":
            return t
    return None


def _tid_of(tab: dict) -> int | None:
    """
    Find the tab ID corresponding to a given tab object.

    This function performs a reverse lookup in the global state to identify
    the unique tab ID associated with the provided tab dictionary.

    Args:
        tab (dict): The tab object whose ID needs to be determined.

    Returns:
        int | None: The tab ID if found, otherwise None.
    """

    for tid, t in state.tabs.items():
        if t is tab:
            return tid
    return None


def _ntag(tid: int, nid: int) -> str:
    """
    Generate a unique node tag string for a given tab and node ID.

    The tag is used internally by DearPyGui to uniquely identify nodes.

    Args:
        tid (int): The tab ID.
        nid (int): The node ID.

    Returns:
        str: A formatted node tag in the form "node_<tid>_<nid>".
    """

    return f"node_{tid}_{nid}"


def ensure_pipeline_inputs() -> None:
    """
    Ensure that essential pipeline input nodes exist in the Training tab.

    This function guarantees that the following nodes are present:
    - ModelBlock: Represents the model definition input.
    - DataLoaderBlock: Represents the data loading input.

    If these nodes do not exist, they are automatically spawned at predefined
    positions and visually locked (styled) to prevent accidental modification.

    Workflow:
    - Locate the training tab and its ID.
    - Check for existence of required nodes using their unique tags.
    - Spawn missing nodes using `raw_spawn_node`.
    - Apply a locked visual theme to newly created nodes.

    Returns:
        None
    """

    tab = _get_training_tab()
    if tab is None:
        return
    tid = _tid_of(tab)
    if tid is None:
        return

    from ml_forge.graph.nodes import raw_spawn_node

    model_tag  = _ntag(tid, _MODEL_NID)
    loader_tag = _ntag(tid, _DATALOADER_NID)

    if not dpg.does_item_exist(model_tag):
        raw_spawn_node(tid, "ModelBlock", nid=_MODEL_NID, pos=_MODEL_POS)
        _lock_node(model_tag)

    if not dpg.does_item_exist(loader_tag):
        raw_spawn_node(tid, "DataLoaderBlock", nid=_DATALOADER_NID, pos=_DATALOADER_POS)
        _lock_node(loader_tag)


def _lock_node(ntag: str) -> None:
    """
    Apply a visual "locked" theme to a node to distinguish it as non-editable.

    This modifies the node's appearance (e.g., title bar and outline colors)
    using a custom DearPyGui theme, indicating that the node is auto-generated
    and should not be altered by the user.

    Args:
        ntag (str): The unique tag of the node to be styled.

    Returns:
        None
    """
    if not dpg.does_item_exist(ntag):
        return
    with dpg.theme() as th:
        with dpg.theme_component(dpg.mvNode):
            dpg.add_theme_color(dpg.mvNodeCol_TitleBar,         (50, 50, 80),   category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered,  (70, 70, 110),  category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected, (90, 90, 140),  category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_NodeOutline,      (80, 120, 200), category=dpg.mvThemeCat_Nodes)
    dpg.bind_item_theme(ntag, th)


def update_block_labels() -> None:
    """
    Update labels or display information for training pipeline blocks.

    Intended to dynamically refresh node labels (e.g., model name, dataset info,
    configuration summaries) based on the current state of the training pipeline.

    This function is currently a placeholder and should be implemented to:
    - Reflect real-time configuration changes in the UI.
    - Improve clarity of pipeline structure for the user.

    Returns:
        None
    """
    pass


def reset_block_labels() -> None:
    """
    Reset labels of training pipeline blocks to their default state.

    Intended to clear any dynamically updated labels and restore the original
    or default naming for the ModelBlock and DataLoaderBlock nodes.

    This function is currently a placeholder and should be implemented to:
    - Remove custom or derived label information.
    - Revert UI elements to their initial state.

    Returns:
        None
    """
    pass
