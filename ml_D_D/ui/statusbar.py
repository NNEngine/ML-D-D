"""
ui/statusbar.py

Utilities for keeping the status bar and window title in sync with the
current application state. `refresh_status()` is triggered by nodes, tabs,
and undo operations whenever counts or history depths change.
"""

import pathlib
import dearpygui.dearpygui as dpg
import ml_D_D.state as state


def refresh_status() -> None:
    """
    Refresh status bar metrics and synchronize the window title.

    This function aggregates information from the active tab and updates
    the following UI elements (if they exist):
        - Node and link counts
        - Undo/redo stack depths

    It also delegates to ``_update_title`` to keep the viewport title and
    footer project name consistent with the currently opened project.

    Behavior:
        - Safely handles missing/invalid active tab
        - Avoids UI errors by checking item existence before updates

    Returns:
        None
    """
    t       = state.tabs.get(state.active_tab_id)
    n_nodes = len(t["nodes"])      if t else 0
    n_links = len(t["links"])      if t else 0
    n_undo  = len(t["undo_stack"]) if t else 0
    n_redo  = len(t["redo_stack"]) if t else 0

    if dpg.does_item_exist("status_nodes"):
        dpg.set_value("status_nodes", f"Nodes: {n_nodes}   Links: {n_links}")
    if dpg.does_item_exist("status_undo"):
        dpg.set_value("status_undo", f"Undo: {n_undo}  Redo: {n_redo}")

    _update_title()


def _update_title() -> None:
    """
    Update the application window title and footer project label.

    Determines the current project name from ``state.current_file`` and:
        - Sets the viewport title accordingly
        - Updates the footer display (``status_project``) if present

    Behavior:
        - Uses file stem (name without extension) for display
        - Falls back to a default "untitled" label when no project is open
        - Applies a generic application title when no file is loaded

    Returns:
        None
    """
    path = getattr(state, "current_file", None)
    if path:
        name = pathlib.Path(path).stem
        title = f"ML D&D - {name}"
    else:
        name  = "untitled"
        title = "ML Forge"

    dpg.set_viewport_title(title)

    if dpg.does_item_exist("status_project"):
        dpg.set_value("status_project", name)
