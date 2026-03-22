"""
ui/statusbar.py
refresh_status() is called by nodes, tabs, and undo whenever the
node/link counts or undo stack depths change.
"""

import pathlib
import dearpygui.dearpygui as dpg
import ml_forge.state as state


def refresh_status() -> None:
    """Update node/link counts, undo depth, window title, and footer project name."""
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
    """Sync the viewport title and footer project field with current_file."""
    path = getattr(state, "current_file", None)
    if path:
        name = pathlib.Path(path).stem
        title = f"ML Forge - {name}"
    else:
        name  = "untitled"
        title = "ML Forge"

    dpg.set_viewport_title(title)

    if dpg.does_item_exist("status_project"):
        dpg.set_value("status_project", name)