"""
ui/toolbar.py
The thin toolbar strip that sits above the canvas panel.
"""

import dearpygui.dearpygui as dpg

import ml_forge.state as state
from ml_forge.constants import TOOLBAR_H


def build_toolbar() -> None:
    from ml_forge.graph.nodes import clear_canvas
    from ml_forge.graph.tabs  import new_tab, close_tab

    with dpg.child_window(tag="toolbar", height=TOOLBAR_H,
                           border=False, no_scrollbar=True):
        with dpg.group(horizontal=True):
            dpg.add_text("Canvas:", color=(120, 120, 120))
            dpg.add_spacer(width=4)

            dpg.add_button(label="Clear",      small=True, callback=clear_canvas)

            dpg.add_separator()
            dpg.add_spacer(width=4)

            dpg.add_button(label="+ New Tab",   small=True,
                           callback=lambda: new_tab())
            dpg.add_button(label="x Close Tab", small=True,
                           callback=lambda: close_tab(state.active_tab_id))

            dpg.add_separator()
            dpg.add_spacer(width=10)
            dpg.add_separator()
            dpg.add_spacer(width=6)
            dpg.add_text(
                "Drag pins to connect  |  Ctrl+Click link to remove  |  Del to delete  |  Middle-mouse to pan  | Drag to select",
                color=(90, 90, 110),
            )
