"""
ui/resize.py
Viewport resize callback.
"""

import dearpygui.dearpygui as dpg

from ml_forge.constants import (
    PALETTE_W, TRAIN_W,
    CONSOLE_H, TOOLBAR_H, STATUSBAR_H, MENUBAR_H,
)

PIPELINE_BAR_H = 24


def resize_callback() -> None:
    vw    = dpg.get_viewport_client_width()
    vh    = dpg.get_viewport_client_height()
    mid_h = vh - CONSOLE_H - TOOLBAR_H - STATUSBAR_H - MENUBAR_H - PIPELINE_BAR_H - 26
    canvas_w = vw - PALETTE_W - TRAIN_W - 24

    dpg.set_item_width("main_window",    vw)
    dpg.set_item_height("main_window",   vh)
    dpg.set_item_width("toolbar",        vw - 8)
    dpg.set_item_width("pipeline_bar",   vw - 8)

    dpg.set_item_height("palette_panel", mid_h)
    dpg.set_item_height("train_panel",   mid_h)
    dpg.set_item_height("canvas_panel",  mid_h)
    dpg.set_item_width("canvas_panel",   canvas_w)

    dpg.set_item_height("console_window", CONSOLE_H - 10)
    dpg.set_item_width("console_window",  vw - 16)
    dpg.set_item_width("statusbar",       vw - 8)
