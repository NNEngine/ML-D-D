"""
engine/training_setup.py
Manages the auto-spawned ModelBlock and DataLoaderBlock nodes in the Training tab.
"""

from __future__ import annotations

import dearpygui.dearpygui as dpg

import ml_forge.state as state
from ml_forge.ui.console import log

_MODEL_POS      = (40,  60)
_DATALOADER_POS = (40, 260)
_MODEL_NID      = 9901
_DATALOADER_NID = 9902


def _get_training_tab() -> dict | None:
    for t in state.tabs.values():
        if t.get("role") == "training":
            return t
    return None


def _tid_of(tab: dict) -> int | None:
    for tid, t in state.tabs.items():
        if t is tab:
            return tid
    return None


def _ntag(tid: int, nid: int) -> str:
    return f"node_{tid}_{nid}"


def ensure_pipeline_inputs() -> None:
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
    pass


def reset_block_labels() -> None:
    pass