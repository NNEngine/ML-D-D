"""
graph/links.py
DearPyGui node-editor link and delink callbacks.
"""

import dearpygui.dearpygui as dpg
import ml_forge.state as state


def link_callback(sender, app_data) -> None:
    from ml_forge.graph.undo import push_undo

    tid = state.active_tab_id
    t   = state.tabs.get(tid)
    if t is None:
        return

    push_undo(tid)
    t["link_counter"] += 1
    link_tag = f"link_{tid}_{t['link_counter']}"
    a1, a2   = app_data[0], app_data[1]

    if isinstance(a1, int):
        a1 = dpg.get_item_alias(a1) or str(a1)
    if isinstance(a2, int):
        a2 = dpg.get_item_alias(a2) or str(a2)

    dpg.add_node_link(a1, a2, parent=sender, tag=link_tag)
    t["links"][link_tag] = (a1, a2)

    # Auto-fill downstream dimensions and check for mismatches
    try:
        sp = a1.split("_"); dp = a2.split("_")
        if len(sp) >= 3 and len(dp) >= 3:
            src_ntag = f"node_{sp[1]}_{sp[2]}"
            dst_ntag = f"node_{dp[1]}_{dp[2]}"
            from ml_forge.engine.autofill import on_link_made
            on_link_made(t, src_ntag, dst_ntag)
    except Exception:
        pass


def delink_callback(sender, app_data) -> None:
    from ml_forge.graph.undo import push_undo

    tid = state.active_tab_id
    t   = state.tabs.get(tid)
    if t is None:
        return

    push_undo(tid)
    link_tag = app_data
    if dpg.does_item_exist(link_tag):
        dpg.delete_item(link_tag)
    t["links"].pop(link_tag, None)